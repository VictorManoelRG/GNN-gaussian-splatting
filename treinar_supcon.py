#!/usr/bin/env python3
"""
Compara a loss contrastiva atual com uma Supervised Contrastive Loss (SupCon)
vetorizada, mantendo TODO o resto do pipeline idêntico ao `train_gnn copy 11.py`.

Motivação
---------
Na loss atual os termos positivo e negativo percorrem `edge_index`, que é um
grafo de ~3 vizinhos espaciais. Como vizinhos espaciais quase sempre pertencem
ao mesmo objeto, as arestas entre rótulos diferentes ficam confinadas às
fronteiras onde dois objetos se tocam — dois objetos que não se encostam nunca
recebem termo de repulsão. O único contrapeso global é o InfoNCE, que amostra
5 âncoras por rótulo (~410 âncoras por época para mais de 100 mil pontos),
cada uma com 1 positivo e 10 negativos.

A SupCon ataca exatamente isso: sobre um lote aleatório de pontos rotulados,
cada âncora usa TODOS os pontos de mesmo rótulo do lote como positivos e todos
os demais como negativos, numa única matriz de similaridade B×B. O sinal é
independente do grafo, então não depende do campo receptivo da rede.

Modos de `--loss`:
  original  — ColorAwareContrastiveLossV2, exatamente como está hoje
  supcon    — apenas a SupCon global
  hibrido   — termos de aresta (coerência local) + SupCon (separação global)

Exemplo:
  python treinar_supcon.py -m output/teatime --iteration 30000 \\
    --masks_path .../Annotations --deva_json .../pred.json \\
    --loss hibrido --seed 42 --experiment_name hibrido_s42
"""
import os
import sys
import time
import importlib.util
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

_PIPELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_gnn copy 11.py")
_spec = importlib.util.spec_from_file_location("pipeline_gnn", _PIPELINE)
tg = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_gnn"] = tg
_spec.loader.exec_module(tg)


class SupConGrafo(nn.Module):
    """Supervised Contrastive Loss vetorizada, opcionalmente com termos de aresta.

    A parte SupCon segue Khosla et al. (2020): para cada âncora do lote,
    maximiza a similaridade com todos os pontos de mesmo rótulo contra todos
    os demais do lote, de uma vez, numa matriz B×B. Com B=4096 são ~16 milhões
    de similaridades (cerca de 64 MB em float32), o que cabe folgado na GPU e
    sai mais rápido que o laço Python da versão atual.

    O lote é amostrado de forma ESTRATIFICADA (teto de `max_por_classe` pontos
    por rótulo) porque os objetos têm tamanhos muito diferentes — uma amostra
    puramente aleatória seria dominada pelos maiores e daria pouco sinal aos
    pequenos.

    Com `usar_arestas=True`, mantém os termos de dobradiça sobre as arestas do
    grafo, que preservam a coerência local, e soma a SupCon com peso
    `peso_supcon`. Os dois termos são complementares: as arestas cuidam de
    vizinhança, a SupCon cuida de identidade de objeto a qualquer distância.
    """

    def __init__(self, temperature=0.1, usar_arestas=True, peso_supcon=1.0,
                 pos_margin=0.2, neg_margin=0.1, tamanho_lote=4096,
                 max_por_classe=64, seed=42):
        super().__init__()
        self.temperature = temperature
        self.usar_arestas = usar_arestas
        self.peso_supcon = peso_supcon
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.tamanho_lote = tamanho_lote
        self.max_por_classe = max_por_classe
        self.rng = np.random.RandomState(seed)
        self._indices_por_rotulo = None
        # Guardados a cada forward para diagnóstico: os dois termos vivem em
        # escalas diferentes (dobradiça satura em zero, entropia cruzada não),
        # então acompanhar a proporção é essencial para escolher `peso_supcon`.
        self.ultimo_arestas = 0.0
        self.ultimo_supcon = 0.0

    def _preparar_indices(self, labels_np):
        """Agrupa os índices por rótulo uma única vez, não a cada época."""
        self._indices_por_rotulo = [
            np.where(labels_np == c)[0] for c in np.unique(labels_np[labels_np != -1])
        ]

    def _amostrar_lote(self):
        partes = []
        for pool in self._indices_por_rotulo:
            k = min(len(pool), self.max_por_classe)
            partes.append(self.rng.choice(pool, k, replace=False))
        idx = np.concatenate(partes)
        if len(idx) > self.tamanho_lote:
            idx = self.rng.choice(idx, self.tamanho_lote, replace=False)
        return idx

    def forward(self, embeddings, labels, edge_index, colors=None, edge_weights=None):
        emb = F.normalize(embeddings, p=2, dim=1)
        device = emb.device
        perda = torch.tensor(0.0, device=device)

        if self.usar_arestas:
            src, dst = edge_index
            sim = (emb[src] * emb[dst]).sum(dim=1)
            valido = (labels[src] != -1) & (labels[dst] != -1)
            iguais = (labels[src] == labels[dst]) & valido
            diferentes = (labels[src] != labels[dst]) & valido

            if iguais.any():
                bruto = torch.relu(1 - sim[iguais] - self.pos_margin)
                perda = perda + ((bruto * edge_weights[iguais]).mean()
                                 if edge_weights is not None else bruto.mean())
            if diferentes.any():
                bruto = torch.relu(sim[diferentes] - self.neg_margin)
                perda = perda + ((bruto * edge_weights[diferentes]).mean()
                                 if edge_weights is not None else bruto.mean())
            self.ultimo_arestas = float(perda.detach())

        # ---------------- SupCon ----------------
        idx = torch.as_tensor(self._amostrar_lote(), dtype=torch.long, device=device)
        z = emb[idx]
        y = labels[idx]

        logits = (z @ z.T) / self.temperature
        # Estabilidade numérica: subtrair o máximo por linha não altera o
        # softmax, mas evita exp() de valores grandes.
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        fora_da_diagonal = ~torch.eye(len(idx), dtype=torch.bool, device=device)
        exp_logits = torch.exp(logits) * fora_da_diagonal
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        positivos = (y.unsqueeze(0) == y.unsqueeze(1)) & fora_da_diagonal
        n_positivos = positivos.sum(1)
        tem_positivo = n_positivos > 0

        if tem_positivo.any():
            soma = (positivos.float() * log_prob).sum(1)[tem_positivo]
            supcon = -(soma / n_positivos[tem_positivo]).mean()
            self.ultimo_supcon = float(supcon.detach())
            perda = perda + self.peso_supcon * supcon

        return perda


def preparar(args, device):
    """Carrega a cena e aplica propagação de labels + filtros (idem pipeline)."""
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().squeeze(1).numpy()
    opacity = torch.sigmoid(gaussians._opacity).detach().cpu().numpy().squeeze()
    max_scaling = np.max(torch.exp(gaussians._scaling).detach().cpu().numpy(), axis=1)

    labels_deva = tg.build_labels_by_majority_vote(
        scene, xyz, args.masks_path, min_votes=args.min_votes)

    base_mask = labels_deva != -1
    alvo, alvo_atual = args.target_gaussians, args.target_gaussians
    mask_filter = base_mask.copy()

    for it in range(4):
        _, _, _, mask_sel = tg.compute_dynamic_filters(
            opacity[base_mask], max_scaling[base_mask],
            target_count=min(alvo_atual, int(base_mask.sum())))
        candidato = np.zeros_like(base_mask)
        candidato[np.where(base_mask)[0][mask_sel]] = True
        mask_filter = tg.remove_spatial_outliers(xyz, candidato, k=16, std_ratio=2.0)
        n = int(mask_filter.sum())
        print(f"   🔁 Iteração {it + 1}: alvo={alvo:,} | real={n:,}")
        if n == 0 or abs(n - alvo) / alvo <= 0.02:
            break
        alvo_atual = min(int(alvo_atual * alvo / max(n, 1)), int(base_mask.sum()))

    return (gaussians, scene, xyz[mask_filter], rgb[mask_filter],
            labels_deva[mask_filter], mask_filter)


def montar_grafo(xyz, rgb, device):
    xyz_norm = (xyz - np.mean(xyz, axis=0)) / (np.std(xyz) + 1e-8)
    rgb_norm = StandardScaler().fit_transform(rgb) * 3.0
    x = torch.tensor(np.concatenate([xyz_norm, rgb_norm], axis=1), dtype=torch.float)

    nbrs = NearestNeighbors(n_neighbors=4, algorithm="auto").fit(xyz)
    distancias, indices = nbrs.kneighbors(xyz)
    viz_idx, viz_dist = indices[:, 1:], distancias[:, 1:]

    sigma = np.median(viz_dist)
    dist_cor = np.linalg.norm(rgb[:, None, :] - rgb[viz_idx], axis=-1)
    sigma_cor = np.median(dist_cor)
    peso_cor = np.exp(-(dist_cor ** 2) / (2 * sigma_cor ** 2 + 1e-8))
    peso_esp = np.exp(-(viz_dist ** 2) / (2 * sigma ** 2 + 1e-8))

    edge_mask = ((viz_dist <= viz_dist.mean() + viz_dist.std())
                 & (dist_cor <= np.percentile(dist_cor, 75)))
    sem_grau = edge_mask.sum(axis=1) < 2
    edge_mask[sem_grau, :2] = True

    src, k_idx = np.where(edge_mask)
    dst = viz_idx[src, k_idx]
    peso = 0.4 * peso_esp[src, k_idx] + 1.2 * peso_cor[src, k_idx]

    arestas = np.concatenate([np.stack([src, dst], 1), np.stack([dst, src], 1)], axis=0)
    pesos = np.concatenate([peso, peso], axis=0)
    arestas, idx_unico = np.unique(arestas, axis=0, return_index=True)
    pesos = pesos[idx_unico]
    print(f"  Arestas no grafo: {len(arestas):,}")

    return (x, torch.tensor(arestas.T, dtype=torch.long),
            torch.tensor(pesos, dtype=torch.float, device=device), rgb_norm)


def main():
    parser = ArgumentParser(description="Compara a loss atual com SupCon vetorizada")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--masks_path", required=True)
    parser.add_argument("--deva_json", default=None)
    parser.add_argument("--min_votes", default=5, type=int)
    parser.add_argument("--target_gaussians", default=250000, type=int)

    parser.add_argument("--loss", default="hibrido", choices=["original", "supcon", "hibrido"])
    parser.add_argument("--supcon_temperature", default=0.1, type=float)
    parser.add_argument("--supcon_peso", default=1.0, type=float)
    parser.add_argument("--supcon_lote", default=4096, type=int)
    parser.add_argument("--supcon_max_por_classe", default=64, type=int)

    parser.add_argument("--gat_heads", default=8, type=int)
    parser.add_argument("--loss_temperature", default=0.05, type=float)
    parser.add_argument("--loss_pos_margin", default=0.2, type=float)
    parser.add_argument("--loss_neg_margin", default=0.1, type=float)
    parser.add_argument("--loss_color_weight", default=0.0, type=float)

    parser.add_argument("--min_cluster_size", default=1500, type=int)
    parser.add_argument("--hdbscan_min_samples", default=3, type=int)
    parser.add_argument("--epocas", default=250, type=int)

    parser.add_argument("--output", default="exp_res_supcon")
    parser.add_argument("--experiment_name", default="supcon")

    args = get_combined_args(parser)
    print(f"\n{'=' * 80}\n🚀 LOSS: {args.loss}  |  semente: {args.seed}\n{'=' * 80}")

    safe_state(False)
    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    margs = model_params.extract(args)
    for a in ["iteration", "masks_path", "min_votes", "target_gaussians"]:
        setattr(margs, a, getattr(args, a))

    gaussians, scene, xyz, rgb, labels, mask_filter = preparar(margs, device)
    print(f"\n✅ Gaussianas para treino: {len(xyz):,}")

    x, edge_index, pesos_t, rgb_norm = montar_grafo(xyz, rgb, device)
    modelo = tg.GaussianGAT(in_channels=x.shape[1], heads=args.gat_heads).to(device)
    data = Data(x=x, edge_index=edge_index, edge_weight=pesos_t).to(device)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    if args.loss == "original":
        criterio = tg.ColorAwareContrastiveLossV2(
            temperature=args.loss_temperature, pos_margin=args.loss_pos_margin,
            neg_margin=args.loss_neg_margin, color_weight=args.loss_color_weight,
            color_sigma=0.3)
    else:
        criterio = SupConGrafo(
            temperature=args.supcon_temperature,
            usar_arestas=(args.loss == "hibrido"),
            peso_supcon=args.supcon_peso,
            pos_margin=args.loss_pos_margin, neg_margin=args.loss_neg_margin,
            tamanho_lote=args.supcon_lote, max_por_classe=args.supcon_max_por_classe,
            seed=args.seed)
        criterio._preparar_indices(labels)
        print(f"  SupCon: lote={args.supcon_lote}, max/classe={args.supcon_max_por_classe}, "
              f"T={args.supcon_temperature}, arestas={'sim' if args.loss == 'hibrido' else 'não'}")

    labels_t = torch.tensor(labels, device=device)
    cores_t = torch.tensor(rgb_norm, dtype=torch.float, device=device)
    parada = tg.EarlyStopping(patience=20, min_delta=0.0001, verbose=False)

    print("\n🚀 Treinando...")
    t0 = time.time()
    for epoca in range(args.epocas):
        modelo.train(); otimizador.zero_grad()
        saida = modelo(data.x, data.edge_index)
        perda = criterio(F.normalize(saida, dim=1), labels_t, data.edge_index,
                         colors=cores_t, edge_weights=data.edge_weight)
        perda.backward(); otimizador.step()
        if epoca % 25 == 0:
            extra = ""
            if isinstance(criterio, SupConGrafo):
                extra = (f" | arestas: {criterio.ultimo_arestas:.4f}"
                         f" | supcon: {criterio.ultimo_supcon:.4f}"
                         f" (x{args.supcon_peso} = {args.supcon_peso * criterio.ultimo_supcon:.4f})")
            print(f"  Época {epoca:4d} | Loss: {perda.item():.4f}{extra}")
        if parada(perda.item(), modelo):
            print(f"  Parada antecipada na época {epoca}")
            break
    parada.restore_best_model(modelo)
    tempo_treino = time.time() - t0

    modelo.eval()
    with torch.no_grad():
        emb = F.normalize(modelo(data.x, data.edge_index), dim=1).cpu().numpy()
    print(f"⏱️ Treino: {tempo_treino:.1f}s")

    print(f"\n📊 HDBSCAN (mcs={args.min_cluster_size}, min_samples={args.hdbscan_min_samples})...")
    rotulos = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        cluster_selection_method="eom", prediction_data=False).fit(emb).labels_
    rotulos = tg.reassign_noise_knn_vectorized(emb, rotulos, k=25, min_agreement=0.8)

    semi = tg.evaluate_semi_supervised(labels, rotulos)
    val = tg.evaluate_cluster_validity(emb, rotulos, xyz=xyz)
    n_clusters = len(np.unique(rotulos[rotulos != -1]))
    print(f"\n📊 ARI: {semi.get('ARI', float('nan')):.4f} | "
          f"NMI: {semi.get('NMI', float('nan')):.4f} | "
          f"Silhouette: {val.get('Silhouette', float('nan')):.4f} | "
          f"clusters: {n_clusters}")

    destino = os.path.join(args.output, args.experiment_name)
    os.makedirs(destino, exist_ok=True)
    args.output = destino
    pipe = pipeline_params.extract(args)
    background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
    tg.render_clusters(rotulos, "images", gaussians, scene, pipe, background,
                       args, device, mask_filter)
    # `images_coverage` é a saída usada pelo calculate_miou_claude_full_table.py
    # nas comparações anteriores; sem ela as métricas 2D não seriam comparáveis.
    tg.render_clusters_coverage(rotulos, "images", gaussians, scene, pipe,
                                args, device, mask_filter)

    with open(os.path.join(destino, "resumo.txt"), "w") as f:
        f.write(f"loss={args.loss} seed={args.seed} clusters={n_clusters}\n")
        f.write(f"ARI={semi.get('ARI', float('nan')):.4f}\n")
        f.write(f"NMI={semi.get('NMI', float('nan')):.4f}\n")
        f.write(f"Silhouette={val.get('Silhouette', float('nan')):.4f}\n")
        f.write(f"tempo_treino={tempo_treino:.1f}\n")
    print(f"\n✅ Concluído em: {destino}")


if __name__ == "__main__":
    main()
