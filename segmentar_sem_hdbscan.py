#!/usr/bin/env python3
"""
Segmentação a partir dos embeddings da GNN SEM passar pelo HDBSCAN.

Reaproveita todo o pipeline do `train_gnn copy 11.py` (propagação de labels do
SAM/DEVA, filtro dinâmico, construção do grafo e treino da GAT/GCN) e substitui
apenas a etapa final de clusterização. Três modos:

  --modo cor
      Não clusteriza nada. Projeta o embedding de 32 dimensões em 3 via PCA e
      usa o resultado diretamente como RGB de cada Gaussiana. Serve para VER a
      estrutura que a rede aprendeu, sem a interferência das decisões do
      algoritmo de clusterização: regiões que a rede considera semelhantes
      aparecem com cores semelhantes, e fronteiras nítidas indicam que o
      embedding de fato separa aqueles objetos.

  --modo kmeans
      KMeans sobre os embeddings, com o número de clusters fixado por
      --n_clusters. Diferente do HDBSCAN, particiona TODOS os pontos (não
      existe rótulo de ruído) e exige saber quantos objetos se espera.

  --modo agglomerative
      Clusterização hierárquica (Ward). Como o custo é quadrático em memória,
      é ajustada sobre uma amostra (--tamanho_amostra) e os demais pontos são
      atribuídos ao centroide mais próximo.

Por que isso é possível: o HDBSCAN é apenas o último passo do pipeline e opera
sobre `emb`, a saída L2-normalizada da rede. Qualquer coisa que consuma uma
matriz (N, 32) pode substituí-lo.

Exemplo:
  python segmentar_sem_hdbscan.py -m output/teatime --iteration 30000 \\
    --masks_path .../teatime/Annotations --deva_json .../teatime/pred.json \\
    --modo cor --output exp_res_sem_hdbscan --experiment_name teatime_cor
"""
import os
import sys
import time
import importlib.util
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

from torch_geometric.data import Data

from scene import Scene
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

# ---------------------------------------------------------------------------
# Reaproveita o pipeline existente em vez de duplicá-lo. O nome do arquivo tem
# espaços, então precisa ser carregado por caminho.
# ---------------------------------------------------------------------------
_PIPELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_gnn copy 11.py")
_spec = importlib.util.spec_from_file_location("pipeline_gnn", _PIPELINE)
tg = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_gnn"] = tg
_spec.loader.exec_module(tg)


def preparar_gaussianas(args, device):
    """Carrega a cena, propaga os labels do SAM/DEVA e aplica os filtros.

    Reproduz exatamente as etapas do pipeline original até o conjunto de
    Gaussianas usado no treino, para que os resultados sejam comparáveis.
    """
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().squeeze(1).numpy()
    opacity = torch.sigmoid(gaussians._opacity).detach().cpu().numpy().squeeze()
    max_scaling = np.max(torch.exp(gaussians._scaling).detach().cpu().numpy(), axis=1)
    total_raw = len(xyz)

    labels_deva = tg.build_labels_by_majority_vote(
        scene, xyz, args.masks_path, min_votes=args.min_votes)

    base_mask = labels_deva != -1
    if args.use_bbox:
        centro = np.array(args.bbox_center, dtype=np.float64)
        largura = np.array(args.bbox_width, dtype=np.float64)
        dentro = np.all(np.abs(xyz - centro) <= largura / 2.0, axis=1)
        base_mask = base_mask & dentro
        print(f"\n📦 Bounding box: {int(dentro.sum()):,}/{total_raw:,} dentro da caixa | "
              f"com label válido: {int(base_mask.sum()):,}")

    alvo = args.target_gaussians
    alvo_atual = alvo
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

    if mask_filter.sum() == 0:
        raise ValueError("Nenhuma Gaussiana sobreviveu aos filtros — revise --target_gaussians.")

    return (gaussians, scene, xyz[mask_filter], rgb[mask_filter],
            labels_deva[mask_filter], mask_filter, total_raw)


def treinar_rede(args, xyz, rgb, labels, device):
    """Monta o grafo e treina a GNN, devolvendo os embeddings L2-normalizados."""
    xyz_norm = (xyz - np.mean(xyz, axis=0)) / (np.std(xyz) + 1e-8)
    rgb_norm = StandardScaler().fit_transform(rgb) * 3.0
    x = torch.tensor(np.concatenate([xyz_norm, rgb_norm], axis=1), dtype=torch.float)

    print("\n🔗 Construindo grafo geométrico...")
    N_VIZINHOS = 4
    nbrs = NearestNeighbors(n_neighbors=N_VIZINHOS, algorithm="auto").fit(xyz)
    distancias, indices = nbrs.kneighbors(xyz)
    viz_idx, viz_dist = indices[:, 1:], distancias[:, 1:]

    sigma = np.median(viz_dist)
    dist_cor = np.linalg.norm(rgb[:, None, :] - rgb[viz_idx], axis=-1)
    sigma_cor = np.median(dist_cor)
    peso_cor = np.exp(-(dist_cor ** 2) / (2 * sigma_cor ** 2 + 1e-8))
    peso_esp = np.exp(-(viz_dist ** 2) / (2 * sigma ** 2 + 1e-8))

    lim_dist = viz_dist.mean() + viz_dist.std()
    lim_cor = np.percentile(dist_cor, 75)
    edge_mask = (viz_dist <= lim_dist) & (dist_cor <= lim_cor)

    GRAU_MINIMO = 2
    sem_grau = edge_mask.sum(axis=1) < GRAU_MINIMO
    if sem_grau.any():
        edge_mask[sem_grau, :GRAU_MINIMO] = True
        print(f"  ⚠️ {int(sem_grau.sum()):,} nós com grau mínimo forçado")

    src, k_idx = np.where(edge_mask)
    dst = viz_idx[src, k_idx]
    peso = 0.4 * peso_esp[src, k_idx] + 1.2 * peso_cor[src, k_idx]

    arestas = np.concatenate([np.stack([src, dst], 1), np.stack([dst, src], 1)], axis=0)
    pesos = np.concatenate([peso, peso], axis=0)
    arestas, idx_unico = np.unique(arestas, axis=0, return_index=True)
    pesos = pesos[idx_unico]
    print(f"  Arestas no grafo: {len(arestas):,}")

    edge_index = torch.tensor(arestas.T, dtype=torch.long)
    in_dim = x.shape[1]

    if args.model_type_choice == "gcn":
        p_min, p_max = pesos.min(), pesos.max()
        pesos_t = torch.tensor((pesos - p_min) / (p_max - p_min + 1e-8),
                               dtype=torch.float, device=device)
        if args.gcn_depth == "deep":
            modelo = tg.DeepGaussianGCN_ResNet(in_channels=in_dim, hidden_dim=64, out_dim=32).to(device)
        else:
            modelo = tg.GaussianGCNV2(in_channels=in_dim, out_dim=32).to(device)
    else:
        pesos_t = torch.tensor(pesos, dtype=torch.float, device=device)
        modelo = tg.GaussianGAT(in_channels=in_dim, heads=args.gat_heads).to(device)

    data = Data(x=x, edge_index=edge_index, edge_weight=pesos_t).to(device)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = tg.ColorAwareContrastiveLossV2(
        temperature=args.loss_temperature, pos_margin=args.loss_pos_margin,
        neg_margin=args.loss_neg_margin, color_weight=args.loss_color_weight,
        color_sigma=0.3)

    labels_t = torch.tensor(labels, device=device)
    cores_t = torch.tensor(rgb_norm, dtype=torch.float, device=device)
    parada = tg.EarlyStopping(patience=20, min_delta=0.0001, verbose=False)

    print("\n🚀 Treinando a rede...")
    t0 = time.time()
    for epoca in range(250):
        modelo.train()
        otimizador.zero_grad()
        if args.model_type_choice == "gcn":
            saida = modelo(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            saida = modelo(data.x, data.edge_index)
        perda = criterio(F.normalize(saida, dim=1), labels_t, data.edge_index,
                         colors=cores_t, edge_weights=data.edge_weight)
        perda.backward()
        otimizador.step()
        if epoca % 25 == 0:
            print(f"  Época {epoca:4d} | Loss: {perda.item():.4f}")
        if parada(perda.item(), modelo):
            print(f"  Parada antecipada na época {epoca}")
            break
    parada.restore_best_model(modelo)

    modelo.eval()
    with torch.no_grad():
        if args.model_type_choice == "gcn":
            saida = modelo(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            saida = modelo(data.x, data.edge_index)
        emb = F.normalize(saida, dim=1).cpu().numpy()

    print(f"⏱️ Treino: {time.time() - t0:.1f}s | embeddings: {emb.shape}")
    return emb


def embeddings_para_cores(emb):
    """Projeta o embedding em 3 dimensões (PCA) e mapeia para RGB em [0,1].

    Cada eixo é normalizado por percentis 2/98 em vez do mínimo e máximo, para
    que alguns poucos pontos extremos não comprimam todo o resto da paleta numa
    faixa estreita de cor.
    """
    pca = PCA(n_components=3, random_state=0)
    proj = pca.fit_transform(emb)
    print(f"  Variância explicada pelas 3 componentes: "
          f"{100 * pca.explained_variance_ratio_.sum():.1f}% "
          f"({', '.join(f'{v*100:.1f}%' for v in pca.explained_variance_ratio_)})")

    baixo = np.percentile(proj, 2, axis=0)
    alto = np.percentile(proj, 98, axis=0)
    cores = np.clip((proj - baixo) / (alto - baixo + 1e-8), 0.0, 1.0)
    return cores.astype(np.float32)


def renderizar_cores_diretas(cores, gaussians, scene, pipe, background, args,
                             device, mask_filter, nome):
    """Renderiza com uma cor arbitrária por Gaussiana (sem noção de cluster).

    Segue a mesma mecânica do `render_clusters`: substitui o SH de grau 0 pela
    cor desejada, zera os harmônicos de ordem superior (senão a variação por
    ponto de vista se soma à cor) e anula a opacidade de quem está fora do
    filtro. Não há "snap" de cor para uma paleta, justamente porque aqui a cor
    é contínua e não representa um rótulo.
    """
    n_total = gaussians._xyz.shape[0]
    todas = np.zeros((n_total, 3), dtype=np.float32)
    todas[mask_filter] = cores

    original = {k: getattr(gaussians, f"_{k}").data.clone()
                for k in ["opacity", "features_dc", "features_rest"]}

    cores_t = torch.tensor(todas, dtype=torch.float32, device=device)
    gaussians._features_dc.data = (cores_t.unsqueeze(1) - 0.5) / 0.28209
    gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)

    nova_opacidade = original["opacity"].clone()
    fora = ~torch.tensor(mask_filter, device=device, dtype=torch.bool)
    nova_opacidade[fora] = -10.0  # sigmoid(-10) ~ 0
    gaussians._opacity.data = nova_opacidade

    destino = os.path.join(args.output, nome)
    os.makedirs(destino, exist_ok=True)
    print(f"\n🖼️  Renderizando embeddings como cor em: {destino}")

    for view in scene.getTrainCameras():
        img = render(view, gaussians, pipe, background)["render"]
        img_np = np.clip(img.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
        cv2.imwrite(os.path.join(destino, f"{view.image_name}.png"),
                    cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    for k, v in original.items():
        getattr(gaussians, f"_{k}").data = v
    return destino


def salvar_nuvem_colorida(xyz, cores, caminho):
    """Grava a nuvem com cor contínua por ponto, em PLY ASCII."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    rgb255 = (np.clip(cores, 0, 1) * 255).astype(np.uint8)
    with open(caminho, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (px, py, pz), (r, g, b) in zip(xyz, rgb255):
            f.write(f"{px} {py} {pz} {r} {g} {b}\n")
    print(f"☁️  Nuvem salva em: {caminho}")


def agrupar(emb, args):
    """Clusteriza os embeddings sem HDBSCAN. Devolve os rótulos."""
    if args.modo == "kmeans":
        print(f"\n📊 KMeans com {args.n_clusters} clusters...")
        km = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=args.seed)
        return km.fit_predict(emb)

    # Ward é O(n²) em memória: com ~100k pontos seria inviável, então ajusta
    # numa amostra e propaga pelos centroides.
    n = len(emb)
    tam = min(args.tamanho_amostra, n)
    print(f"\n📊 Agglomerative (Ward) com {args.n_clusters} clusters, "
          f"ajustado em {tam:,} de {n:,} pontos...")
    rng = np.random.RandomState(args.seed)
    amostra = rng.choice(n, tam, replace=False)

    rotulos_amostra = AgglomerativeClustering(
        n_clusters=args.n_clusters, linkage="ward").fit_predict(emb[amostra])

    centroides = np.stack([emb[amostra][rotulos_amostra == c].mean(axis=0)
                           for c in range(args.n_clusters)])
    nn = NearestNeighbors(n_neighbors=1).fit(centroides)
    return nn.kneighbors(emb)[1].ravel()


def main():
    parser = ArgumentParser(description="Segmentação a partir dos embeddings, sem HDBSCAN")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--masks_path", required=True)
    parser.add_argument("--deva_json", default=None,
                        help="Aceito por compatibilidade com os scripts existentes; a "
                             "propagação por voto de maioria não usa o JSON.")
    parser.add_argument("--min_votes", default=5, type=int)
    parser.add_argument("--target_gaussians", default=250000, type=int)

    parser.add_argument("--modo", default="cor", choices=["cor", "kmeans", "agglomerative"],
                        help="'cor' não clusteriza: pinta cada Gaussiana com o embedding "
                             "projetado em RGB via PCA.")
    parser.add_argument("--n_clusters", default=30, type=int,
                        help="Número de clusters para kmeans/agglomerative.")
    parser.add_argument("--tamanho_amostra", default=20000, type=int,
                        help="Pontos usados para ajustar o Agglomerative.")

    parser.add_argument("--model_type_choice", default="gat", choices=["gat", "gcn"])
    parser.add_argument("--gcn_depth", default="v2", choices=["v2", "deep"])
    parser.add_argument("--gat_heads", default=8, type=int)
    parser.add_argument("--loss_temperature", default=0.05, type=float)
    parser.add_argument("--loss_pos_margin", default=0.2, type=float)
    parser.add_argument("--loss_neg_margin", default=0.1, type=float)
    parser.add_argument("--loss_color_weight", default=0.0, type=float)

    parser.add_argument("--use_bbox", action="store_true", default=False)
    parser.add_argument("--bbox_center", nargs=3, type=float,
                        default=[0.87771511, 1.94553256, 0.34934092])
    parser.add_argument("--bbox_width", nargs=3, type=float,
                        default=[7.83855629, 6.98039675, 7.24066257])

    parser.add_argument("--output", default="exp_res_sem_hdbscan")
    parser.add_argument("--experiment_name", default="sem_hdbscan")

    args = get_combined_args(parser)
    print(f"\n{'=' * 80}\n🚀 SEGMENTAÇÃO SEM HDBSCAN — modo: {args.modo}\n{'=' * 80}")

    safe_state(False)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo_args = model_params.extract(args)
    modelo_args.iteration = args.iteration
    for atributo in ["masks_path", "min_votes", "target_gaussians", "use_bbox",
                     "bbox_center", "bbox_width"]:
        setattr(modelo_args, atributo, getattr(args, atributo))

    gaussians, scene, xyz, rgb, labels_deva, mask_filter, total_raw = \
        preparar_gaussianas(modelo_args, device)
    print(f"\n✅ Gaussianas para treino: {len(xyz):,} de {total_raw:,}")

    emb = treinar_rede(args, xyz, rgb, labels_deva, device)

    pipe = pipeline_params.extract(args)
    background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)

    # `render_clusters` (importado do pipeline) escreve em `args.output`
    # diretamente, sem conhecer o nome do experimento. Apontar `args.output`
    # para a pasta do experimento mantém tudo junto e evita que rodadas
    # diferentes sobrescrevam as imagens umas das outras.
    destino_base = os.path.join(args.output, args.experiment_name)
    os.makedirs(destino_base, exist_ok=True)
    args.output = destino_base

    if args.modo == "cor":
        print("\n🎨 Projetando embeddings em RGB (PCA), sem clusterização...")
        cores = embeddings_para_cores(emb)
        renderizar_cores_diretas(cores, gaussians, scene, pipe, background, args,
                                 device, mask_filter, "images_embedding")
        salvar_nuvem_colorida(xyz, cores,
                              os.path.join(destino_base, "pointclouds", "embedding.ply"))
        print("\n✅ Concluído. As cores são contínuas: tons parecidos indicam "
              "embeddings parecidos, e fronteiras nítidas indicam que a rede "
              "separa aqueles objetos no espaço latente.")
        return

    rotulos = agrupar(emb, args)
    n_clusters = len(np.unique(rotulos))
    print(f"  Clusters formados: {n_clusters}")

    semi = tg.evaluate_semi_supervised(labels_deva, rotulos)
    validade = tg.evaluate_cluster_validity(emb, rotulos, xyz=xyz)
    print(f"\n📊 ARI: {semi.get('ARI', float('nan')):.4f} | "
          f"NMI: {semi.get('NMI', float('nan')):.4f} | "
          f"Silhouette: {validade.get('Silhouette', float('nan')):.4f} | "
          f"Davies-Bouldin: {validade.get('Davies_Bouldin', float('nan')):.4f}")

    tg.render_clusters(rotulos, "images", gaussians, scene, pipe, background,
                       args, device, mask_filter)
    tg.save_pointcloud_with_metadata(
        xyz, rotulos, os.path.join(destino_base, "pointclouds", "clusters.ply"),
        metadata={"modo": args.modo, "n_clusters": int(n_clusters)})
    print(f"\n✅ Concluído em: {destino_base}")


if __name__ == "__main__":
    main()
