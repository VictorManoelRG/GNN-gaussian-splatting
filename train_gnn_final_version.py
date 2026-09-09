import os
import json
import cv2
import time
import threading
import torch.nn as nn

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: nvidia-ml-py não instalado. Instale com: pip install nvidia-ml-py "
          "para medir VRAM real (equivalente ao nvidia-smi) durante o treino.")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mode
from scipy.stats import mode as scipy_mode
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import hdbscan  

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from scene import Scene
from gaussian_renderer import GaussianModel, render, render_uniform
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not installed. Install with: pip install open3d")


class NvmlVRAMSampler:
    """
    Amostra, em uma thread separada, o mesmo valor de memória de GPU que o
    `nvidia-smi` reporta para este processo (via NVML), em vez de depender das
    estatísticas internas do allocator do PyTorch (`torch.cuda.max_memory_*`).

    Isso é necessário porque `max_memory_reserved` só contabiliza memória
    pedida através do caching allocator do PyTorch; alocações feitas por
    baixo dos panos por extensões CUDA de terceiros (ex.: kernels de
    scatter/sort usados pelas camadas de grafo do torch_geometric) não
    passam por esse allocator e ficam invisíveis para essas estatísticas,
    o que pode subestimar o pico real e até inverter a ordem entre
    configurações (uma rede mais "leve" nas estatísticas do PyTorch pode na
    prática reservar mais memória de contexto CUDA fora do allocator).
    """

    def __init__(self, device_index=0, interval=0.2):
        self.interval = interval
        self.device_index = device_index
        self._pid = os.getpid()
        self._peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread = None

    def _current_process_bytes(self):
        total = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except pynvml.NVMLError:
                continue
            for p in procs:
                if p.pid == self._pid and p.usedGpuMemory:
                    total += p.usedGpuMemory
        return total

    def _run(self):
        while not self._stop_event.is_set():
            try:
                current = self._current_process_bytes()
                if current > self._peak_bytes:
                    self._peak_bytes = current
            except pynvml.NVMLError:
                pass
            self._stop_event.wait(self.interval)

    def start(self):
        if not PYNVML_AVAILABLE:
            return
        pynvml.nvmlInit()
        self._stop_event.clear()
        self._peak_bytes = self._current_process_bytes()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_and_get_peak_gb(self):
        if not PYNVML_AVAILABLE:
            return None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1)
        try:
            current = self._current_process_bytes()
            self._peak_bytes = max(self._peak_bytes, current)
        except pynvml.NVMLError:
            pass
        pynvml.nvmlShutdown()
        return self._peak_bytes / (1024 ** 3)


# ==========================================================
# LOSS CONTRASTIVA PONDERADA POR OPACIDADE
# ==========================================================

class ColorAwareContrastiveLossV2(nn.Module):
    """
    Loss contrastiva em três termos, toda em espaço perceptual CIELAB.

    1. MARGEM POSITIVA / NEGATIVA sobre as arestas do grafo, ponderadas pelo
       peso da aresta (geometria x ΔE). Agem só entre vizinhos.
    2. CONSISTÊNCIA DE COR: aproxima vizinhos de ΔE pequeno.
    3. INFO-NCE CONTRA CENTRÓIDES: cada rótulo DEVA vira um centróide no
       espaço de embedding e toda gaussiana é classificada contra todos eles.

    O termo 3 já foi um InfoNCE de PARES minerados por dureza (o positivo menos
    parecido do rótulo contra os negativos mais parecidos de fora). Essa versão
    foi removida porque colapsa o embedding: os rótulos vêm de máscaras 2D do
    DEVA projetadas em 3D, uma fração deles está errada, e a mineração por
    dureza SELECIONA exatamente essa fração — o "positivo mais difícil" tende a
    ser um ponto mal rotulado e o "negativo mais difícil" um ponto do mesmo
    objeto com outro rótulo. A loss passa a exigir "aproxime-se do que é
    diferente" e "afaste-se do que é igual" ao mesmo tempo, e a única saída é
    aproximar tudo de tudo. Medido em cena sintética: 1% de rótulos trocados já
    derruba a margem intra-inter de +1.06 para +0.014; na cena real a margem
    era +0.0003 (cosseno 0.998 entre QUALQUER par).

    O centróide de um rótulo com milhares de gaussianas continua apontando para
    o objeto certo mesmo com 20-40% dos pontos errados, porque a média DILUI o
    ruído em vez de selecioná-lo. Na mesma cena real: margem +0.6278, ARI de
    0.176 para 0.265.

    QUATRO CHAVES OPCIONAIS abaixo (todas desligadas por padrão, preservando
    bit-a-bit o comportamento anterior) endereçam o sintoma de PCA com cores
    muito parecidas entre objetos distintos apesar de rótulos DEVA bem
    separados — diagnóstico discutido com o Gemini em 2026-09-08:

    - `color_exclude_diff_label`: a consistência de cor (item 2) por padrão
      pondera TODAS as arestas do grafo, inclusive as que ligam dois rótulos
      diferentes (`diff_label`). Se as duas pontas tiverem cor parecida (ex.:
      brinquedo sobre a mesa), a MSE empurra sim->1 exatamente onde a margem
      negativa (item 1) está tentando empurrar sim->0. As duas loss brigam e
      o resultado observável é a fronteira borrada no PCA. Só tem efeito
      quando `color_weight > 0`.
    - `detach_centroids`: por padrão o centróide de um rótulo é recalculado a
      cada passo A PARTIR dos MESMOS embeddings que serão classificados
      contra ele — o próprio ponto entra na média do seu centróide-alvo. O
      produto escalar do InfoNCE pode então subir "roubando", puxando o
      centróide até o ponto em vez de reorganizar genuinamente o espaço.
      Desconectar o centróide do autograd fecha esse atalho.
    - `nce_class_balanced`: por padrão `F.cross_entropy` faz a média POR NÓ.
      Um rótulo com dezenas de milhares de gaussianas (a mesa) domina o
      gradiente do InfoNCE inteiro; rótulos pequenos (objetos em cima dela)
      recebem sinal de separação proporcionalmente fraco. Fazer a média POR
      CLASSE dá o mesmo peso a cada rótulo independente do seu tamanho.
    - `centroid_repulsion_weight`: o InfoNCE já afasta cada PONTO dos
      centróides que não são o seu, mas nunca afasta os CENTRÓIDES entre si.
      Um peso > 0 adiciona essa repulsão direta (mesma margem negativa),
      espalhando os centróides pela hiperesfera.
    """

    def __init__(self, temperature=0.07, pos_margin=0.5, neg_margin=0.2,
                 color_weight=0.3, color_sigma=0.15,
                 color_exclude_diff_label=False,
                 detach_centroids=False,
                 nce_class_balanced=False,
                 centroid_repulsion_weight=0.0,
                 pos_weight=1.0, neg_weight=1.0, nce_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.color_weight = color_weight
        self.color_sigma = color_sigma
        self.color_exclude_diff_label = color_exclude_diff_label
        self.detach_centroids = detach_centroids
        self.nce_class_balanced = nce_class_balanced
        self.centroid_repulsion_weight = centroid_repulsion_weight
        # Pesos por termo. Todos 1.0 por padrão porque era exatamente assim que
        # os três entravam na soma antes (pos + neg + nce, sem coeficiente):
        # rodar sem passar nada reproduz o comportamento anterior bit-a-bit.
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.nce_weight = nce_weight

    def forward(self, embeddings, labels, edge_index, colors, edge_weights=None):
        src, dst = edge_index
        device = embeddings.device

        # Normalização L2 prévia: o produto escalar vira cosseno.
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        sim = (emb_norm[src] * emb_norm[dst]).sum(dim=1)

        # ==========================================================
        # 1. MARGENS SOBRE AS ARESTAS
        # ==========================================================
        valid = (labels[src] != -1) & (labels[dst] != -1)
        same_label = (labels[src] == labels[dst]) & valid
        diff_label = (labels[src] != labels[dst]) & valid

        pos_loss = torch.tensor(0.0, device=device)
        if same_label.any():
            raw_pos = torch.relu(1 - sim[same_label] - self.pos_margin)
            pos_loss = (raw_pos * edge_weights[same_label]).mean() if edge_weights is not None else raw_pos.mean()

        neg_loss = torch.tensor(0.0, device=device)
        if diff_label.any():
            raw_neg = torch.relu(sim[diff_label] - self.neg_margin)
            neg_loss = (raw_neg * edge_weights[diff_label]).mean() if edge_weights is not None else raw_neg.mean()

        # ==========================================================
        # 2. CONSISTÊNCIA DE COR (ΔE em CIELAB)
        # ==========================================================
        if self.color_exclude_diff_label:
            color_scope = ~diff_label
            color_src, color_dst, sim_color = src[color_scope], dst[color_scope], sim[color_scope]
        else:
            color_src, color_dst, sim_color = src, dst, sim

        if color_src.numel() > 0:
            color_dist = torch.norm(colors[color_src] - colors[color_dst], p=2, dim=1)
            color_sim = torch.exp(- (color_dist ** 2) / (2 * (self.color_sigma ** 2) + 1e-8))
            color_loss = F.mse_loss(sim_color, color_sim)
        else:
            color_loss = torch.tensor(0.0, device=device)

        # ==========================================================
        # 3. INFO-NCE CONTRA CENTRÓIDES DE RÓTULO
        # ==========================================================
        mask_v = labels != -1
        nce_loss = torch.tensor(0.0, device=device)
        cent_repulsion_loss = torch.tensor(0.0, device=device)
        n_classes = 0
        if mask_v.any():
            uniq_c, inv = torch.unique(labels[mask_v], return_inverse=True)
            n_classes = uniq_c.shape[0]
            if n_classes > 1:
                z_v = emb_norm[mask_v]
                soma = torch.zeros(n_classes, z_v.shape[1], device=device).index_add_(0, inv, z_v)
                cont = torch.zeros(n_classes, device=device).index_add_(
                    0, inv, torch.ones(inv.shape[0], device=device))
                centroides = F.normalize(soma / cont.unsqueeze(1).clamp(min=1.0), dim=1)

                # `detach_centroids=False` (padrão) reproduz o comportamento
                # anterior bit-a-bit: o alvo do produto escalar é o MESMO
                # tensor `centroides`, com gradiente fluindo de volta pelos
                # próprios embeddings que estão sendo classificados.
                centroides_logits = centroides.detach() if self.detach_centroids else centroides
                logits = (z_v @ centroides_logits.T) / self.temperature

                if self.nce_class_balanced:
                    ce_per_node = F.cross_entropy(logits, inv, reduction='none')
                    ce_per_class = torch.zeros(n_classes, device=device).index_add_(
                        0, inv, ce_per_node) / cont.clamp(min=1.0)
                    nce_loss = ce_per_class.mean()
                else:
                    nce_loss = F.cross_entropy(logits, inv)

                if self.centroid_repulsion_weight > 0:
                    cent_sim_matrix = centroides @ centroides.T
                    off_diag_mask = ~torch.eye(n_classes, dtype=torch.bool, device=device)
                    cent_repulsion_loss = torch.relu(
                        cent_sim_matrix[off_diag_mask] - self.neg_margin).mean()

        total_loss = ((1 - self.color_weight) * (self.pos_weight * pos_loss
                                                 + self.neg_weight * neg_loss
                                                 + self.nce_weight * nce_loss)
                      + (self.color_weight * color_loss)
                      + (self.centroid_repulsion_weight * cent_repulsion_loss))

        # Decomposição só para diagnóstico; não entra em nenhuma conta. O piso
        # de referência do InfoNCE é log(n_classes): é o valor que ele assume
        # quando o embedding não separa nada. Ficar colado nesse piso, com
        # `neg` SUBINDO ao longo do treino, é a assinatura do colapso.
        self.last_terms = {
            "pos": float(pos_loss.detach()),
            "neg": float(neg_loss.detach()),
            "nce": float(nce_loss.detach()),
            "color": float(color_loss.detach()),
            "cent_rep": float(cent_repulsion_loss.detach()),
            "nce_floor": float(np.log(max(n_classes, 1))),
            "n_anchors": int(mask_v.sum()),
        }
        return total_loss

# ==========================================================
# GAT ARCHITECTURE
# ==========================================================

# ==========================================================
# GCN ARCHITECTURE
# ==========================================================
class GaussianGCNV2(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, out_dim=32, dropout=0.1):
        super().__init__()
        # add_self_loops=False para não sobrescrever a proporção dos seus edge_weights
        self.gcn1 = GCNConv(in_channels, hidden_dim, add_self_loops=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.gcn2 = GCNConv(hidden_dim, hidden_dim * 2, add_self_loops=False)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

        # Projeção de atalho apenas entre as camadas ocultas, não do x bruto
        self.proj_res = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # Bloco 1
        h1 = self.gcn1(x, edge_index, edge_weight=edge_weight)
        h1 = self.norm1(h1)
        h1 = F.elu(h1)
        h1 = self.dropout(h1)

        # Bloco 2
        h2 = self.gcn2(h1, edge_index, edge_weight=edge_weight)
        h2 = self.norm2(h2)

        # Resíduo interno (entre representações aprendidas, sem o x bruto)
        res = self.proj_res(h1)
        h = F.elu(h2 + res)
        h = self.dropout(h)

        # Projeção de Saída
        x_out = self.linear(h)
        return x_out

class DeepGaussianGCN_ResNet(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, out_dim=32, dropout=0.1):
        super().__init__()
        # Entrada
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        
        # Bloco 1 (hidden_dim -> hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Bloco 2 (hidden_dim -> hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Bloco 3 (hidden_dim -> hidden_dim * 2)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim * 2, add_self_loops=False)
        self.norm3 = nn.LayerNorm(hidden_dim * 2)
        self.res_proj3 = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # Bloco 4 (hidden_dim * 2 -> hidden_dim * 2)
        self.gcn4 = GCNConv(hidden_dim * 2, hidden_dim * 2, add_self_loops=False)
        self.norm4 = nn.LayerNorm(hidden_dim * 2)

        # Projeção de Saída
        self.linear = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # Mapeamento inicial
        h0 = F.elu(self.in_proj(x))
        
        # Bloco 1 + Resíduo
        h1 = self.gcn1(h0, edge_index, edge_weight=edge_weight)
        h1 = self.norm1(h1)
        h1 = F.elu(h1 + h0)
        h1 = self.dropout(h1)

        # Bloco 2 + Resíduo
        h2 = self.gcn2(h1, edge_index, edge_weight=edge_weight)
        h2 = self.norm2(h2)
        h2 = F.elu(h2 + h1)
        h2 = self.dropout(h2)

        # Bloco 3 + Resíduo Projetado (Aumento de Dimensão)
        h3 = self.gcn3(h2, edge_index, edge_weight=edge_weight)
        h3 = self.norm3(h3)
        res3 = self.res_proj3(h2)
        h3 = F.elu(h3 + res3)
        h3 = self.dropout(h3)

        # Bloco 4 + Resíduo
        h4 = self.gcn4(h3, edge_index, edge_weight=edge_weight)
        h4 = self.norm4(h4)
        h4 = F.elu(h4 + h3)
        h4 = self.dropout(h4)

        # Saída
        x_out = self.linear(h4)
        return x_out
    
class GaussianGAT(torch.nn.Module):
    def __init__(self, in_channels, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_channels, 32, heads=heads, concat=True, dropout=0.1)
        self.bn1 = torch.nn.BatchNorm1d(32 * heads)
        self.gat2 = GATConv(32 * heads, 64, heads=2, concat=True, dropout=0.1)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.linear = torch.nn.Linear(128, 32)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.linear(x)
        return x


# ==========================================================
# DIAGNÓSTICO DO EMBEDDING (opcional, --diagnose_embedding)
# ==========================================================
def diagnose_embedding(emb, deva_labels, min_cluster_size, seed=0):
    """
    Responde três perguntas que o valor da loss sozinho não responde, e que
    separam "o modelo aprendeu mal" de "o HDBSCAN não tem onde cortar":

    1. SEPARAÇÃO: a similaridade média entre pares do MESMO rótulo DEVA é
       maior que entre pares de rótulos diferentes? Se a margem for pequena,
       o embedding não codifica identidade de instância, e nenhum ajuste de
       HDBSCAN conserta isso.
    2. MINERAÇÃO: em que fração das âncoras o positivo mais difícil está mais
       longe que o negativo mais difícil? É exatamente a condição que faz o
       InfoNCE ficar acima do piso log(1+K) e não convergir.
    3. ESTABILIDADE: quantos clusters o HDBSCAN acha para vários min_samples.
       Se o número desaba de dezenas para uns poucos ao subir min_samples, o
       embedding é um CONTÍNUO (sem vales de densidade entre objetos) e o
       resultado da clusterização é um corte arbitrário, não uma estrutura.
    """
    rng = np.random.default_rng(seed)
    print("\n" + "=" * 60)
    print("🔬 DIAGNÓSTICO DO EMBEDDING")
    print("=" * 60)

    valid = np.where(deva_labels != -1)[0]
    if len(valid) < 1000:
        print("  (poucos pontos rotulados, diagnóstico pulado)")
        return

    # ---- 1. separação intra vs inter rótulo ----
    sample = rng.choice(valid, size=min(20000, len(valid)), replace=False)
    e, l = emb[sample], deva_labels[sample]
    a, b = rng.integers(0, len(sample), 200000), rng.integers(0, len(sample), 200000)
    keep = a != b
    a, b = a[keep], b[keep]
    sims = np.einsum("ij,ij->i", e[a], e[b])
    same = l[a] == l[b]
    intra, inter = sims[same], sims[~same]
    if len(intra) and len(inter):
        print(f"  Similaridade média INTRA-rótulo : {intra.mean():+.4f} (desvio {intra.std():.4f})")
        print(f"  Similaridade média INTER-rótulo : {inter.mean():+.4f} (desvio {inter.std():.4f})")
        margem = intra.mean() - inter.mean()
        print(f"  Margem de separação             : {margem:+.4f}", end="")
        print("   ⚠️ margem quase nula: o embedding não separa instâncias"
              if margem < 0.10 else "")

    # ---- 2. taxa de falha da mineração adversarial ----
    falhas, testes = 0, 0
    uniq = np.unique(l)
    for lab in uniq[: min(40, len(uniq))]:
        pos = np.where(l == lab)[0]
        neg = np.where(l != lab)[0]
        if len(pos) < 2 or len(neg) < 10:
            continue
        for anchor in rng.choice(pos, size=min(5, len(pos)), replace=False):
            outros = pos[pos != anchor]
            cand = rng.choice(neg, size=min(512, len(neg)), replace=False)
            pior_pos = (e[outros] @ e[anchor]).min()
            pior_neg = np.sort(e[cand] @ e[anchor])[-10:].max()
            falhas += int(pior_pos < pior_neg)
            testes += 1
    if testes:
        taxa = 100 * falhas / testes
        print(f"  Âncoras com positivo mais difícil PIOR que o negativo mais "
              f"difícil: {taxa:.1f}% ({falhas}/{testes})", end="")
        print("   ⚠️ mineração hard impossível de vencer" if taxa > 50 else "")

    # ---- 3. estabilidade do HDBSCAN em min_samples ----
    sub_n = min(30000, len(emb))
    sub_idx = rng.choice(len(emb), size=sub_n, replace=False)
    mcs = max(20, int(min_cluster_size * sub_n / len(emb)))
    print(f"  Estabilidade do HDBSCAN (subamostra de {sub_n:,} pontos, "
          f"min_cluster_size={mcs}):")
    for ms in [3, 10, 25, 50, None]:
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                             prediction_data=False).fit(emb[sub_idx])
        n = len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0)
        ruido = 100 * float((cl.labels_ == -1).mean())
        rotulo = "None (=min_cluster_size)" if ms is None else str(ms)
        print(f"    min_samples={rotulo:<24} -> {n:3d} clusters | {ruido:5.1f}% ruído")
    print("  Se o nº de clusters desaba ao subir min_samples, o embedding é um "
          "contínuo\n  sem vales de densidade — o corte do HDBSCAN é arbitrário.")
    print("=" * 60)


# ==========================================================
# FILTRO DINÂMICO
# ==========================================================
def compute_dynamic_filters(opacity, scaling, target_count=220000):
    """
    Calcula thresholds de opacidade e escala que selecionam, de forma precisa,
    aproximadamente `target_count` gaussianas dentre as candidatas.

    Primeiro descarta os maiores outliers de escala (floaters) e depois
    seleciona, por RANKING, as `target_count` gaussianas de maior opacidade,
    desempatando pela menor escala.

    A seleção é por ranking, e não por um limiar de opacidade, por causa dos
    EMPATES: a opacidade passa por uma sigmoide que satura em 1.0 no float32.
    Em `waldo_kitchen`, por exemplo, 68.785 gaussianas (20,3% da cena) têm
    opacidade exatamente 1.0. Quando o `target_count` é menor que esse bloco
    empatado, qualquer limiar cai em cima de 1.0 e o teste `opacity > 1.0` não
    seleciona NINGUÉM — o conjunto zerava silenciosamente e só estourava
    depois, na renderização, com um erro de broadcast sem relação aparente.
    O desempate pela menor escala é coerente com o objetivo da função, já que
    gaussianas menores são mais bem definidas.

    Os thresholds retornados são DESCRITIVOS (a menor opacidade e a maior
    escala entre as selecionadas), mantidos para relatório; quem seleciona de
    fato é a máscara booleana, o quarto valor de retorno.
    """
    opacity = np.asarray(opacity).ravel()
    scaling = np.asarray(scaling).ravel()
    n_total = len(opacity)

    print(f"\n🎯 Calculando filtros dinâmicos para ~{target_count:,} gaussianas...")
    print(f"   Total disponível: {n_total:,}")

    if n_total <= target_count:
        print(f"   ⚠️ Total ({n_total:,}) já é menor que o target ({target_count:,})")
        print(f"   Mantendo todas as gaussianas disponíveis")
        return (float(opacity.min()) - 1e-6, float(scaling.max()) + 1e-6, n_total,
                np.ones(n_total, dtype=bool))

    # 1) Descarta floaters claros por escala (maior 1%), sem violar o target.
    sc_corte = float(np.percentile(scaling, 99.0))
    scale_mask = scaling < sc_corte
    if np.sum(scale_mask) < target_count:
        scale_mask = np.ones(n_total, dtype=bool)

    # 2) Dentre quem passa no filtro de escala, seleciona por ranking as
    #    `target_count` de maior opacidade, desempatando pela menor escala.
    idx_candidatos = np.where(scale_mask)[0]
    ordem = np.lexsort((scaling[idx_candidatos], -opacity[idx_candidatos]))
    escolhidos = idx_candidatos[ordem[:target_count]]

    mask = np.zeros(n_total, dtype=bool)
    mask[escolhidos] = True
    final_count = int(mask.sum())

    # Descritivos, para relatório: a faixa efetivamente selecionada.
    op_thresh = float(opacity[mask].min())
    sc_thresh = float(scaling[mask].max())

    n_empatados = int((opacity[mask] == op_thresh).sum())

    print(f"   ✅ Seleção concluída:")
    print(f"      Menor opacidade selecionada: {op_thresh:.4f}")
    print(f"      Maior escala selecionada: {sc_thresh:.4f}")
    print(f"      Gaussianas resultantes: {final_count:,} (target: {target_count:,})")
    if n_empatados > 1:
        print(f"      ({n_empatados:,} gaussianas empatadas na opacidade mínima, "
              f"desempatadas pela menor escala)")

    return op_thresh, sc_thresh, final_count, mask








def remove_spatial_outliers(xyz, mask, k=16, std_ratio=2.0):
    """
    Remove floaters: Gaussianas espacialmente isoladas dentro do subconjunto
    já marcado por `mask` (opacidade, escala, label DEVA válido, bbox). Um
    floater pode passar em todos esses filtros e ainda assim ser ruído, pois
    nenhum deles olha para a DENSIDADE local — um floater típico está sozinho
    no espaço 3D, longe da superfície real, mesmo tendo opacidade/escala/label
    "normais".

    Mesmo critério do `remove_statistical_outlier` do Open3D (distância média
    aos k vizinhos mais próximos, descartando quem fica a mais de
    `std_ratio` desvios-padrão da média da nuvem), reimplementado com sklearn
    para não depender do Open3D estar instalado no ambiente de treino.
    """
    idx = np.where(mask)[0]
    if len(idx) < k + 1:
        return mask

    sub_xyz = xyz[idx]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(sub_xyz)
    dist, _ = nbrs.kneighbors(sub_xyz)
    mean_dist = dist[:, 1:].mean(axis=1)  # exclui o próprio ponto (distância 0)

    global_mean = mean_dist.mean()
    global_std = mean_dist.std()
    keep_local = mean_dist <= (global_mean + std_ratio * global_std)

    new_mask = mask.copy()
    new_mask[idx[~keep_local]] = False

    n_removed = int(np.sum(~keep_local))
    print(f"  🧹 Remoção de floaters (outlier espacial, k={k}, std_ratio={std_ratio}): "
          f"{n_removed:,}/{len(idx):,} gaussianas removidas ({100*n_removed/len(idx):.2f}%)")

    return new_mask


# ==========================================================
# PROPAGAÇÃO 2D -> 3D POR VOTO SIMPLES DE MAIORIA
# ==========================================================
def build_labels_by_majority_vote(
    scene,
    xyz,
    masks_path,
    min_votes=20,
    depth_tolerance=0.05,
    depth_rel_tol=0.015,
    flush_every=25,
):
    """
    Propaga as máscaras 2D (SAM/DEVA) para as Gaussianas por contagem pura.

    Para cada câmera de treino:
      1. projeta o centro de cada Gaussiana na imagem;
      2. descarta as ocluídas com um z-buffer por pixel — só vota quem está a
         até `depth_tolerance` (absoluto, em unidades de mundo) ou
         `depth_rel_tol` (relativo) do ponto mais próximo daquele pixel. Sem
         isso, uma Gaussiana atrás da parede votaria na máscara da parede;
      3. conta UM voto para o ID de segmento da máscara naquele pixel — o
         pixel preto (ID 0) é FUNDO e vota como qualquer outro candidato.

    No fim, cada Gaussiana recebe o ID em que ela caiu mais vezes, desde que
    tenha ao menos `min_votes` votos nesse ID. Se o vencedor for o fundo, ela
    fica sem rótulo (-1). Sem peso por score/área/distância, sem validação de
    cor, sem fusão de tracks — só contagem.

    Deixar o fundo competir importa: se o voto de fundo fosse descartado, uma
    Gaussiana que cai no fundo na maioria dos frames ainda receberia o rótulo
    de um objeto sempre que os poucos frames restantes passassem de
    `min_votes`. Medido nas cenas de teste, descartar o fundo rotulava
    indevidamente 9% (counter) a 26% (bicycle) dos pixels de fundo; com o
    fundo competindo isso cai para 4,5% e 10,7%, e a acurácia 2D sobe
    3,0 e 7,4 pontos respectivamente.

    Não usa o JSON do DEVA: o valor do pixel da máscara já É o ID do segmento,
    então o JSON só acrescentaria filtros de score/área capazes de descartar
    segmentos silenciosamente.

    A projeção segue a convenção do rasterizador do 3DGS, SEM inversão do eixo
    vertical (v = (ndc_y * 0.5 + 0.5) * H). Isso foi verificado empiricamente
    pintando a cor base das Gaussianas nos pixels projetados e comparando com
    a foto de treino: sem inversão a correlação com a foto real fica em ~0.67,
    com inversão cai para ~0.10 (ou seja, a máscara seria lida na linha
    espelhada).
    """
    N = xyz.shape[0]
    STRIDE = 1 << 20  # nº máximo de IDs de segmento no empacotamento do voto

    p_hom = np.hstack([xyz, np.ones((N, 1), dtype=np.float64)])
    train_cameras = scene.getTrainCameras()

    # Votos são acumulados como chaves inteiras (gaussiana * STRIDE + sid) e
    # consolidados em blocos com np.unique. Uma versão com dict por Gaussiana
    # faria dezenas de milhões de iterações em Python puro; aqui tudo é
    # vetorizado e a memória fica limitada pelo tamanho do bloco.
    vote_keys = np.empty(0, dtype=np.int64)
    vote_counts = np.empty(0, dtype=np.int64)
    key_buffer = []

    def flush_votes():
        nonlocal vote_keys, vote_counts, key_buffer
        if not key_buffer:
            return
        novos = np.concatenate(key_buffer)
        key_buffer = []
        todas = np.concatenate([vote_keys, novos])
        pesos = np.concatenate([vote_counts, np.ones(novos.size, dtype=np.int64)])
        uniq, inv = np.unique(todas, return_inverse=True)
        vote_keys = uniq
        vote_counts = np.bincount(inv, weights=pesos).astype(np.int64)

    # Máscaras coloridas (Annotations_color) precisam de cor -> ID compacto;
    # as de 1 canal (Annotations) já trazem o ID no próprio valor do pixel.
    cor_para_id = {}

    frames_lidos = 0
    total_projetado = 0
    total_apos_zbuffer = 0

    print(f"🔍 Propagando máscaras para {N:,} Gaussianas por voto de maioria...")

    for i_view, view in enumerate(train_cameras):
        mask = cv2.imread(os.path.join(masks_path, f"{view.image_name}.png"),
                          cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        W, H = view.image_width, view.image_height

        if mask.ndim == 3:
            empacotada = ((mask[..., 0].astype(np.int64) << 16)
                          | (mask[..., 1].astype(np.int64) << 8)
                          | mask[..., 2].astype(np.int64))
            sid_map = np.zeros_like(empacotada)
            for cor in np.unique(empacotada):
                if cor == 0:
                    continue
                if cor not in cor_para_id:
                    cor_para_id[cor] = len(cor_para_id) + 1
                sid_map[empacotada == cor] = cor_para_id[cor]
        else:
            sid_map = mask.astype(np.int64)

        # Vizinho mais próximo por indexação: preserva os IDs exatos e não
        # esbarra nos dtypes que o cv2.resize não aceita (int64/int32).
        if sid_map.shape[0] != H or sid_map.shape[1] != W:
            lin = np.clip(np.arange(H) * sid_map.shape[0] // H, 0, sid_map.shape[0] - 1)
            col = np.clip(np.arange(W) * sid_map.shape[1] // W, 0, sid_map.shape[1] - 1)
            sid_map = sid_map[np.ix_(lin, col)]

        frames_lidos += 1
        if frames_lidos % 50 == 0:
            print(f"  Frame {i_view + 1}/{len(train_cameras)}...")

        # --- projeção 3D -> 2D (convenção do rasterizador 3DGS) ---
        w2c = view.world_view_transform.detach().cpu().numpy()
        P = view.full_proj_transform.detach().cpu().numpy()

        z_cam = (p_hom @ w2c)[:, 2]
        proj = p_hom @ P
        w_seguro = np.where(np.abs(proj[:, 3:4]) < 1e-8, 1e-8, proj[:, 3:4])
        ndc = proj[:, :3] / w_seguro

        u = np.floor((ndc[:, 0] * 0.5 + 0.5) * W).astype(np.int64)
        v = np.floor((ndc[:, 1] * 0.5 + 0.5) * H).astype(np.int64)

        visivel = (z_cam > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(visivel)[0]
        if idx.size == 0:
            continue

        # --- z-buffer por pixel: quem está atrás da superfície não vota ---
        u_i, v_i, z_i = u[idx], v[idx], z_cam[idx]
        pixel = v_i * W + u_i

        ordem = np.lexsort((z_i, pixel))  # agrupa por pixel, mais perto primeiro
        pixel_s, z_s = pixel[ordem], z_i[ordem]
        idx_s, u_s, v_s = idx[ordem], u_i[ordem], v_i[ordem]

        pixels_uniq, primeiro = np.unique(pixel_s, return_index=True)
        z_frente = z_s[primeiro][np.searchsorted(pixels_uniq, pixel_s)]
        na_frente = z_s <= z_frente + np.maximum(z_frente * depth_rel_tol, depth_tolerance)

        idx_s, u_s, v_s = idx_s[na_frente], u_s[na_frente], v_s[na_frente]

        total_projetado += idx.size
        total_apos_zbuffer += idx_s.size
        if idx_s.size == 0:
            continue

        # --- um voto por Gaussiana para o segmento daquele pixel ---
        # O pixel preto (ID 0) é FUNDO, e vota como qualquer outro candidato.
        # Descartar o voto de fundo faria uma Gaussiana que cai no fundo na
        # maioria dos frames ainda receber o rótulo de um objeto, bastando que
        # os poucos frames restantes passassem de `min_votes` — parede, chão e
        # céu acabavam absorvidos pelos objetos vizinhos.
        sids = sid_map[v_s, u_s]
        if sids.max() >= STRIDE:
            raise ValueError(
                f"ID de segmento {int(sids.max())} excede o limite de "
                f"empacotamento ({STRIDE}). Aumente STRIDE."
            )

        key_buffer.append(idx_s * STRIDE + sids)

        if frames_lidos % flush_every == 0:
            flush_votes()

    flush_votes()

    if total_projetado > 0:
        descartado = 100 * (1 - total_apos_zbuffer / total_projetado)
        print(f"📉 Z-buffer: {total_projetado:,} → {total_apos_zbuffer:,} projeções "
              f"({descartado:.1f}% descartadas por oclusão)")

    # -------------------------------------------------------------------------
    # Vencedor por Gaussiana = ID com mais votos
    # -------------------------------------------------------------------------
    labels = np.full(N, -1, dtype=np.int64)

    if vote_keys.size > 0:
        gauss_id = vote_keys // STRIDE
        seg_id = vote_keys % STRIDE

        # Ordena por (gaussiana, contagem): o último de cada grupo é o mais
        # votado. Empate é resolvido pelo maior ID de segmento, o que mantém o
        # resultado determinístico entre execuções — e, como o fundo é o menor
        # ID (0), um empate entre fundo e objeto fica com o objeto.
        ordem = np.lexsort((vote_counts, gauss_id))
        g_ord, s_ord, c_ord = gauss_id[ordem], seg_id[ordem], vote_counts[ordem]

        ultimo_do_grupo = np.ones(g_ord.size, dtype=bool)
        ultimo_do_grupo[:-1] = g_ord[1:] != g_ord[:-1]

        g_venc = g_ord[ultimo_do_grupo]
        s_venc = s_ord[ultimo_do_grupo]
        c_venc = c_ord[ultimo_do_grupo]

        # Vencedor 0 = a Gaussiana caiu no fundo na maioria das vezes; ela fica
        # sem rótulo em vez de herdar o objeto mais votado depois do fundo.
        venceu_fundo = s_venc == 0
        aceito = (c_venc >= min_votes) & ~venceu_fundo
        labels[g_venc[aceito]] = s_venc[aceito]

        rotuladas = int(aceito.sum())
        print(f"\n✅ Concluído:")
        print(f"  • Frames com máscara lida: {frames_lidos}/{len(train_cameras)}")
        print(f"  • Gaussianas rotuladas: {rotuladas:,}/{N:,} ({100 * rotuladas / N:.1f}%)")
        print(f"  • Segmentos distintos atribuídos: {len(np.unique(s_venc[aceito])):,}")
        if rotuladas > 0:
            print(f"  • Votos do vencedor: mediana={int(np.median(c_venc[aceito]))} | "
                  f"média={c_venc[aceito].mean():.1f} | máx={int(c_venc[aceito].max())}")
        print(f"  • Gaussianas vetadas por maioria de fundo: {int(venceu_fundo.sum()):,}")
        abaixo = int(((c_venc < min_votes) & ~venceu_fundo).sum())
        print(f"  • Gaussianas com voto de objeto mas abaixo de min_votes={min_votes}: {abaixo:,}")
    else:
        print("⚠️  Nenhum voto acumulado — verifique masks_path e os nomes das imagens.")

    return labels




# ==========================================================
# PROPAGAÇÃO DE LABELS PARA O CONJUNTO COMPLETO (SPAÇAL)
# ==========================================================
def propagate_labels_to_full_set(xyz_filtered, cluster_labels, xyz_full, k=1, max_distance=None):
    """
    Propaga os labels de cluster do subconjunto filtrado para TODAS as gaussianas
    originais usando o vizinho espacial mais próximo (corrige buracos brancos).

    `max_distance`: se definido, gaussianas cujo vizinho rotulado mais próximo
    está mais longe do que isso permanecem com label -1 em vez de herdar um
    rótulo espúrio. Sem esse limite, TODA gaussiana do conjunto completo recebe
    um cluster (mesmo floaters/outliers já removidos do subconjunto filtrado),
    o que pode reintroduzir na renderização exatamente os artefatos que a
    filtragem/remoção de floaters tentou eliminar.
    """
    print(f"\n🔁 Propagando labels para o conjunto completo de gaussianas...")
    print(f"   Subconjunto filtrado: {len(xyz_filtered):,} | Conjunto completo: {len(xyz_full):,}")

    k = min(k, len(xyz_filtered))
    nbrs_full = NearestNeighbors(n_neighbors=k).fit(xyz_filtered)
    dist, nn_idx = nbrs_full.kneighbors(xyz_full)

    if k == 1:
        full_labels = cluster_labels[nn_idx.flatten()]
    else:
        neighbor_labels = cluster_labels[nn_idx]
        full_labels = np.array([
            np.bincount(row[row != -1]).argmax() if np.any(row != -1) else -1
            for row in neighbor_labels
        ])

    if max_distance is not None:
        too_far = dist.min(axis=1) > max_distance
        n_too_far = int(np.sum(too_far & (full_labels != -1)))
        full_labels[too_far] = -1
        print(f"   🚫 {n_too_far:,} gaussianas fora do raio máximo ({max_distance}) mantidas sem label")

    n_clusters = len(np.unique(full_labels[full_labels != -1]))
    print(f"   ✅ {len(xyz_full):,} gaussianas agora possuem label ({n_clusters} clusters)")

    return full_labels


# ==========================================================
# [MODIFICADO] REATRIBUIÇÃO VELOZ DE RUÍDO VIA KNN (EMBEDDINGS)
# ==========================================================
def reassign_noise_knn_vectorized(emb, cluster_labels, k=7, min_agreement=0.5):
    """
    Reatribui pontos de ruído (-1) com base no voto majoritário dos seus K vizinhos
    mais próximos dentro do espaço de embeddings. Totalmente vetorizado.
    """
    noise_mask = cluster_labels == -1
    n_noise_before = np.sum(noise_mask)

    if n_noise_before == 0:
        print("\n🔁 Reatribuição de ruído: nenhum ponto de ruído encontrado.")
        return cluster_labels

    print(f"\n🔁 Reatribuindo ruído via KNN nos Embeddings (K={k})...")
    print(f"   Pontos de ruído antes: {n_noise_before:,}")

    core_mask = ~noise_mask
    if np.sum(core_mask) == 0:
        print("   ⚠️ Erro: Nenhum cluster válido encontrado para herdar labels.")
        return cluster_labels

    # Treina o KNN exclusivamente nos pontos que possuem uma classe atribuída
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(emb[core_mask])
    _, nn_idx = nbrs.kneighbors(emb[noise_mask])
    
    # Busca os labels reais mapeados dos vizinhos estruturados
    core_indices = np.where(core_mask)[0]
    mapped_neigh_labels = cluster_labels[core_indices[nn_idx]] 

    # Calcula a moda (voto majoritário) de forma vetorizada
    modes, counts = mode(mapped_neigh_labels, axis=1, keepdims=False)

    # Filtra com base na concordância mínima exigida dos vizinhos
    agreement_mask = counts >= (k * min_agreement)

    new_labels = cluster_labels.copy()
    noise_indices = np.where(noise_mask)[0]
    new_labels[noise_indices[agreement_mask]] = modes[agreement_mask]

    n_noise_after = np.sum(new_labels == -1)
    print(f"   Pontos de ruído após reatribuição: {n_noise_after:,} "
          f"(reatribuídos: {n_noise_before - n_noise_after:,})")

    return new_labels


# ==========================================================
# VISUALIZATION & RENDER BY CLUSTER
# ==========================================================
def visualize_clusters_open3d(xyz, labels, window_title="Segmentação GAT + HDBSCAN", save_path=None):
    if not OPEN3D_AVAILABLE:
        return None
    
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        labels_normalized = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
        colors = plt.get_cmap("tab20")(labels_normalized)
        colors[labels == -1] = [0.1, 0.1, 0.1, 1]
    else:
        colors = np.full((len(xyz), 4), [0.5, 0.5, 0.5, 1])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)
    
    n_clusters = len(np.unique(labels[labels != -1]))
    n_noise = np.sum(labels == -1)
    print(f"\n📊 {window_title}")
    print(f"  Total points: {len(xyz):,}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Noise (outliers): {n_noise} ({100*n_noise/len(xyz):.1f}%)")
    
    o3d.visualization.draw_geometries([pcd], window_name=window_title, width=1280, height=720)
    return pcd




# ==========================================================
# DIAGNÓSTICO / VISUALIZAÇÃO DO GRAFO INICIAL
# ==========================================================
# Todo este bloco só roda quando --graph_debug é passado. Ele NÃO altera o
# grafo nem nenhum resultado: apenas lê as estruturas já construídas
# (edges_np, weights_np, a matriz kNN e os thresholds) e escreve
# estatísticas, figuras e nuvens/linhas em disco.
#
# A pergunta que este diagnóstico responde é "como está a conexão inicial?",
# ou seja: antes de a GNN aprender qualquer coisa, quais gaussianas o grafo
# já está costurando entre si — e, principalmente, ONDE ele costura pontos de
# objetos diferentes (vazamento) ou deixa um mesmo objeto partido em vários
# pedaços desconectados (fragmentação). Esses dois erros limitam o teto do
# que o treino consegue alcançar: mensagens só passam por arestas, então um
# objeto partido no grafo tende a virar dois clusters, e uma aresta entre
# dois objetos "cola" embeddings que deveriam se separar.


def _write_ply_points(path, xyz, rgb01):
    """
    Escreve uma nuvem de pontos colorida em PLY binário, sem depender do
    Open3D (que é opcional aqui). Serve para abrir em CloudCompare/MeshLab.
    """
    xyz = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32))
    rgb = np.clip(np.asarray(rgb01, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr = np.empty(len(xyz), dtype=dt)
    arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(arr)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        arr.tofile(f)


def _write_ply_lineset(path, xyz, point_rgb01, edges, edge_rgb01):
    """
    Escreve as ARESTAS do grafo como um PLY com `element edge`, formato que
    MeshLab e o próprio Open3D (read_line_set) abrem. Só os vértices tocados
    pelas arestas exportadas entram no arquivo (reindexados), para o arquivo
    não carregar a nuvem inteira junto.
    """
    edges = np.asarray(edges, dtype=np.int64)
    used, remap = np.unique(edges.reshape(-1), return_inverse=True)
    remap = remap.reshape(edges.shape).astype(np.int32)

    v = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32)[used])
    vc = np.clip(np.asarray(point_rgb01, dtype=np.float64)[used] * 255.0, 0, 255).astype(np.uint8)
    ec = np.clip(np.asarray(edge_rgb01, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)

    vdt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    edt = np.dtype([('vertex1', '<i4'), ('vertex2', '<i4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    va = np.empty(len(v), dtype=vdt)
    va['x'], va['y'], va['z'] = v[:, 0], v[:, 1], v[:, 2]
    va['red'], va['green'], va['blue'] = vc[:, 0], vc[:, 1], vc[:, 2]

    ea = np.empty(len(remap), dtype=edt)
    ea['vertex1'], ea['vertex2'] = remap[:, 0], remap[:, 1]
    ea['red'], ea['green'], ea['blue'] = ec[:, 0], ec[:, 1], ec[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(va)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element edge {len(ea)}\n"
        "property int vertex1\nproperty int vertex2\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        va.tofile(f)
        ea.tofile(f)


def _colormap01(values, cmap="viridis"):
    """Mapeia um vetor qualquer para RGB 0-1 normalizando por min/max."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    vn = (v - lo) / (hi - lo + 1e-12)
    return plt.get_cmap(cmap)(vn)[:, :3]


def debug_graph_connectivity(
    xyz,
    labels,
    edges_np,
    weights_np,
    point_rgb01,
    neighbor_idx,
    neighbor_dist,
    color_dist,
    edge_mask,
    edge_mask_pre_force=None,
    dist_ok=None,
    distance_threshold=None,
    distance_desc=None,
    color_dist_threshold=None,
    out_dir="graph_debug",
    max_edges_export=150000,
    max_edges_plot=30000,
    show=False,
    seed=42,
):
    """
    Gera um retrato completo do grafo RECÉM-CONSTRUÍDO (antes do treino).

    Escreve em `out_dir`:
      - graph_report.txt / graph_stats.json : todas as estatísticas abaixo
      - graph_hists.png       : histogramas de grau, ΔE, distância e peso
      - graph_edges_proj.png  : projeções XY/XZ das arestas (vermelho =
                                aresta entre labels diferentes)
      - graph_edges_label.ply : arestas 3D coloridas por same/diff label
      - graph_edges_weight.ply: as mesmas arestas coloridas pelo peso
      - graph_nodes_degree.ply: nuvem colorida pelo grau de cada nó
      - graph_nodes_leak.ply  : nuvem colorida pela fração de arestas do nó
                                que vão para outro label (vermelho = vazando)
      - graph_components.ply  : nuvem colorida por componente conexa

    Retorna o dicionário de estatísticas.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    N = len(xyz)
    stats = {}
    lines = []

    def _log(msg):
        print(msg)
        lines.append(msg)

    _log("\n" + "=" * 70)
    _log("🔎 DIAGNÓSTICO DO GRAFO INICIAL (antes do treino)")
    _log("=" * 70)
    # Nos modos de distância local o limiar varia por par; o escalar recebido é
    # a mediana deles, e os rótulos abaixo precisam dizer isso.
    _thr_lbl = "limiar" if distance_desc is None else "limiar mediano"
    if distance_desc is not None:
        _log(f"    critério de distância: {distance_desc}")

    # ---------- 1. Poda do kNN: o que sobrou dos candidatos ----------
    # Cada linha da matriz kNN é um nó e cada coluna um vizinho candidato.
    # Aqui se vê QUEM está cortando aresta: o limiar de distância, o de cor,
    # ou os dois juntos.
    if distance_threshold is not None and color_dist_threshold is not None:
        # `dist_ok` é a máscara realmente aplicada: nos modos locais o limiar é
        # por par e não existe escalar que a reproduza.
        m_far = (~dist_ok) if dist_ok is not None else (neighbor_dist > distance_threshold)
        m_col = color_dist > color_dist_threshold
        total_cand = m_far.size
        stats["knn_candidatos"] = int(total_cand)
        stats["knn_aceitos_pct"] = float(100.0 * (~m_far & ~m_col).mean())
        stats["knn_corte_so_distancia_pct"] = float(100.0 * (m_far & ~m_col).mean())
        stats["knn_corte_so_cor_pct"] = float(100.0 * (~m_far & m_col).mean())
        stats["knn_corte_ambos_pct"] = float(100.0 * (m_far & m_col).mean())
        _log(f"\n[1] Poda do kNN ({total_cand:,} pares candidatos = N x (k-1))")
        _log(f"    aceitos ................ {stats['knn_aceitos_pct']:.1f}%")
        _log(f"    cortados só por distância {stats['knn_corte_so_distancia_pct']:.1f}%")
        _log(f"    cortados só por cor ..... {stats['knn_corte_so_cor_pct']:.1f}%")
        _log(f"    cortados pelos dois ..... {stats['knn_corte_ambos_pct']:.1f}%")

    if edge_mask_pre_force is not None:
        n_forced = int((edge_mask & ~edge_mask_pre_force).sum())
        stats["arestas_forcadas_grau_minimo"] = n_forced
        stats["arestas_forcadas_pct"] = float(100.0 * n_forced / max(int(edge_mask.sum()), 1))
        _log(f"    arestas forçadas pelo grau mínimo: {n_forced:,} "
             f"({stats['arestas_forcadas_pct']:.2f}% das arestas dirigidas antes da simetrização)")

    # ---------- 2. Grau e conectividade básica ----------
    src_all = edges_np[:, 0]
    dst_all = edges_np[:, 1]
    deg = np.bincount(src_all, minlength=N)
    und = edges_np[src_all < dst_all]
    und_w = weights_np[src_all < dst_all]

    stats["n_nos"] = int(N)
    stats["n_arestas_nao_direcionadas"] = int(len(und))
    stats["n_arestas_direcionadas"] = int(len(edges_np))
    stats["grau_medio"] = float(deg.mean())
    stats["grau_mediana"] = float(np.median(deg))
    stats["grau_min"] = int(deg.min())
    stats["grau_max"] = int(deg.max())
    stats["nos_isolados"] = int((deg == 0).sum())
    stats["nos_grau_1"] = int((deg == 1).sum())

    _log(f"\n[2] Nós e arestas")
    _log(f"    nós ..................... {N:,}")
    _log(f"    arestas (não direcionadas) {len(und):,}  | direcionadas: {len(edges_np):,}")
    _log(f"    grau: média={deg.mean():.2f} mediana={np.median(deg):.0f} "
         f"min={deg.min()} max={deg.max()}")
    _log(f"    nós isolados (grau 0) ... {stats['nos_isolados']:,} "
         f"({100.0 * stats['nos_isolados'] / N:.3f}%)")
    _log(f"    nós com grau 1 .......... {stats['nos_grau_1']:,} "
         f"({100.0 * stats['nos_grau_1'] / N:.3f}%)")
    if stats["nos_isolados"] > 0:
        _log("    ⚠️ Nós isolados não recebem mensagem nenhuma no GCN "
             "(add_self_loops=False): o embedding deles colapsa para o bias.")

    w_p = np.percentile(und_w, [1, 25, 50, 75, 99])
    stats["peso_percentis_1_25_50_75_99"] = [float(v) for v in w_p]
    _log(f"    peso das arestas: p1={w_p[0]:.3f} p25={w_p[1]:.3f} mediana={w_p[2]:.3f} "
         f"p75={w_p[3]:.3f} p99={w_p[4]:.3f}")

    # ---------- 3. Vazamento: arestas entre labels diferentes ----------
    lab_s, lab_d = labels[und[:, 0]], labels[und[:, 1]]
    valid = (lab_s != -1) & (lab_d != -1)
    same = (lab_s == lab_d) & valid
    diff = (lab_s != lab_d) & valid
    n_same, n_diff = int(same.sum()), int(diff.sum())
    n_unlab = int((~valid).sum())
    stats["arestas_same_label"] = n_same
    stats["arestas_diff_label"] = n_diff
    stats["arestas_com_no_sem_label"] = n_unlab
    stats["pct_arestas_diff_label"] = float(100.0 * n_diff / max(n_same + n_diff, 1))

    _log(f"\n[3] Pureza das arestas em relação aos labels do DEVA")
    _log(f"    same-label .............. {n_same:,}")
    _log(f"    diff-label (vazamento) .. {n_diff:,} "
         f"({stats['pct_arestas_diff_label']:.2f}% das arestas rotuladas)")
    _log(f"    com pelo menos um nó -1 . {n_unlab:,}")

    if n_diff > 0:
        pairs = np.sort(np.stack([lab_s[diff], lab_d[diff]], axis=1), axis=1)
        uniq_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        order = np.argsort(-pair_counts)[:10]
        stats["top_pares_vazamento"] = [
            [int(uniq_pairs[i, 0]), int(uniq_pairs[i, 1]), int(pair_counts[i])] for i in order
        ]
        _log("    pares de labels mais colados pelo grafo (label_a, label_b, nº arestas):")
        for i in order:
            _log(f"      {int(uniq_pairs[i,0]):>4} <-> {int(uniq_pairs[i,1]):>4} : "
                 f"{int(pair_counts[i]):,}")

    # fração de arestas "vazando" por nó — é isso que vira o mapa 3D de vazamento
    leak_num = np.bincount(und[diff][:, 0], minlength=N) + np.bincount(und[diff][:, 1], minlength=N)
    leak_frac = leak_num / np.maximum(deg, 1)

    # ---------- 3b. De onde vem cada aresta: ΔE, comprimento e forçadas ----------
    # `und` já perdeu a origem na matriz kNN (foi simetrizada e deduplicada),
    # então aqui se refaz o vínculo aresta -> (ΔE, distância, foi forçada?).
    # Uma aresta não direcionada é considerada FORÇADA quando nenhuma das
    # direções que a criaram passava nos limiares: ela só existe porque o
    # grau mínimo a impôs. Isso importa porque a aresta forçada ignora tanto
    # o corte de cor quanto o de distância — e no GAT, que não usa
    # edge_weight, ela transmite mensagem com força total mesmo tendo peso ~0.
    src_k, k_k = np.where(edge_mask)
    dst_k = neighbor_idx[src_k, k_k]
    a_k = np.minimum(src_k, dst_k).astype(np.int64)
    b_k = np.maximum(src_k, dst_k).astype(np.int64)
    key_k = a_k * N + b_k
    uk, inv = np.unique(key_k, return_inverse=True)

    de_k = color_dist[src_k, k_k]
    dist_k = neighbor_dist[src_k, k_k]
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uk)))
    de_u = np.minimum.reduceat(de_k[order], starts)
    dist_u = np.minimum.reduceat(dist_k[order], starts)
    if edge_mask_pre_force is not None:
        unforced_k = edge_mask_pre_force[src_k, k_k]
        forced_u = np.bincount(inv, weights=unforced_k.astype(np.float64), minlength=len(uk)) == 0
    else:
        forced_u = np.zeros(len(uk), dtype=bool)

    pos = np.searchsorted(uk, und[:, 0].astype(np.int64) * N + und[:, 1].astype(np.int64))
    de_e, dist_e, forced_e = de_u[pos], dist_u[pos], forced_u[pos]

    n_forced_e = int(forced_e.sum())
    stats["arestas_forcadas_no_grafo_final"] = n_forced_e
    stats["arestas_forcadas_no_grafo_final_pct"] = float(100.0 * n_forced_e / max(len(und), 1))
    stats["de_mediano_arestas"] = float(np.median(de_e))
    stats["de_max_arestas"] = float(de_e.max())
    stats["comprimento_mediano_arestas"] = float(np.median(dist_e))
    stats["comprimento_max_arestas"] = float(dist_e.max())

    if edge_mask_pre_force is None:
        # Não há mecanismo de reposição de aresta: estes números TÊM que ficar
        # dentro dos limiares. Se o máximo ultrapassar, algo repôs aresta.
        _log(f"\n[3b] ΔE e comprimento das arestas efetivamente criadas")
        _log(f"    ΔE .................... mediana={np.median(de_e):.1f} "
             f"p90={np.percentile(de_e, 90):.1f} máx={de_e.max():.1f} "
             f"(limiar {color_dist_threshold:.1f})")
        _log(f"    comprimento ........... mediana={np.median(dist_e):.4f} "
             f"p90={np.percentile(dist_e, 90):.4f} máx={dist_e.max():.4f} "
             f"({_thr_lbl} {distance_threshold:.4f})")
        _log(f"    toda aresta passou nos dois critérios (sem grau mínimo forçado). "
             f"Distância: {distance_desc or 'limiar global'}.")
    else:
        _log(f"\n[3b] Arestas que só existem por causa do grau mínimo")
        _log(f"    total ................... {n_forced_e:,} "
             f"({stats['arestas_forcadas_no_grafo_final_pct']:.2f}% das arestas do grafo)")
    if n_forced_e > 0:
        _log(f"    ΔE ...................... mediana={np.median(de_e[forced_e]):.1f} "
             f"p90={np.percentile(de_e[forced_e], 90):.1f} máx={de_e[forced_e].max():.1f} "
             f"(limiar era {color_dist_threshold:.1f})")
        _log(f"    comprimento ............. mediana={np.median(dist_e[forced_e]):.4f} "
             f"p90={np.percentile(dist_e[forced_e], 90):.4f} máx={dist_e[forced_e].max():.4f} "
             f"({_thr_lbl} era {distance_threshold:.4f})")
        frac_leak_forced = float(100.0 * (forced_e & diff).sum() / max(n_diff, 1))
        stats["pct_vazamento_vindo_de_arestas_forcadas"] = frac_leak_forced
        _log(f"    quanto do vazamento vem delas: {frac_leak_forced:.1f}% "
             f"({int((forced_e & diff).sum()):,} das {n_diff:,} arestas diff-label)")

        # O gatilho do grau mínimo olha `edge_mask.sum(axis=1)`, que é só o grau
        # de SAÍDA (a linha do nó na matriz kNN). Como o grafo é simetrizado
        # depois, um nó com linha vazia ainda pode receber aresta de um vizinho
        # que o escolheu. Ou seja, o gatilho dispara em muito mais nós do que os
        # que de fato ficariam isolados. Estes são os números reais, medidos no
        # grafo não direcionado que existiria SEM o forçamento.
        keep_u = ~forced_u
        deg_pre = (np.bincount(uk[keep_u] // N, minlength=N)
                   + np.bincount(uk[keep_u] % N, minlength=N))
        n_iso_real = int((deg_pre == 0).sum())
        n_lt2_real = int((deg_pre < 2).sum())
        stats["nos_realmente_isolados_sem_forcamento"] = n_iso_real
        stats["nos_com_grau_menor_que_2_sem_forcamento"] = n_lt2_real
        _log(f"    sem o grau mínimo, ficariam de fato isolados (grau 0): {n_iso_real:,} nós "
             f"({100.0 * n_iso_real / N:.2f}%)")
        _log(f"    sem o grau mínimo, ficariam com grau < 2: {n_lt2_real:,} nós "
             f"({100.0 * n_lt2_real / N:.2f}%)")
        _log(f"    -> compare com as {n_forced_e:,} arestas que o mecanismo criou: "
             f"se o número de nós realmente isolados for pequeno, o grau mínimo "
             f"está pagando caro por um problema que quase não existe.")

    # Tabela completa de vazamento (não só o top 10), para poder consultar
    # qualquer par de objetos: "o label X está colado no label Y?".
    if n_diff > 0:
        pairs_all = np.sort(np.stack([lab_s[diff], lab_d[diff]], axis=1), axis=1)
        up, ip = np.unique(pairs_all, axis=0, return_inverse=True)
        cnt = np.bincount(ip, minlength=len(up))
        fcnt = np.bincount(ip, weights=forced_e[diff].astype(np.float64), minlength=len(up))
        ordp = np.argsort(-cnt)
        # Agrupamento por ordenação em vez de uma máscara booleana por par:
        # com centenas de pares e centenas de milhares de arestas, a versão
        # com máscara varreria o vetor inteiro uma vez por par.
        idx_diff = np.where(diff)[0]
        order_p = np.argsort(ip, kind="stable")
        starts_p = np.searchsorted(ip[order_p], np.arange(len(up)))
        ends_p = np.append(starts_p[1:], len(ip))
        csv_path = os.path.join(out_dir, "graph_leak_pairs.csv")
        with open(csv_path, "w") as f:
            f.write("label_a,label_b,n_arestas,n_forcadas,de_mediano,dist_mediana\n")
            for i in ordp:
                g = idx_diff[order_p[starts_p[i]:ends_p[i]]]
                f.write(f"{int(up[i,0])},{int(up[i,1])},{int(cnt[i])},{int(fcnt[i])},"
                        f"{np.median(de_e[g]):.2f},{np.median(dist_e[g]):.5f}\n")
        _log(f"    tabela completa de pares em vazamento: {csv_path} ({len(up)} pares)")

    # PLY só com as arestas forçadas: são as candidatas naturais a "por que
    # esses dois objetos ficaram grudados?".
    if n_forced_e > 0:
        idx_f = np.where(forced_e)[0]
        if len(idx_f) > max_edges_export:
            idx_f = rng.choice(idx_f, size=max_edges_export, replace=False)
        ec_f = np.tile(np.array([1.0, 0.55, 0.0]), (len(idx_f), 1))
        ec_f[diff[idx_f]] = [0.9, 0.0, 0.9]   # magenta: forçada E ligando objetos diferentes
        _write_ply_lineset(os.path.join(out_dir, "graph_edges_forced.ply"),
                           xyz, point_rgb01, und[idx_f], ec_f)

    # ---------- 4. Componentes conexas ----------
    A = coo_matrix((np.ones(len(und)), (und[:, 0], und[:, 1])), shape=(N, N))
    n_comp, comp_id = connected_components(A, directed=False)
    comp_sizes = np.bincount(comp_id)
    big = np.sort(comp_sizes)[::-1][:10]
    stats["n_componentes_conexas"] = int(n_comp)
    stats["maior_componente"] = int(big[0])
    stats["maior_componente_pct"] = float(100.0 * big[0] / N)
    stats["top10_componentes"] = [int(v) for v in big]
    stats["componentes_com_1_no"] = int((comp_sizes == 1).sum())

    _log(f"\n[4] Componentes conexas do grafo inteiro")
    _log(f"    total ................... {n_comp:,}")
    _log(f"    maior ................... {big[0]:,} nós ({stats['maior_componente_pct']:.2f}%)")
    _log(f"    10 maiores .............. {', '.join(f'{v:,}' for v in big)}")
    _log(f"    componentes de 1 nó só .. {stats['componentes_com_1_no']:,}")

    # ---------- 5. Fragmentação: cada objeto está inteiro no grafo? ----------
    # Subgrafo só com arestas same-label: se um label aparece partido em
    # várias componentes, nenhuma quantidade de treino junta esses pedaços
    # por troca de mensagem — eles só podem se reencontrar no HDBSCAN.
    und_same = und[same]
    A_same = coo_matrix((np.ones(len(und_same)), (und_same[:, 0], und_same[:, 1])), shape=(N, N))
    _, comp_same = connected_components(A_same, directed=False)

    frag_rows = []
    for lb in np.unique(labels[labels != -1]):
        idx = np.where(labels == lb)[0]
        comps, counts = np.unique(comp_same[idx], return_counts=True)
        frag_rows.append((int(lb), len(idx), len(comps), float(100.0 * counts.max() / len(idx))))
    frag_rows.sort(key=lambda r: -r[2])
    stats["fragmentacao_por_label"] = [
        {"label": r[0], "n_pontos": r[1], "n_fragmentos": r[2], "maior_fragmento_pct": r[3]}
        for r in frag_rows
    ]
    if frag_rows:
        med_frag = float(np.median([r[2] for r in frag_rows]))
        stats["mediana_fragmentos_por_label"] = med_frag
        _log(f"\n[5] Fragmentação por label (subgrafo só com arestas same-label)")
        _log(f"    mediana de fragmentos por label: {med_frag:.0f}")
        _log("    labels mais partidos (label, nº pontos, nº fragmentos, maior fragmento %):")
        for r in frag_rows[:10]:
            _log(f"      label {r[0]:>4} | {r[1]:>7,} pts | {r[2]:>5} fragmentos | "
                 f"maior = {r[3]:.1f}%")

    # ---------- 6. Figuras ----------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes[0, 0].hist(deg, bins=np.arange(deg.max() + 2) - 0.5, color="#4C72B0")
    axes[0, 0].set_title("Grau dos nós"); axes[0, 0].set_xlabel("grau")

    axes[0, 1].hist(neighbor_dist.ravel(), bins=100, color="#999999", label="todos os candidatos")
    axes[0, 1].hist(neighbor_dist[edge_mask], bins=100, color="#55A868", alpha=0.8, label="aceitos")
    if distance_threshold is not None:
        axes[0, 1].axvline(distance_threshold, color="r", ls="--",
                           label="threshold" if distance_desc is None else "threshold (mediano)")
    axes[0, 1].set_title("Distância espacial dos vizinhos"); axes[0, 1].legend(); axes[0, 1].set_yscale("log")

    axes[0, 2].hist(color_dist.ravel(), bins=100, color="#999999", label="todos os candidatos")
    axes[0, 2].hist(color_dist[edge_mask], bins=100, color="#C44E52", alpha=0.8, label="aceitos")
    if color_dist_threshold is not None:
        axes[0, 2].axvline(color_dist_threshold, color="r", ls="--", label="threshold ΔE")
    axes[0, 2].set_title("ΔE entre vizinhos"); axes[0, 2].legend(); axes[0, 2].set_yscale("log")

    axes[1, 0].hist(und_w, bins=100, color="#8172B2")
    axes[1, 0].set_title("Peso das arestas"); axes[1, 0].set_xlabel("0.4*w_espacial + 1.2*w_cor")

    axes[1, 1].bar(["same-label", "diff-label", "com -1"], [n_same, n_diff, n_unlab],
                   color=["#55A868", "#C44E52", "#999999"])
    axes[1, 1].set_title(f"Pureza das arestas ({stats['pct_arestas_diff_label']:.2f}% diff)")
    axes[1, 1].set_yscale("log")

    axes[1, 2].hist(comp_sizes, bins=np.logspace(0, np.log10(max(comp_sizes.max(), 10)), 60),
                    color="#CCB974")
    axes[1, 2].set_xscale("log"); axes[1, 2].set_yscale("log")
    axes[1, 2].set_title(f"Tamanho das componentes ({n_comp:,} no total)")

    fig.suptitle("Diagnóstico do grafo inicial", fontsize=14)
    fig.tight_layout()
    hist_path = os.path.join(out_dir, "graph_hists.png")
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)

    # Projeções 2D das arestas: dá para "ver o grafo" sem abrir nenhum viewer.
    from matplotlib.collections import LineCollection
    n_plot = min(max_edges_plot, len(und))
    sel = rng.choice(len(und), size=n_plot, replace=False) if len(und) > n_plot else np.arange(len(und))
    e_plot = und[sel]
    is_diff = diff[sel]
    fig2, axs2 = plt.subplots(1, 2, figsize=(20, 9))
    for ax, (a, b), nm in zip(axs2, [(0, 1), (0, 2)], ["XY", "XZ"]):
        # xyz[e] tem shape (n_arestas, 2, 3); pegar duas colunas de coordenada
        # já entrega o formato (n_arestas, 2, 2) que o LineCollection espera.
        segs_ok = xyz[e_plot[~is_diff]][:, :, [a, b]]
        segs_bad = xyz[e_plot[is_diff]][:, :, [a, b]]
        ax.add_collection(LineCollection(segs_ok, colors="#6baed6", linewidths=0.25, alpha=0.6))
        ax.add_collection(LineCollection(segs_bad, colors="#d62728", linewidths=0.6, alpha=0.95))
        ax.set_title(f"Arestas projetadas em {nm} — vermelho = entre labels diferentes")
        ax.set_xlabel(nm[0]); ax.set_ylabel(nm[1])
        ax.set_xlim(xyz[:, a].min(), xyz[:, a].max())
        ax.set_ylim(xyz[:, b].min(), xyz[:, b].max())
        ax.set_aspect("equal")
    fig2.tight_layout()
    proj_path = os.path.join(out_dir, "graph_edges_proj.png")
    fig2.savefig(proj_path, dpi=140)
    plt.close(fig2)

    # ---------- 7. Exports 3D ----------
    n_exp = min(max_edges_export, len(und))
    sel_e = rng.choice(len(und), size=n_exp, replace=False) if len(und) > n_exp else np.arange(len(und))
    e_exp = und[sel_e]
    diff_exp = diff[sel_e]
    valid_exp = valid[sel_e]

    ec_label = np.tile(np.array([0.6, 0.6, 0.6]), (len(e_exp), 1))   # cinza: envolve label -1
    ec_label[valid_exp & ~diff_exp] = [0.15, 0.70, 0.25]             # verde: mesmo objeto
    ec_label[diff_exp] = [0.90, 0.10, 0.10]                          # vermelho: vaza entre objetos
    ec_weight = _colormap01(und_w[sel_e], "viridis")

    ply_label = os.path.join(out_dir, "graph_edges_label.ply")
    ply_weight = os.path.join(out_dir, "graph_edges_weight.ply")
    _write_ply_lineset(ply_label, xyz, point_rgb01, e_exp, ec_label)
    _write_ply_lineset(ply_weight, xyz, point_rgb01, e_exp, ec_weight)

    _write_ply_points(os.path.join(out_dir, "graph_nodes_degree.ply"), xyz, _colormap01(deg, "plasma"))
    _write_ply_points(os.path.join(out_dir, "graph_nodes_leak.ply"), xyz,
                      plt.get_cmap("coolwarm")(leak_frac)[:, :3])
    comp_rank = np.argsort(np.argsort(-comp_sizes))[comp_id]   # 0 = maior componente
    _write_ply_points(os.path.join(out_dir, "graph_components.ply"), xyz,
                      plt.get_cmap("tab20")((comp_rank % 20) / 19.0)[:, :3])

    # ---------- 8. Relatório ----------
    with open(os.path.join(out_dir, "graph_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "graph_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Diagnóstico do grafo salvo em: {out_dir}")
    print(f"    graph_report.txt / graph_stats.json  (números)")
    print(f"    graph_hists.png                      (histogramas)")
    print(f"    graph_edges_proj.png                 (arestas projetadas em XY/XZ)")
    print(f"    graph_edges_label.ply                ({n_exp:,} arestas: verde=same, vermelho=diff)")
    print(f"    graph_edges_weight.ply               (mesmas arestas coloridas pelo peso)")
    if n_diff > 0:
        print(f"    graph_leak_pairs.csv                 (todos os pares de labels colados)")
    if n_forced_e > 0:
        print(f"    graph_edges_forced.ply               (só as arestas impostas pelo grau mínimo)")
    print(f"    graph_nodes_degree.ply / graph_nodes_leak.ply / graph_components.ply")

    # ---------- 9. Janela interativa ----------
    if show:
        if not OPEN3D_AVAILABLE:
            print("    ⚠️ --graph_debug_show pedido, mas Open3D não está instalado.")
        else:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(xyz)
            ls.lines = o3d.utility.Vector2iVector(e_exp)
            ls.colors = o3d.utility.Vector3dVector(ec_label)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(point_rgb01))
            o3d.visualization.draw_geometries(
                [pcd, ls], window_name="Grafo inicial (verde=same-label, vermelho=diff-label)",
                width=1280, height=720)

    return stats



import colorsys
import random

def generate_distinct_colors(n, seed=42):
    """
    Gera N cores RGB maximamente distintas por amostragem greedy de maior
    distância no espaço perceptual CIELAB (farthest-point sampling).

    Diferente de uma roda de matizes (hue) simples, isso garante que a
    distância MÍNIMA entre QUAISQUER dois clusters seja a maior possível,
    evitando que clusters diferentes (ex: a perna do urso, o guardanapo e a
    cadeira) caiam todos na mesma faixa perceptual de "azul" mesmo tendo
    matizes numericamente distintos.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    rng = np.random.RandomState(seed)

    # Pool grande de candidatos cobrindo várias faixas de saturação/brilho,
    # para não ficar restrito a um único anel de matizes.
    hues = np.linspace(0, 1, 360, endpoint=False)
    sv_tiers = [(0.55, 0.95), (0.9, 0.9), (0.75, 0.6), (1.0, 0.75), (0.6, 0.6)]
    candidates_rgb = np.array(
        [colorsys.hsv_to_rgb(h, s, v) for s, v in sv_tiers for h in hues],
        dtype=np.float64,
    )
    candidates_lab = rgb2lab(candidates_rgb.reshape(1, -1, 3)).reshape(-1, 3)

    pool_size = len(candidates_rgb)
    n_unique = min(n, pool_size)
    if n > pool_size:
        print(f"⚠️ generate_distinct_colors: {n} clusters pedidos, mas só há "
              f"{pool_size} cores perceptualmente distintas no pool; algumas "
              f"cores serão reaproveitadas.")

    chosen = [rng.randint(pool_size)]
    min_dist = np.linalg.norm(candidates_lab - candidates_lab[chosen[0]], axis=1)
    for _ in range(1, n_unique):
        next_idx = int(np.argmax(min_dist))
        chosen.append(next_idx)
        d = np.linalg.norm(candidates_lab - candidates_lab[next_idx], axis=1)
        min_dist = np.minimum(min_dist, d)

    colors = candidates_rgb[chosen]
    if n > n_unique:
        reps = int(np.ceil(n / n_unique))
        colors = np.tile(colors, (reps, 1))[:n]

    return colors.astype(np.float32)


def cameras_para_renderizar(scene, args):
    """
    Devolve as câmeras que devem ser renderizadas, aplicando o filtro opcional
    de --render_frames.

    Motivação: as métricas 2D (mIoU/PQ) do LERF-OVS só têm ground-truth para
    um punhado de frames por cena (4 a 7), mas o render varre as ~300 câmeras
    de treino. Num sweep de hiperparâmetros isso domina o tempo total de cada
    rodada sem produzir nenhuma informação nova — os outros ~292 frames nunca
    são comparados com nada. Restringir aos frames anotados corta o passe de
    render em ~50x e deixa o sweep viável.

    Sem --render_frames o comportamento é o de sempre (todas as câmeras), então
    as rodadas de resultado final continuam idênticas às anteriores.
    """
    cams = scene.getTrainCameras()
    nomes = getattr(args, "render_frames", None)
    if not nomes:
        return cams

    alvo = {n.strip() for n in nomes.split(",") if n.strip()}
    filtradas = [c for c in cams if c.image_name in alvo]
    if not filtradas:
        print(f"⚠️ --render_frames não casou com nenhuma câmera ({len(alvo)} nomes pedidos); "
              f"renderizando todas as {len(cams)}.")
        return cams

    faltando = alvo - {c.image_name for c in filtradas}
    if faltando:
        print(f"⚠️ --render_frames: {len(faltando)} nome(s) sem câmera correspondente: "
              f"{sorted(faltando)[:5]}")
    print(f"🎬 Renderizando apenas {len(filtradas)} de {len(cams)} câmeras (--render_frames).")
    return filtradas


# Renderizar é inferência pura — ver a nota em render_clusters.
@torch.no_grad()
def render_embedding_pca(emb, gaussians, scene, pipe, background, args, device,
                         mask_filter=None, out_name="PCA"):
    """
    Renderiza o EMBEDDING aprendido, e não os clusters: reduz os vetores de
    saída da GNN a 3 dimensões por PCA, mapeia essas componentes em RGB e
    renderiza uma imagem por câmera de treino em `<output>/PCA`.

    Serve para ver o que a rede aprendeu ANTES do HDBSCAN entrar. As imagens de
    cluster mostram a partição já decidida, então não distinguem "o embedding
    não separa nada" de "o embedding separa e o HDBSCAN cortou mal" — os dois
    casos produzem manchas erradas.

    ATENÇÃO ao ler a imagem: as componentes são normalizadas por percentil, ou
    seja, o contraste é REESCALADO para preencher a faixa de cor inteira. Um
    embedding colapsado NÃO sai cinza — sai colorido do mesmo jeito, porque a
    normalização amplifica a variação residual. Medido com embeddings
    sintéticos, o desvio de cor entre gaussianas foi 0.21 no caso colapsado
    contra 0.30 no estruturado: perto demais para se julgar a olho.

    Quem carrega o sinal de colapso são os dois números impressos abaixo:

    - AMPLITUDE (faixa p1-p99 da projeção, antes de normalizar). É a escala
      real do embedding. Colapso deixa isso perto de zero.
    - VARIÂNCIA EXPLICADA pelas 3 primeiras componentes. Contra-intuitivo:
      valor BAIXO indica colapso, não alto. Num embedding colapsado o que
      sobra é ruído isotrópico espalhado por todas as dimensões, e 3 de 32
      dimensões explicam ~3/32 ≈ 9%. Com estrutura real a variância se
      concentra em poucas direções e as 3 primeiras passam de 50%.

    O que a imagem mostra bem é a GEOMETRIA: se objetos distintos aparecem em
    tons próprios com bordas nítidas, ou se a cor varia suavemente pela cena
    sem respeitar fronteira de objeto.
    """
    if emb is None or len(emb) == 0:
        print("⚠️ render_embedding_pca: embedding vazio, pulando.")
        return

    # PCA por SVD econômica. Em (N x 32) isso é barato e evita depender de
    # sklearn.decomposition, que não é importado neste arquivo.
    X = np.asarray(emb, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    n_comp = min(3, Vt.shape[0])
    proj = X @ Vt[:n_comp].T
    if n_comp < 3:  # embedding com menos de 3 dimensões: completa com zeros
        proj = np.concatenate([proj, np.zeros((len(proj), 3 - n_comp))], axis=1)

    var = S ** 2
    var_ratio = var / (var.sum() + 1e-12)

    # Normalização por PERCENTIL (1-99) e não por min-max: um punhado de
    # gaussianas extremas comprimiria todo o resto num tom só, escondendo
    # justamente a estrutura que se quer ver.
    lo = np.percentile(proj, 1, axis=0)
    hi = np.percentile(proj, 99, axis=0)
    amplitude = float(np.mean(hi - lo))

    print(f"🎨 PCA do embedding:")
    print(f"   Amplitude (faixa p1-p99 da projeção): {amplitude:.4f}"
          + ("   ⚠️ perto de zero: embedding colapsado" if amplitude < 0.05 else ""))
    print(f"   Variância explicada pelas 3 primeiras componentes: "
          f"{100 * var_ratio[:3].sum():.2f}% "
          f"({', '.join(f'{100*v:.1f}%' for v in var_ratio[:3])})"
          + (f"   ⚠️ abaixo de ~{100*3/max(X.shape[1],1):.0f}% = ruído isotrópico"
             if var_ratio[:3].sum() < 1.5 * 3 / max(X.shape[1], 1) else ""))
    print(f"   Lembre: a imagem é normalizada por percentil, então colapso NÃO "
          f"aparece como cor uniforme — use os números acima.")
    rgb = np.clip((proj - lo) / (hi - lo + 1e-12), 0.0, 1.0).astype(np.float32)

    n_total = gaussians._xyz.shape[0]
    all_colors = np.zeros((n_total, 3), dtype=np.float32)
    if mask_filter is not None:
        if int(np.sum(mask_filter)) != len(rgb):
            print(f"⚠️ render_embedding_pca: mask_filter seleciona "
                  f"{int(np.sum(mask_filter)):,} gaussianas mas o embedding tem "
                  f"{len(rgb):,}. Pulando.")
            return
        all_colors[mask_filter] = rgb
    else:
        if n_total != len(rgb):
            print(f"⚠️ render_embedding_pca: {n_total:,} gaussianas mas o "
                  f"embedding tem {len(rgb):,}. Pulando.")
            return
        all_colors = rgb

    colors_tensor = torch.tensor(all_colors, dtype=torch.float32, device=device)

    orig_data = {
        'opacity': gaussians._opacity.data.clone(),
        'features_dc': gaussians._features_dc.data.clone(),
        'features_rest': gaussians._features_rest.data.clone(),
    }

    gaussians._features_dc.data = (colors_tensor.unsqueeze(1) - 0.5) / 0.28209
    # Sem zerar os SH de ordem maior, a variação especular da cena original se
    # soma por cima da cor do embedding e a imagem deixa de representar o PCA.
    gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)
    if mask_filter is not None:
        nova_opacidade = orig_data['opacity'].clone()
        fora = ~torch.tensor(mask_filter, device=device, dtype=torch.bool)
        nova_opacidade[fora] = -10.0  # sigmoid(-10) ~ 0
        gaussians._opacity.data = nova_opacidade

    out_dir = os.path.join(args.output, out_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"🖼️ Renderizando PCA do embedding para {out_dir} ...")

    try:
        for view in cameras_para_renderizar(scene, args):
            img = render(view, gaussians, pipe, background)["render"]
            img_np = np.clip(img.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"),
                        cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        gaussians._opacity.data = orig_data['opacity']
        gaussians._features_dc.data = orig_data['features_dc']
        gaussians._features_rest.data = orig_data['features_rest']

    print(f"✅ PCA do embedding salvo em: {out_dir}")


# Renderizar é inferência pura: sem @torch.no_grad() cada chamada ao
# rasterizador (uma autograd.Function) guarda geomBuffer/binningBuffer/imgBuffer
# para um backward que nunca acontece, e as saídas encadeadas em `melhor_valor`
# e `soma_coverage` mantêm vivos os buffers de TODAS as chamadas da câmera. O
# custo cresce com o número de clusters — com 68 clusters (23 chamadas por
# câmera) cabia nos 8 GB, com 209 (70 chamadas) estourava com 6,7 GB presos.
@torch.no_grad()
def render_clusters(labels, name, gaussians, scene, pipe, background, args, device, mask_filter=None, color_map=None):
    """
    Renderiza Gaussianas coloridas por cluster.
    Gaussianas fora do mask_filter têm opacidade ZERO (invisíveis).

    Pass `color_map` (dict: label -> RGB) to reuse the exact same palette as
    another render/point-cloud export (e.g. save_pointcloud_with_metadata),
    so the same cluster gets the same color everywhere.
    """
    if len(labels) == 0:
        raise ValueError(
            "render_clusters recebeu 0 gaussianas. Isso indica que o filtro "
            "anterior (opacidade/escala/label/bbox) não selecionou nada — "
            "verifique --target_gaussians e o log do filtro dinâmico, em vez "
            "de procurar o problema na renderização."
        )

    unique_labels = np.unique(labels)
    valid_labels = [l for l in unique_labels if l != -1]

    if color_map is None:
        colors = generate_distinct_colors(len(valid_labels))
        color_map = {label: colors[i] for i, label in enumerate(valid_labels)}
        color_map[-1] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    num_clusters = len(valid_labels)

    # ==========================================================
    # 1. Preparar cores para TODAS as Gaussianas
    # ==========================================================
    n_total = gaussians._xyz.shape[0]
    
    # Inicializar com cor preta para todas
    all_colors = np.zeros((n_total, 3), dtype=np.float32)
    
    if mask_filter is not None:
        # Apenas as Gaussianas filtradas recebem cores dos clusters
        all_colors[mask_filter] = np.array([color_map[l] for l in labels])
    else:
        # Todas as Gaussianas recebem cores
        all_colors = np.array([color_map[l] for l in labels])
    
    colors_tensor = torch.tensor(all_colors, dtype=torch.float32, device=device)

    # ==========================================================
    # 2. Salvar dados originais
    # ==========================================================
    orig_data = {
        'xyz': gaussians._xyz.data.clone(),
        'opacity': gaussians._opacity.data.clone(),
        'features_dc': gaussians._features_dc.data.clone(),
        'features_rest': gaussians._features_rest.data.clone(),
        'scaling': gaussians._scaling.data.clone(),
        'rotation': gaussians._rotation.data.clone()
    }

    # ==========================================================
    # 3. Aplicar cores e opacidade
    # ==========================================================
    # 3a. Converter cores para SH (Spherical Harmonics)
    colors_sh = (colors_tensor.unsqueeze(1) - 0.5) / 0.28209
    gaussians._features_dc.data = colors_sh
    
    # 3b. 🔑 CORAÇÃO DA SOLUÇÃO: Opacidade ZERO para Gaussianas fora do mask_filter
    if mask_filter is not None:
        # Criar tensor de opacidade com valores originais
        new_opacity = orig_data['opacity'].clone()
        
        # ZERAR opacidade das Gaussianas que NÃO estão no mask_filter
        mask_inv = ~torch.tensor(mask_filter, device=device, dtype=torch.bool)
        new_opacity[mask_inv] = -10.0  # sigmoid(-10) ≈ 0
        
        gaussians._opacity.data = new_opacity
    else:
        # Manter opacidade original
        gaussians._opacity.data = orig_data['opacity']
    
    # 3c. Zerar features_rest (harmônicos esféricos de ordem > 0). Se deixados
    # com os valores originais, cada Gaussiana mantém sua variação de cor
    # dependente de ponto de vista (especular/sombra) da cena original por
    # cima da cor plana do cluster, o que por si só já tornaria a máscara
    # não-homogênea mesmo sem nenhuma mistura alfa entre Gaussianas vizinhas.
    gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)

    # ==========================================================
    # 4. Renderizar
    # ==========================================================
    # Duas saídas: a renderização crua do rasterizador (com o blending alfa
    # normal, sem nenhum ajuste de cor) e a versão com "snap" de cor
    # homogêneo por cluster. Mantém `out_dir` (a versão homogênea) no mesmo
    # caminho de sempre, para não quebrar nada que já dependa dele.
    out_dir = os.path.join(args.output, name)
    raw_dir = os.path.join(args.output, f"{name}_raw")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    n_active = np.sum(mask_filter) if mask_filter is not None else n_total
    print(f"Rendering {num_clusters} discrete clusters...")
    print(f"  Cru (sem padronização de cor): {raw_dir}")
    print(f"  Padronizado (cor homogênea por cluster): {out_dir}")
    print(f"  Gaussianas totais: {n_total:,}")
    print(f"  Gaussianas ativas (opacity>0): {n_active:,}")
    print(f"  Gaussianas ocultas (opacity=0): {n_total - n_active:,}")

    # 4a. Paleta de referência para o "snap" de cor pós-render: o rasterizador
    # mistura (alpha-blend) as cores de todas as Gaussianas que caem no mesmo
    # raio, então mesmo com uma cor única por cluster, os pixels resultantes
    # não são homogêneos (halos/gradientes nas bordas). Em vez de depender de
    # um pós-processo externo (CRF), forçamos cada pixel de primeiro plano
    # para a cor EXATA do cluster mais próximo em CIELAB.
    palette_labels = list(color_map.keys())
    palette_rgb = np.array([color_map[l] for l in palette_labels], dtype=np.float64)
    palette_lab = rgb2lab(palette_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    color_tree = cKDTree(palette_lab)
    bg_np = background.detach().cpu().numpy()

    for view in cameras_para_renderizar(scene, args):
        render_pkg = render(view, gaussians, pipe, background)
        img = render_pkg["render"]
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)

        # 4a. Salva a versão CRUA (blending alfa do rasterizador, sem ajuste).
        raw_uint8 = (img_np * 255).astype(np.uint8)
        raw_bgr = cv2.cvtColor(raw_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(raw_dir, f"{view.image_name}.png"), raw_bgr)

        # 4b. Aplica o snap de cor e salva a versão PADRONIZADA (homogênea).
        h, w, c = img_np.shape
        flat_img = img_np.reshape(-1, 3).copy()

        # Só faz snap dos pixels que não são o fundo; o fundo (branco) fica intacto.
        # FOREGROUND_DIST_THRESHOLD = 0.3 (era 0.05): pixels de névoa/floaters
        # residuais (opacidade baixíssima) ficam bem perto do branco — na prática
        # medi distância mediana ~0.12 do branco para essa névoa, contra um mínimo
        # de ~0.43 para pixels de clusters sólidos de verdade. Um threshold de 0.05
        # tratava qualquer leve tingimento como "confiante", forçando o snap de cor
        # em pixels quase-brancos para a cor de PALETA mais próxima (geralmente uma
        # cor pouco saturada), o que pintava a névoa inteira de uma única cor
        # dominante cobrindo boa parte da cena. 0.3 fica seguramente abaixo do piso
        # observado para objetos reais, então não arrisca apagar nenhum pixel
        # legítimo — só a névoa de fundo.
        FOREGROUND_DIST_THRESHOLD = 0.3
        fg_mask = np.linalg.norm(flat_img - bg_np, axis=1) > FOREGROUND_DIST_THRESHOLD
        if np.any(fg_mask):
            flat_lab = rgb2lab(flat_img[fg_mask].reshape(1, -1, 3)).reshape(-1, 3)
            _, nearest_idx = color_tree.query(flat_lab)
            flat_img[fg_mask] = palette_rgb[nearest_idx]
        flat_img[~fg_mask] = bg_np

        uniform_img_np = flat_img.reshape(h, w, c)
        uniform_uint8 = (uniform_img_np * 255).astype(np.uint8)
        uniform_bgr = cv2.cvtColor(uniform_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"), uniform_bgr)

    # ==========================================================
    # 5. Restaurar dados originais
    # ==========================================================
    gaussians._xyz.data = orig_data['xyz']
    gaussians._opacity.data = orig_data['opacity']
    gaussians._features_dc.data = orig_data['features_dc']
    gaussians._features_rest.data = orig_data['features_rest']
    gaussians._scaling.data = orig_data['scaling']
    gaussians._rotation.data = orig_data['rotation']


# Renderizar é inferência pura: sem @torch.no_grad() cada chamada ao
# rasterizador (uma autograd.Function) guarda geomBuffer/binningBuffer/imgBuffer
# para um backward que nunca acontece, e as saídas encadeadas em `melhor_valor`
# e `soma_coverage` mantêm vivos os buffers de TODAS as chamadas da câmera. O
# custo cresce com o número de clusters — com 68 clusters (23 chamadas por
# câmera) cabia nos 8 GB, com 209 (70 chamadas) estourava com 6,7 GB presos.
@torch.no_grad()
def render_clusters_coverage(labels, name, gaussians, scene, pipe, args, device, mask_filter=None):
    """
    Gera a máscara de segmentação SEM decodificar cor. Em vez de pintar cada
    cluster com uma cor RGB e depois tentar recuperar a classe a partir do
    pixel renderizado (o alpha-blending do rasterizador sempre mistura cores
    nas bordas e em regiões com Gaussianas sobrepostas, então essa decodificação
    é um problema mal-posto por construção — é o que `render_clusters` tenta
    remediar com o "snap" para a cor mais próxima em CIELAB, e o que os scripts
    de CRF tentam remediar depois disso), renderizamos um indicador one-hot por
    cluster através do MESMO compositing alfa do rasterizador. O valor
    acumulado em cada pixel é exatamente a fração do peso do raio (Σ α_i·T_i)
    pertencente àquele cluster — uma grandeza bem definida mesmo em bordas e
    oclusões, ao contrário da cor resultante. O argmax dessa pilha já É a
    máscara discreta: não sobra nenhuma cor "ambígua" para KMeans/CRF resolver
    depois.

    Empacota 3 clusters por passada de rasterização (um por canal R/G/B), já
    que os canais são compostos de forma independente no alpha blending —
    corta em ~3x o número de renderizações necessárias.

    Salva em `{args.output}/{name}_coverage_labels/*.npy` (int32 HxW, valores
    0..num_clusters-1 = cluster, num_clusters = fundo/não coberto por nenhum
    cluster) — essa é a fonte de verdade para qualquer métrica, sem precisar
    de CRF/KMeans a jusante. Também salva `{name}_coverage/*.png`, uma versão
    colorida só para inspeção visual humana: como a cor é atribuída DEPOIS do
    argmax, cada região sai solida de verdade, sem halo de mistura nas bordas.
    """
    unique_labels = np.unique(labels)
    valid_labels = [l for l in unique_labels if l != -1]
    num_clusters = len(valid_labels)
    if num_clusters == 0:
        print("⚠️ render_clusters_coverage: nenhum cluster válido, abortando.")
        return

    label_to_idx = {l: i for i, l in enumerate(valid_labels)}

    n_total = gaussians._xyz.shape[0]
    cluster_idx_per_gaussian = np.full(n_total, -1, dtype=np.int64)
    if mask_filter is not None:
        cluster_idx_per_gaussian[mask_filter] = np.array([label_to_idx.get(l, -1) for l in labels])
    else:
        cluster_idx_per_gaussian[:] = np.array([label_to_idx.get(l, -1) for l in labels])
    cluster_idx_t = torch.tensor(cluster_idx_per_gaussian, device=device)

    out_dir = os.path.join(args.output, f"{name}_coverage")
    labels_dir = os.path.join(args.output, f"{name}_coverage_labels")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Opacidade zero para Gaussianas fora do mask_filter, igual ao render_clusters,
    # para o fundo bater com o das saídas "images"/"images_raw" já existentes.
    orig_opacity = gaussians._opacity.data.clone()
    if mask_filter is not None:
        mask_inv = ~torch.tensor(mask_filter, device=device, dtype=torch.bool)
        new_opacity = orig_opacity.clone()
        new_opacity[mask_inv] = -10.0
        gaussians._opacity.data = new_opacity

    black_bg = torch.zeros(3, dtype=torch.float32, device=device)
    palette = generate_distinct_colors(num_clusters)

    print(f"Rendering {num_clusters} clusters como coverage (sem decodificar cor)...")
    print(f"  Labels (fonte de verdade): {labels_dir}")
    print(f"  Preview colorido: {out_dir}")

    for view in cameras_para_renderizar(scene, args):
        H, W = int(view.image_height), int(view.image_width)

        # O argmax sobre os canais é acumulado de forma incremental, guardando
        # só o máximo corrente e seu índice. Materializar (num_clusters, H, W)
        # custava mais de 700 MB por câmera com algumas centenas de clusters
        # (min_cluster_size baixo) e estourava a VRAM. O resultado é idêntico:
        # a comparação estrita `>` mantém o primeiro máximo, que é o mesmo
        # critério de desempate do torch.argmax.
        melhor_valor = torch.full((H, W), float("-inf"), device=device)
        melhor_idx = torch.zeros((H, W), dtype=torch.long, device=device)
        soma_coverage = torch.zeros((H, W), device=device)

        for start in range(0, num_clusters, 3):
            group = list(range(start, min(start + 3, num_clusters)))
            group_colors = torch.zeros(n_total, 3, device=device)
            for channel, c in enumerate(group):
                group_colors[:, channel] = (cluster_idx_t == c).float()
            render_pkg = render_uniform(view, gaussians, pipe, black_bg, group_colors, device=device)
            for channel, c in enumerate(group):
                canal = render_pkg["render"][channel]
                soma_coverage += canal
                troca = canal > melhor_valor
                melhor_valor = torch.where(troca, canal, melhor_valor)
                melhor_idx = torch.where(troca, c, melhor_idx)
            # Solta as referências antes da próxima alocação. Com
            # min_cluster_size baixo este laço roda ~70 vezes por câmera.
            del render_pkg, group_colors

        # O fundo entra como o último índice (num_clusters) e, no argmax
        # original, só vencia se fosse ESTRITAMENTE maior que todos os canais.
        bg_coverage = (1.0 - soma_coverage).clamp(min=0)
        label_map = torch.where(bg_coverage > melhor_valor,
                                torch.tensor(num_clusters, device=device),
                                melhor_idx)
        label_map = label_map.cpu().numpy().astype(np.int32)

        np.save(os.path.join(labels_dir, f"{view.image_name}.npy"), label_map)

        color_img = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(num_clusters):
            color_img[label_map == c] = (palette[c] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"),
                    cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

        del melhor_valor, melhor_idx, soma_coverage
        torch.cuda.empty_cache()

    gaussians._opacity.data = orig_opacity

    return label_to_idx





# MAIN
# ==========================================================
# ==========================================================
# EARLY STOPPING
# ==========================================================
import copy

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        # Verifica se a melhoria foi maior que o min_delta estipulado
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # Usa deepcopy para garantir uma cópia estática dos pesos em memória
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"   ↳ [EarlyStopping] Melhora detectada! Loss: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"   ↳ [EarlyStopping] Sem melhora significativa há {self.counter}/{self.patience} épocas.")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"   🛑 [EarlyStopping] Gatilho ativado na época por falta de progresso.")
        
        return self.early_stop

    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"   ✅ [EarlyStopping] Melhores pesos restaurados com sucesso (Melhor Loss: {self.best_loss:.4f}).")


import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode




from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
from datetime import datetime

def evaluate_semi_supervised(initial_labels, refined_labels):
    """
    Avaliação Semi-Supervisionada (Invariante a permutações de IDs)
    """
    valid_mask = (initial_labels != -1) & (refined_labels != -1)
    
    if valid_mask.sum() == 0:
        print("⚠️ Nenhum ponto válido para avaliação semi-supervisionada!")
        return {'ARI': 0.0, 'NMI': 0.0, 'Valid_Points': 0}
    
    initial_filtered = initial_labels[valid_mask]
    refined_filtered = refined_labels[valid_mask]
    
    # ARI e NMI lidam corretamente com a troca de IDs entre algoritmos
    ari = adjusted_rand_score(initial_filtered, refined_filtered)
    nmi = normalized_mutual_info_score(initial_filtered, refined_filtered)
    
    n_initial = len(np.unique(initial_filtered))
    n_refined = len(np.unique(refined_filtered))
    
    print(f"\n📊 Avaliação Semi-Supervisionada:")
    print(f"  ARI: {ari:.4f} (0 = aleatório, 1 = perfeito)")
    print(f"  NMI: {nmi:.4f} (0 = sem informação, 1 = perfeito)")
    print(f"  Clusters iniciais: {n_initial} → Clusters refinados: {n_refined}")
    
    return {
        'ARI': ari,
        'NMI': nmi,
        'Initial_Clusters': n_initial,
        'Refined_Clusters': n_refined,
        'Valid_Points': int(valid_mask.sum())
    }

def evaluate_cluster_validity(embeddings, labels, xyz=None, max_samples_silhouette=10000):
    """
    Métricas de Coerência Interna com subamostramento para evitar estouro de memória RAM.
    """
    valid_mask = labels != -1
    
    if valid_mask.sum() == 0:
        print("⚠️ Nenhum cluster válido para avaliar!")
        return {
            'Silhouette': -1.0,
            'Davies_Bouldin': float('inf'),
            'Calinski_Harabasz': 0.0,
            'N_Valid_Points': 0,
            'N_Clusters': 0
        }
    
    emb_valid = embeddings[valid_mask]
    labels_valid = labels[valid_mask]
    
    n_clusters = len(np.unique(labels_valid))
    if n_clusters < 2:
        return {'Silhouette': -1.0, 'Davies_Bouldin': float('inf'), 'Calinski_Harabasz': 0.0, 'N_Clusters': n_clusters}

    # Silhouette com subamostramento para não estourar a memória RAM (O(N^2))
    try:
        sample_size = min(len(emb_valid), max_samples_silhouette)
        silhouette = silhouette_score(emb_valid, labels_valid, metric='cosine', sample_size=sample_size)
    except Exception as e:
        silhouette = -1.0
    
    try:
        davies_bouldin = davies_bouldin_score(emb_valid, labels_valid)
    except Exception as e:
        davies_bouldin = float('inf')
    
    try:
        calinski_harabasz = calinski_harabasz_score(emb_valid, labels_valid)
    except Exception as e:
        calinski_harabasz = 0.0
    
    unique, counts = np.unique(labels_valid, return_counts=True)
    mean_size = counts.mean()
    std_size = counts.std()
    
    print(f"\n📊 Coerência Interna dos Clusters:")
    print(f"  Silhouette Score: {silhouette:.4f} (amostrado com N={sample_size})")
    print(f"  Davies-Bouldin: {davies_bouldin:.4f} (quanto menor, melhor)")
    print(f"  Calinski-Harabasz: {calinski_harabasz:.2f} (quanto maior, melhor)")
    print(f"  Clusters: {n_clusters} | Tamanho médio: {mean_size:.1f} ± {std_size:.1f}")
    
    return {
        'Silhouette': silhouette,
        'Davies_Bouldin': davies_bouldin,
        'Calinski_Harabasz': calinski_harabasz,
        'N_Clusters': n_clusters,
        'Mean_Cluster_Size': mean_size,
        'Std_Cluster_Size': std_size,
        'N_Valid_Points': int(valid_mask.sum())
    }
from scipy.spatial.distance import pdist
def evaluate_spatial_consistency(xyz, labels, max_samples_per_cluster=2000):
    spatial_metrics = {}
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Ignora ruído
    
    intra_cluster_distances = []
    
    for label in unique_labels:
        cluster_points = xyz[labels == label]
        
        # Subamostragem para evitar estouro de memória (MemoryError no pdist)
        if len(cluster_points) > max_samples_per_cluster:
            indices = np.random.choice(len(cluster_points), size=max_samples_per_cluster, replace=False)
            cluster_points_sampled = cluster_points[indices]
        else:
            cluster_points_sampled = cluster_points
            
        if len(cluster_points_sampled) > 1:
            # pdist agora consumirá no máximo ~15MB em vez de 94GB
            distances = pdist(cluster_points_sampled)
            intra_cluster_distances.append(np.mean(distances))
            
    spatial_metrics['Mean_Intra_Cluster_Distance'] = float(np.mean(intra_cluster_distances)) if intra_cluster_distances else 0.0
    return spatial_metrics


def plot_loss_curve(loss_history, output_path, model_name="GAT", training_time=None,):
    """
    Plota a curva de loss do treinamento com valores nos eixos.
    
    Args:
        loss_history: Lista com os valores de loss por época
        output_path: Caminho para salvar a imagem
        model_name: Nome do modelo para o título
        training_time: Tempo total de treinamento em segundos (opcional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(14, 8))
    
    epochs = range(1, len(loss_history) + 1)
    
    # Plota a curva principal
    plt.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    
    # Encontra o melhor loss e marca
    best_epoch = np.argmin(loss_history) + 1
    best_loss = min(loss_history)
    plt.scatter(best_epoch, best_loss, color='red', s=120, zorder=5, 
                label=f'Best Loss: {best_loss:.6f} (Epoch {best_epoch})')
    
    # Adiciona linha vertical no melhor ponto
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Adiciona linha horizontal no melhor valor
    plt.axhline(y=best_loss, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Configurações do gráfico
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    
    # Adiciona informações no título
    title = f'Curva de Aprendizado - {model_name}'
    if training_time is not None:
        minutes = int(training_time // 60)
        seconds = int(training_time % 60)
        title += f'  |  Tempo: {minutes}m {seconds}s'
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Ajusta limites dos eixos
    plt.xlim(0, len(loss_history) + 1)
    y_min = min(loss_history) - 0.1 * (max(loss_history) - min(loss_history)) if len(loss_history) > 1 else min(loss_history) - 0.1
    y_max = max(loss_history) + 0.1 * (max(loss_history) - min(loss_history)) if len(loss_history) > 1 else max(loss_history) + 0.1
    plt.ylim(max(0, y_min), y_max)
    
    # Adiciona valores nos eixos X (mostra alguns epochs)
    step = max(1, len(loss_history) // 20)
    plt.xticks(np.arange(0, len(loss_history) + 1, step))
    
    # Adiciona anotações com valores de loss em alguns pontos
    for i in range(0, len(loss_history), max(1, len(loss_history) // 10)):
        plt.annotate(f'{loss_history[i]:.3f}', 
                    (i+1, loss_history[i]),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=8,
                    alpha=0.7)
    
    # Adiciona o valor final destacado
    if len(loss_history) > 0:
        final_loss = loss_history[-1]
        plt.annotate(f'Final: {final_loss:.4f}', 
                    (len(loss_history), final_loss),
                    textcoords="offset points", 
                    xytext=(0, -15), 
                    ha='center',
                    fontsize=10,
                    fontweight='bold',
                    color='blue')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Curva de loss salva em: {output_path}")


def save_experiment_results(metrics, loss_history, output_dir, model_name="GAT", training_time=None, graph_build_time=None, hdbscan_time=None, max_vram_gb=None, training_vram_gb=None):
    """
    Salva todos os resultados do experimento em formato organizado, incluindo
    tempos de construção do grafo e treinamento.

    Args:
        metrics: Dict com todas as métricas
        loss_history: Lista com histórico de loss
        output_dir: Diretório para salvar
        model_name: Nome do modelo
        training_time: Tempo de treinamento
        graph_build_time: Tempo de construção do grafo
        hdbscan_time: Tempo do HDBSCAN
        max_vram_gb: Pico de VRAM reservada pelo allocator do PyTorch, medido
            no fim do pipeline inteiro (opcional; subestima o valor real do
            nvidia-smi, ver NvmlVRAMSampler)
        training_vram_gb: Pico de VRAM real do processo (via NVML, mesma
            fonte que o nvidia-smi), amostrado só durante o laço de treino
            da rede (opcional)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Função auxiliar para formatar segundos em min/seg
    def format_time_str(t_sec):
        if t_sec is None:
            return "N/A", 0.0
        m, s = divmod(t_sec, 60)
        time_str = f"{int(m)}m {int(s)}s" if m > 0 else f"{s:.2f}s"
        return time_str, float(t_sec)

    train_str, train_sec = format_time_str(training_time)
    graph_str, graph_sec = format_time_str(graph_build_time)
    hdbscan_str, hdbscan_sec = format_time_str(hdbscan_time)
    total_sec = train_sec + graph_sec + hdbscan_sec
    total_str, _ = format_time_str(total_sec)

    # ==========================================================
    # 1. SALVAR MÉTRICAS EM EXCEL
    # ==========================================================
    excel_path = os.path.join(output_dir, f"{model_name}_metrics_{timestamp}.xlsx")
    
    metrics_dict = {
        'Métricas de Clusterização 3D': {
            'ARI (Adjusted Rand Index)': metrics.get('ARI', 'N/A'),
            'NMI (Normalized Mutual Info)': metrics.get('NMI', 'N/A'),
            'Silhouette Score': metrics.get('Silhouette', 'N/A'),
            'Davies-Bouldin Index': metrics.get('Davies_Bouldin', 'N/A'),
            'Calinski-Harabasz': metrics.get('Calinski_Harabasz', 'N/A'),
        },
        'Estatísticas dos Clusters': {
            'Clusters Iniciais (DEVA/SAM)': metrics.get('Initial_Clusters', 'N/A'),
            'Clusters Finais (HDBSCAN)': metrics.get('Refined_Clusters', 'N/A'),
            'Pontos Válidos': metrics.get('Valid_Points', 'N/A'),
            'Tamanho Médio dos Clusters': metrics.get('Mean_Cluster_Size', 'N/A'),
            'Desvio Padrão dos Clusters': metrics.get('Std_Cluster_Size', 'N/A'),
        },
        'Métricas Espaciais': {
            'Distância Intra-Cluster Média': metrics.get('Mean_Intra_Cluster_Distance', 'N/A'),
        },
        'Configuração do Experimento': {
            'Modelo': metrics.get('Model_Type', 'N/A'),
            'GCN Depth': metrics.get('GCN_Depth', 'N/A'),
            'Cabeças de Atenção (GAT)': metrics.get('GAT_Heads', 'N/A'),
            'Min Cluster Size': metrics.get('Min_Cluster_Size', 'N/A'),
            'KNN Reassign K': metrics.get('KNN_Reassign_K', 'N/A'),
            'Total de Gaussianas (Raw)': metrics.get('Raw_Gaussians', 'N/A'),
            'Gaussianas Filtradas': metrics.get('Filtered_Gaussians', 'N/A'),
        },
        'Desempenho Computacional': {
            'Tempo de Construção do Grafo': graph_str,
            'Tempo de Construção do Grafo (segundos)': round(graph_sec, 2),
            'Tempo de Treinamento': train_str,
            'Tempo de Treinamento (segundos)': round(train_sec, 2),
            'Tempo HDBSCAN': hdbscan_str,
            'Tempo HDBSCAN (segundos)': round(hdbscan_sec, 2),
            'Tempo Total': total_str,
            'Tempo Total (segundos)': round(total_sec, 2),
            'Pico de VRAM Reservada (GB)': round(max_vram_gb, 2) if max_vram_gb is not None else 'N/A',
            'VRAM Real Durante o Treino - NVML (GB)': round(training_vram_gb, 2) if training_vram_gb is not None else 'N/A',
        },
        'Resumo do Treinamento': {
            'Épocas Treinadas': len(loss_history),
            'Loss Final': loss_history[-1] if loss_history else 'N/A',
            'Melhor Loss': min(loss_history) if loss_history else 'N/A',
            'Época do Melhor Loss': np.argmin(loss_history) + 1 if loss_history else 'N/A',
        }
    }
    
    rows = []
    for category, values in metrics_dict.items():
        for key, value in values.items():
            rows.append({
                'Categoria': category,
                'Métrica': key,
                'Valor': value
            })
    
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_excel(excel_path, index=False, sheet_name='Métricas')
    
    if loss_history:
        df_loss = pd.DataFrame({
            'Época': range(1, len(loss_history) + 1),
            'Loss': loss_history
        })
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
            df_loss.to_excel(writer, sheet_name='Loss History', index=False)
    
    print(f"📊 Tabela de métricas salva em: {excel_path}")
    
    # ==========================================================
    # 2. SALVAR RELATÓRIO EM TXT
    # ==========================================================
    txt_path = os.path.join(output_dir, f"{model_name}_report_{timestamp}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO DE EXPERIMENTO - {model_name}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for category, values in metrics_dict.items():
            f.write(f"\n{category}:\n")
            f.write("-"*40 + "\n")
            for key, value in values.items():
                if isinstance(value, float):
                    f.write(f"  {key:40s}: {value:.4f}\n")
                else:
                    f.write(f"  {key:40s}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("HISTÓRICO DE LOSS\n")
        f.write("="*80 + "\n")
        f.write(f"{'Época':>8} | {'Loss':>12}\n")
        f.write("-"*24 + "\n")
        for i, loss in enumerate(loss_history):
            f.write(f"{i+1:>8} | {loss:>12.6f}\n")
    
    print(f"📄 Relatório salvo em: {txt_path}")
    
    # ==========================================================
    # 3. SALVAR CURVA DE LOSS
    # ==========================================================
    if loss_history:
        loss_plot_path = os.path.join(output_dir, f"{model_name}_loss_curve.png")
        plot_loss_curve(loss_history, loss_plot_path, model_name, train_sec)
    
    return excel_path, txt_path
def save_pointcloud_with_metadata(xyz, labels, output_path, metadata=None, color_map=None):
    """
    Salva a nuvem de pontos 3D com cores discretas para cada cluster e grava os metadados.

    Pass `color_map` (dict: label -> RGB) to reuse the exact same palette as
    the 2D render (render_clusters), so the same cluster gets the same color
    in both the point cloud and the images.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Fallback se Open3D não estiver instalado no ambiente de salvamento
    if not OPEN3D_AVAILABLE:
        npz_path = output_path.replace('.ply', '.npz')
        print(f"⚠️ Open3D não disponível. Salvando nuvem em NumPy: {npz_path}")
        np.savez(npz_path, points=xyz, labels=labels, metadata=metadata)
        return

    # Mapeamento discreto de cores para os clusters. IMPORTANTE: NÃO usar
    # `cmap(label % 20)` — com mais de 20 clusters (comum aqui), dois labels
    # que diferem por exatamente 20 recebem a MESMA cor. Usamos em vez disso
    # a mesma paleta maximamente distinta (CIELAB farthest-point) do render 2D.
    unique_labels = np.unique(labels)
    colors = np.zeros((len(labels), 3), dtype=np.float32)

    if color_map is None:
        valid_labels = [l for l in unique_labels if l != -1]
        palette = generate_distinct_colors(len(valid_labels))
        color_map = {label: palette[i] for i, label in enumerate(valid_labels)}
        color_map[-1] = np.array([0.12, 0.12, 0.12], dtype=np.float32)

    for label in unique_labels:
        mask = (labels == label)
        colors[mask] = color_map.get(label, np.array([0.12, 0.12, 0.12], dtype=np.float32))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Grava o arquivo .ply com posições e cores
    o3d.io.write_point_cloud(output_path, pcd)

    # Salva os metadados em JSON para auditoria do experimento
    if metadata:
        meta_path = output_path.replace('.ply', '_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    print(f"☁️ Nuvem de pontos salva em: {output_path}")

# ==========================================================
# CAIXA DE RECORTE (CLIPPING BOX) POR CENA
# ==========================================================
# Cada cena do LERF-OVS tem a sua própria caixa, tirada do "Edit clipping box"
# do CloudCompare, para limitar as gaussianas à região que o ground-truth
# cobre. Antes esses números eram o default de --bbox_center/--bbox_width e
# valiam para TODAS as cenas, então rodar outro dataset recortava a região
# errada silenciosamente. Agora a caixa é escolhida pelo nome da cena.
#
# `rotation` é a matriz de Orientação do CloudCompare, com as COLUNAS sendo os
# eixos da caixa expressos em coordenadas de mundo. Quando ela existe, a caixa
# é ORIENTADA (OBB) e o teste de pertencimento roda no referencial dela; quando
# é None, a caixa é alinhada aos eixos (AABB) e o teste é o de sempre. Sem esse
# tratamento, uma caixa girada testada como AABB recorta um volume diferente do
# que aparece na tela do CloudCompare.
BBOX_POR_CENA = {
    "teatime": {
        "center": [0.87771511, 1.94553256, 0.34934092],
        "width": [7.83855629, 6.98039675, 7.24066257],
        "rotation": None,
    },
    "figurines": {
        "center": [-0.48581076, 3.30481219, 1.14021492],
        "width": [7.46123409, 13.85611153, 10.74768734],
        "rotation": [
            [0.95109922, 0.05829292, 0.30333825],
            [0.01158329, 0.97460973, -0.22360983],
            [-0.30867133, 0.21618825, 0.92627533],
        ],
    },
}


def detectar_cena(args):
    """
    Descobre o nome da cena a partir de --scene, do caminho de -m
    (ex.: output/teatime -> "teatime") ou, em último caso, de --masks_path
    (.../teatime/Annotations -> "teatime"). Devolve o nome em minúsculas ou
    None se nada casar com uma cena conhecida.
    """
    if getattr(args, "scene", None):
        return str(args.scene).strip().lower()

    candidatos = []
    mp = getattr(args, "model_path", "") or ""
    if mp:
        candidatos.append(os.path.basename(os.path.normpath(mp)))
    masks = getattr(args, "masks_path", "") or ""
    # .../<cena>/Annotations -> pega o componente ANTES do diretório de máscaras
    candidatos.extend(p for p in os.path.normpath(masks).split(os.sep) if p)

    for c in candidatos:
        if c.lower() in BBOX_POR_CENA:
            return c.lower()
    return None


def resolver_bbox(args):
    """
    Decide qual caixa usar, nesta ordem de precedência:
      1. o que o usuário passou explicitamente em --bbox_center/--bbox_width
         (e opcionalmente --bbox_rotation);
      2. a caixa da cena detectada, se ela estiver em BBOX_POR_CENA;
      3. nada — o recorte é desligado, com aviso, em vez de aplicar a caixa de
         outra cena por engano.

    Devolve (center, half_width, rotation, descricao), com `center` None quando
    não há caixa aplicável.
    """
    cena = detectar_cena(args)

    if args.bbox_center is not None and args.bbox_width is not None:
        center = np.array(args.bbox_center, dtype=np.float64)
        width = np.array(args.bbox_width, dtype=np.float64)
        rot = (np.array(args.bbox_rotation, dtype=np.float64).reshape(3, 3)
               if args.bbox_rotation else None)
        desc = "valores passados na linha de comando"
    elif cena is not None:
        box = BBOX_POR_CENA[cena]
        center = np.array(box["center"], dtype=np.float64)
        width = np.array(box["width"], dtype=np.float64)
        rot = np.array(box["rotation"], dtype=np.float64) if box["rotation"] else None
        desc = f"preset da cena '{cena}'"
    else:
        print("\n⚠️  --use_bbox está ligado, mas não foi possível identificar a cena "
              "(nem por --scene, nem por -m, nem por --masks_path) e nenhuma caixa foi "
              "passada em --bbox_center/--bbox_width.")
        print(f"    Cenas com caixa cadastrada: {', '.join(sorted(BBOX_POR_CENA))}.")
        print("    O recorte por caixa foi DESLIGADO para esta rodada — aplicar a caixa "
              "de outra cena recortaria a região errada em silêncio.")
        return None, None, None, "nenhuma"

    if args.bbox_center is not None and args.bbox_width is None:
        raise ValueError("--bbox_center foi passado sem --bbox_width; passe os dois ou nenhum.")

    return center, width / 2.0, rot, desc


def pontos_dentro_da_bbox(xyz, center, half_width, rotation=None):
    """
    Máscara booleana das gaussianas dentro da caixa.

    Sem `rotation`, é o teste AABB de sempre. Com `rotation` (colunas = eixos da
    caixa em coordenadas de mundo), os pontos são primeiro levados para o
    referencial da caixa — `(p - c) @ R` é exatamente `R.T @ (p - c)` — e só
    então comparados com a meia-largura. É por isso que a caixa girada precisa
    da matriz: testar |p - c| <= half direto no mundo mediria a caixa alinhada
    que CIRCUNSCREVE a girada, um volume maior e com outro formato.
    """
    delta = xyz - center
    if rotation is not None:
        delta = delta @ rotation
    return np.all(np.abs(delta) <= half_width, axis=1)


def main():
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    # ==========================================================
    # DEFAULTS DA CONFIGURAÇÃO QUE FUNCIONOU (cena teatime, GAT)
    # ==========================================================
    # Os valores abaixo eram passados na linha de comando a cada rodada; viraram
    # default nesta versão só para encurtar o comando. Todos continuam sendo
    # argumentos normais, então qualquer um pode ser sobrescrito na chamada.
    # Comando que originou estes defaults:
    #   python "train_gnn copy 19.py" -m output/teatime \
    #     --masks_path .../teatime/Annotations
    parser.add_argument("--iteration", default=30000, type=int,
                        help="Iteração do checkpoint do 3DGS a carregar. Default 30000 (o "
                             "checkpoint final usado nos experimentos); -1 carrega a última "
                             "iteração disponível, que era o default das versões anteriores.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Seed do torch/numpy/random para inicialização do modelo GNN. "
                             "Necessário pra comparar configurações de HDBSCAN de forma justa "
                             "entre execuções — sem isso, a inicialização aleatória dos pesos "
                             "muda o embedding a cada rodada, e a diferença de métrica observada "
                             "deixa de ser atribuível só ao parâmetro que mudou.")
    parser.add_argument("--masks_path", required=True)
    # ATENÇÃO: aceito por compatibilidade com os comandos existentes, mas NÃO
    # é lido por ninguém. O único construtor de rótulos em uso é
    # build_labels_by_majority_vote(scene, xyz, masks_path, min_votes), que lê
    # apenas os PNGs de --masks_path. O consumidor deste JSON era
    # build_labels_with_deva_json_v2_black(), que nunca chegou a ser chamada e
    # foi removida na limpeza.
    parser.add_argument("--deva_json", default=None,
                        help="IGNORADO. Mantido só para não quebrar comandos antigos: os "
                             "rótulos vêm exclusivamente dos PNGs de --masks_path.")
    parser.add_argument("--min_votes", default=5, type=int,
                        help="Mínimo de votos consistentes para um ID de segmento DEVA ser "
                             "aceito como rótulo de uma Gaussiana.")
    # --merge_fragmented_tracks / --merge_jaccard_threshold foram removidas:
    # a fusão de tracks fragmentadas do DEVA por Jaccard só existia dentro de
    # build_labels_with_deva_json_v2_black(), que nunca era chamada. As flags
    # existiam e tinham default True, dando a impressão de que a fusão estava
    # ativa — ela nunca esteve.
    parser.add_argument("--output", default="exp_res")
    parser.add_argument("--visualize_3d", action="store_true", default=False,
                        help="Abre uma janela interativa do Open3D com a nuvem de pontos ao "
                             "final do experimento (bloqueia até a janela ser fechada). "
                             "DESLIGADO por padrão nesta versão, porque ele trava o fim de toda "
                             "rodada em lote esperando a janela ser fechada; passe "
                             "--visualize_3d para abrir a janela.")
    parser.add_argument("--no_visualize_3d", dest="visualize_3d", action="store_false")
    parser.add_argument("--render_images", action="store_true", default=True)

    parser.add_argument("--model_type_choice", default="gat", choices=["gat", "gcn"])
    parser.add_argument("--gcn_depth", default="v2", choices=["v2", "deep"], 
                        help="v2: 2 camadas GCN | deep: 4 camadas com skip connections")
    parser.add_argument("--gat_heads", default=8, type=int,
                        help="Cabeças de atenção da primeira camada do GAT. Default 8 (era 4 "
                             "nas versões anteriores).")

    parser.add_argument("--loss_temperature", default=0.05, type=float,
                        help="Temperatura do InfoNCE (ColorAwareContrastiveLossV2). Valores "
                             "menores deixam o softmax mais 'afiado', punindo mais forte qualquer "
                             "similaridade residual com negativos difíceis; valores maiores "
                             "suavizam essa distinção, relaxando a exigência de separação entre "
                             "instâncias.")
    parser.add_argument("--loss_pos_margin", default=0.2, type=float,
                        help="Margem positiva da loss contrastiva: só penaliza um par do mesmo "
                             "rótulo se sim < 1 - margem. Margem maior tolera mais dispersão "
                             "intra-instância dos embeddings antes de penalizar.")
    parser.add_argument("--loss_neg_margin", default=0.1, type=float,
                        help="Margem negativa da loss contrastiva: só penaliza um par de rótulos "
                             "diferentes se sim > margem. Margem maior tolera mais proximidade "
                             "entre embeddings de instâncias distintas antes de penalizar.")
    parser.add_argument("--loss_color_weight", default=0.0, type=float,
                        help="Peso do termo de consistência de cor na loss total: 0 ignora a cor "
                             "e treina só com os rótulos SAM/DEVA (margens + InfoNCE); 1 ignora os "
                             "rótulos e treina só com a similaridade de cor entre vizinhos do "
                             "grafo.")

    # ----- QUATRO CHAVES OPCIONAIS PARA O PCA COM CORES BORRADAS ENTRE OBJETOS -----
    # Diagnóstico discutido com o Gemini em 2026-09-08: PCA do embedding mostra
    # cores muito parecidas entre objetos distintos apesar dos rótulos DEVA
    # serem bem separados. Ver docstring de ColorAwareContrastiveLossV2 para o
    # mecanismo de cada chave. Todas desligadas/neutras por padrão — sem
    # nenhuma delas o comportamento é idêntico ao de antes, bit-a-bit.
    parser.add_argument("--loss_color_exclude_diff_label", action="store_true", default=False,
                        help="Exclui arestas diff-label (rótulos DEVA diferentes nas duas "
                             "pontas) do termo de consistência de cor. Sem isso, esse termo "
                             "empurra sim->1 mesmo em pares que a margem negativa está "
                             "tentando afastar, quando a cor coincide entre instâncias "
                             "diferentes (ex.: brinquedo sobre a mesa) — as duas loss brigam "
                             "e a fronteira sai borrada no PCA. Só tem efeito com "
                             "--loss_color_weight > 0. Desligado por padrão (comportamento "
                             "anterior: cor pondera TODAS as arestas).")
    parser.add_argument("--loss_detach_centroids", action="store_true", default=True,
                        help="Desconecta os centróides de rótulo do grafo de autograd antes "
                             "do produto escalar do InfoNCE. Sem isso, o centróide de uma "
                             "gaussiana inclui ELA MESMA na média, e o gradiente pode "
                             "'trapacear' puxando o centróide até o ponto em vez de organizar "
                             "o embedding de fato. Desligado por padrão (comportamento "
                             "anterior).")
    parser.add_argument("--loss_nce_class_balanced", action="store_true", default=True,
                        help="Faz a média da cross-entropy do InfoNCE POR CLASSE em vez de "
                             "por nó/gaussiana. Sem isso, um rótulo grande (ex.: a mesa, com "
                             "dezenas de milhares de gaussianas) domina o gradiente do "
                             "InfoNCE inteiro, e rótulos pequenos (objetos em cima dela) "
                             "recebem sinal de separação proporcionalmente fraco — assinatura "
                             "do 'vazamento' de cor da mesa para os objetos observado no PCA. "
                             "Desligado por padrão (comportamento anterior).")
    # Pesos por termo da loss. 1.0 em todos reproduz o comportamento anterior
    # (os três termos entravam na soma sem coeficiente nenhum). Existem para
    # permitir varrer o equilíbrio entre eles medindo mIoU/PQ, em vez de
    # assumir que 1:1:1 é o ponto certo.
    parser.add_argument("--loss_pos_weight", default=1.0, type=float,
                        help="Peso do termo de margem POSITIVA (aproxima vizinhos do mesmo "
                             "rótulo DEVA). 1.0 = comportamento anterior.")
    parser.add_argument("--loss_neg_weight", default=1.0, type=float,
                        help="Peso do termo de margem NEGATIVA (afasta vizinhos de rótulos "
                             "diferentes). 1.0 = comportamento anterior.")
    parser.add_argument("--loss_nce_weight", default=1.0, type=float,
                        help="Peso do InfoNCE contra centróides de rótulo. 1.0 = comportamento "
                             "anterior.")
    parser.add_argument("--loss_centroid_repulsion_weight", default=0.3, type=float,
                        help="Peso de um termo extra que penaliza similaridade de cosseno > "
                             "--loss_neg_margin ENTRE os centróides de rótulos diferentes, "
                             "além da repulsão ponto-a-centróide que o InfoNCE já faz. "
                             "Espalha os centróides pela hiperesfera antes mesmo de qualquer "
                             "ponto ser classificado contra eles. 0.0 desliga (comportamento "
                             "anterior).")

    # ----- LIGAÇÃO DAS ARESTAS POR COR PERCEPTUAL (CIELAB + ΔE) -----
    # O corte de cor do grafo é feito em CIELAB e não em RGB/SH: distância
    # euclidiana em RGB não é perceptual (o mesmo salto numérico é invisível
    # nos claros e gritante nos escuros/saturados), então um percentil de
    # distância RGB corta arestas em lugares que não correspondem a bordas
    # reais de objeto. Em ΔE o limiar tem significado fixo e auditável:
    #   ΔE00 ≈ 1.0  -> diferença no limite do perceptível (1 JND)
    #   ΔE00 ≈ 2.3  -> "just noticeable difference" clássica
    #   ΔE00 ≈ 5    -> diferença perceptível num relance
    #   ΔE00 ≈ 10   -> cores claramente distintas para um observador
    # Como dentro de UM objeto a cor varia por sombreamento/especular, o
    # padrão fica acima da JND (12.0): tolera variação de iluminação da mesma
    # superfície, mas ainda corta a transição para um objeto de cor diferente.
    parser.add_argument("--edge_delta_e_threshold", default=10.0, type=float,
                        help="Limiar de ΔE (CIELAB) para CRIAR a aresta entre dois vizinhos "
                             "espaciais. Referências: 1≈JND, 2.3=JND clássica, 5=perceptível "
                             "num relance, 10=cores claramente distintas. Acima de ~15 o corte "
                             "de cor praticamente deixa de podar; use <=5 para grafos bem "
                             "conservadores (só une o que o olho lê como a MESMA cor).")
    parser.add_argument("--edge_delta_e_sigma", default=6.0, type=float,
                        help="Sigma (em unidades de ΔE) da gaussiana que converte a diferença "
                             "de cor em PESO da aresta. Com o padrão 6.0, ΔE=2.3 (JND) pesa "
                             "~0.93, ΔE=6 pesa ~0.61 e ΔE=12 (limiar) pesa ~0.14.")
    parser.add_argument("--edge_delta_e_metric", default="ciede2000",
                        choices=["ciede2000", "cie76"],
                        help="Métrica de diferença de cor em CIELAB: 'ciede2000' (padrão CIE "
                             "mais fiel à percepção, corrige a não-uniformidade residual do Lab "
                             "em azuis e baixa saturação) ou 'cie76' (euclidiana no Lab, mais "
                             "barata e suficiente para diferenças grandes).")
    parser.add_argument("--feature_lab_scale", default=0.0, type=float,
                        help="Escala ISOTRÓPICA aplicada ao Lab centrado nas features "
                             "(um único escalar para L, a e b, para não destruir a uniformidade "
                             "perceptual). 0 = auto: iguala a magnitude do bloco de cor à da "
                             "versão RGB antiga. Ex.: 0.1 faria 1 unidade de feature = 10 ΔE.")
    parser.add_argument("--loss_delta_e_sigma", default=12.0, type=float,
                        help="Sigma (em ΔE) do kernel de similaridade de cor DENTRO da loss. "
                             "Maior que --edge_delta_e_sigma de propósito: a loss compara também "
                             "pares distantes/negativos, não só vizinhos do grafo, então uma "
                             "banda mais larga evita saturar em 0 para qualquer par não-idêntico.")
    parser.add_argument("--edge_color_threshold_mode", default="perceptual",
                        choices=["perceptual", "percentile"],
                        help="'perceptual' usa --edge_delta_e_threshold (limiar absoluto, com "
                             "significado fixo entre cenas); 'percentile' reproduz o "
                             "comportamento antigo, cortando no percentil 75 das diferenças "
                             "observadas (limiar relativo, muda de cena para cena).")

    parser.add_argument("--opacity_threshold", default=0.05, type=float)
    parser.add_argument("--max_scale_threshold", default=10, type=float)
    parser.add_argument("--hdbscan_cluster_selection_method", default="eom",
                        choices=["eom", "leaf"],
                        help="Como o HDBSCAN escolhe os clusters na árvore condensada. 'eom' "
                             "(excess of mass) tende a escolher clusters maiores e mais "
                             "estáveis; 'leaf' seleciona os nós das pontas da árvore, "
                             "produzindo clusters mais miúdos e homogêneos.")
    parser.add_argument("--render_pca", action="store_true", default=True,
                        help="Salva em <output>/PCA uma imagem por camera com o EMBEDDING "
                             "reduzido a 3 dimensoes por PCA e mapeado em RGB. Mostra o que a "
                             "GNN aprendeu antes do HDBSCAN: embedding colapsado sai como uma "
                             "imagem de cor uniforme, embedding com estrutura sai com cada "
                             "objeto num tom proprio.")
    parser.add_argument("--no_render_pca", dest="render_pca", action="store_false",
                        help="Nao gerar o diretorio PCA (economiza um passe de render).")
    parser.add_argument("--diagnose_embedding", action="store_true", default=False,
                        help="Roda um diagnóstico do embedding depois do treino: separação "
                             "intra vs inter rótulo DEVA, taxa de falha da mineração hard e "
                             "estabilidade do nº de clusters do HDBSCAN em vários min_samples. "
                             "Só imprime — não altera nada do resultado. Custa ~1min.")
    parser.add_argument("--min_cluster_size", default=500, type=int)
    parser.add_argument("--hdbscan_min_samples", default=3, type=int,
                        help="min_samples do HDBSCAN. Valores maiores tornam o HDBSCAN mais "
                             "conservador para chamar uma região esparsa (ex: casca/borda de "
                             "um objeto) de cluster próprio, reduzindo fragmentos que depois "
                             "aparecem como uma cor separada na renderização. Também empurra "
                             "pontos ambíguos de fronteira entre dois objetos para ruído (-1) "
                             "em vez de forçar um lado errado, reduzindo espalhamento — o "
                             "reassign_noise_knn_vectorized só recupera esses pontos depois se "
                             "houver consenso forte dos vizinhos.")
    # Default 0.0 (e não 0.1): este parâmetro estava comentado na chamada do
    # HDBSCAN, então o que rodava de fato era o padrão do HDBSCAN, que é 0.0.
    # Manter 0.0 preserva o comportamento já observado nos testes.
    parser.add_argument("--hdbscan_cluster_selection_epsilon", default=0.0, type=float,
                        help="cluster_selection_epsilon do HDBSCAN: funde, na árvore condensada, "
                             "clusters cuja distância (no espaço de embedding normalizado, L2) "
                             "for menor que este valor. É o mecanismo nativo do HDBSCAN para "
                             "'muitos fragmentos pequenos e parecidos' — mais direto que subir "
                             "min_samples para esse fim específico. 0.0 desliga (comportamento "
                             "padrão do HDBSCAN, sem fusão por distância).")
    # Bounding box (AABB) para limitar a cena aos objetos do ground-truth.
    # Os valores default são os do "Edit clipping box" do CloudCompare e são
    # ESPECÍFICOS DA CENA — ao trocar de dataset, passe os valores da cena
    # correspondente ou desligue com --no_bbox.
    parser.add_argument("--use_bbox", action="store_true", default=True,
                        help="Restringe as Gaussianas a uma caixa alinhada aos eixos antes "
                             "do filtro dinâmico de opacidade/escala.")
    parser.add_argument("--no_bbox", dest="use_bbox", action="store_false",
                        help="Desliga o recorte por caixa e usa a cena inteira.")
    parser.add_argument("--scene", default=None, type=str,
                        help="Nome da cena, usado para escolher a caixa de recorte em "
                             "BBOX_POR_CENA. Se omitido, é deduzido do caminho de -m "
                             "(ex.: output/teatime -> 'teatime') e, em último caso, de "
                             "--masks_path. Cenas cadastradas: teatime, figurines.")
    parser.add_argument("--bbox_center", nargs=3, type=float, default=None,
                        help="Centro (x y z) da caixa, no mesmo sistema de coordenadas das "
                             "Gaussianas. Se omitido, usa a caixa da cena detectada; passar "
                             "este argumento (junto com --bbox_width) sobrescreve o preset.")
    parser.add_argument("--bbox_width", nargs=3, type=float, default=None,
                        help="Largura total (x y z) da caixa; o recorte usa metade disso para "
                             "cada lado do centro. Só é lido junto com --bbox_center.")
    parser.add_argument("--bbox_rotation", nargs=9, type=float, default=None,
                        help="Matriz de Orientação 3x3 da caixa, em ordem por LINHA (os 9 "
                             "números do campo 'Orientation' do CloudCompare, lidos da esquerda "
                             "para a direita e de cima para baixo). Só faz sentido junto com "
                             "--bbox_center/--bbox_width; sem ela a caixa é alinhada aos eixos.")
    parser.add_argument("--target_gaussians", default=200000, type=int,
                        help="Nº alvo de gaussianas após os filtros de opacidade/escala/label/"
                             "bbox e a remoção de floaters. Default 200000 (era 230000).")

    parser.add_argument("--render_full_pointcloud", dest="render_full_pointcloud",
                        action="store_true", default=False,
                        help="Propaga os labels para TODAS as gaussianas antes de renderizar "
                             "(via propagate_labels_to_full_set), em vez de renderizar apenas "
                             "o subconjunto filtrado. Desligado por padrão; passe esta flag para "
                             "ligar e comparar as duas condições.")
    parser.add_argument("--no_render_full_pointcloud", dest="render_full_pointcloud",
                        action="store_false")
    parser.add_argument("--propagate_max_distance", default=-1.0, type=float,
                        help="Raio máximo (mesma unidade de xyz) para a propagação de labels "
                             "ao conjunto completo. Gaussianas mais distantes do que isso do "
                             "seu vizinho rotulado mais próximo ficam sem label (-1) em vez de "
                             "herdar um rótulo espúrio. Valor <= 0 desliga o limite (comportamento "
                             "anterior).")
    parser.add_argument("--knn_reassign_k", default=25, type=int)
    parser.add_argument("--experiment_name", default="teatime", type=str,
                        help="Nome do experimento (subpasta de --output onde tudo é gravado). "
                             "Default 'teatime'. ATENÇÃO: como é fixo, duas rodadas seguidas sem "
                             "passar este argumento escrevem na MESMA pasta e a segunda "
                             "sobrescreve as imagens/nuvem da primeira (o .xlsx e o .txt levam "
                             "timestamp no nome e sobrevivem). Passe um nome diferente por "
                             "rodada ao comparar configurações, ou 'auto' para voltar ao nome "
                             "com timestamp das versões anteriores.")
    # Ablações para testar se a etapa de aprendizado contribui de fato, ou se
    # o resultado vem das features de entrada + HDBSCAN. Os defaults preservam
    # o comportamento anterior (200 épocas, clusterizando o embedding).
    parser.add_argument("--train_epochs", default=800, type=int,
                        help="Épocas de treino da GNN. 0 = não treina: o embedding sai da rede "
                             "recém-inicializada (pesos aleatórios), isolando o que a loss "
                             "efetivamente aprendeu do que já vinha da arquitetura e das "
                             "features.")
    parser.add_argument("--cluster_on_raw_features", action="store_true", default=False,
                        help="Clusteriza as FEATURES DE ENTRADA (xyz normalizado + Lab) em vez "
                             "do embedding da GNN, descartando a rede inteira. Se empatar com "
                             "o pipeline treinado, a GNN não está agregando nada.")
    parser.add_argument("--render_frames", default=None, type=str,
                        help="Lista de nomes de frame separados por vírgula (ex.: "
                             "'frame_00041,frame_00105') a renderizar, em vez de todas as "
                             "~300 câmeras de treino. Serve para sweeps de hiperparâmetro "
                             "avaliados por mIoU/PQ: o ground-truth do LERF-OVS só cobre 4-7 "
                             "frames por cena, então renderizar o resto é tempo gasto sem "
                             "gerar informação. Sem esta flag, renderiza tudo (comportamento "
                             "anterior).")

    # ==========================================================
    # ESTRUTURA DO GRAFO: nº de candidatos e critério de distância
    # ==========================================================
    parser.add_argument("--graph_knn", default=7, type=int,
                        help="Valor passado ao NearestNeighbors, que CONTA O PRÓPRIO PONTO: "
                             "--graph_knn 7 (default desta versão) dá 6 vizinhos candidatos por "
                             "nó; --graph_knn 20 dá 19, que era o valor fixo das versões "
                             "anteriores. Os limiares continuam "
                             "decidindo quem vira aresta — subir isto só dá mais candidatos "
                             "para eles escolherem, e é a forma legítima de reduzir nó isolado "
                             "(ao contrário do grau mínimo forçado, removido nesta versão). "
                             "Custo: o nº de arestas cresce quase proporcionalmente, e com ele "
                             "a VRAM do GAT.")
    parser.add_argument("--edge_distance_mode", default="local_scaling",
                        choices=["global", "local_scaling", "gaussian_scale"],
                        help="Critério de distância para criar a aresta. DEFAULT desta versão é "
                             "'local_scaling'. 'global' (comportamento das versões anteriores): "
                             "um único limiar mean+std para a cena "
                             "inteira. 'local_scaling': limiar por par, c*sqrt(sigma_i*sigma_j), "
                             "onde sigma_i é a distância de i ao seu k-ésimo vizinho "
                             "(--edge_local_scaling_k) — livre de escala, região densa ganha "
                             "limiar apertado e região esparsa um proporcionalmente maior. "
                             "'gaussian_scale': limiar c*(s_i+s_j) com s = maior eixo da própria "
                             "gaussiana, ou seja, 'as duas gaussianas se tocam' — a única das "
                             "três com significado físico e sem constante empírica da cena.")
    parser.add_argument("--edge_local_scale_c", default=1.0, type=float,
                        help="Multiplicador c dos modos locais de --edge_distance_mode. Em "
                             "'gaussian_scale', c=1.0 significa aceitar a aresta quando a "
                             "distância entre os centros for menor que a soma dos raios.")
    parser.add_argument("--edge_local_scaling_k", default=7, type=int,
                        help="Qual vizinho define sigma_i no modo 'local_scaling' (7 é o valor "
                             "clássico de Zelnik-Manor & Perona). É truncado ao nº de vizinhos "
                             "disponíveis, então com --graph_knn 7 (6 candidatos) vale 6.")

    # ==========================================================
    # DIAGNÓSTICO DO GRAFO INICIAL (não altera nada do pipeline)
    # ==========================================================
    parser.add_argument("--graph_debug", action="store_true", default=False,
                        help="Depois de montar o grafo (e antes de treinar), escreve em "
                             "{output}/graph_debug um retrato das conexões iniciais: grau dos "
                             "nós, nós isolados, quanto do kNN cada limiar podou, quantas "
                             "arestas ligam labels DIFERENTES (vazamento entre objetos), em "
                             "quantos pedaços desconectados cada objeto ficou, componentes "
                             "conexas, histogramas, projeções 2D das arestas e PLYs 3D das "
                             "arestas/nós para abrir no MeshLab ou CloudCompare. Desligado por "
                             "padrão; ligar NÃO muda o grafo nem o resultado do treino.")
    parser.add_argument("--graph_debug_only", action="store_true", default=False,
                        help="Monta o grafo, gera o diagnóstico e ENCERRA, sem treinar nem "
                             "renderizar. Serve para calibrar --edge_delta_e_threshold e o "
                             "grau mínimo em segundos, em vez de esperar o pipeline inteiro.")
    parser.add_argument("--graph_debug_show", action="store_true", default=False,
                        help="Além dos arquivos, abre uma janela do Open3D com a nuvem e as "
                             "arestas (verde = mesmo label, vermelho = labels diferentes). "
                             "Precisa de display e do Open3D instalado.")
    parser.add_argument("--graph_debug_max_edges", default=150000, type=int,
                        help="Máximo de arestas (amostradas aleatoriamente com --seed) "
                             "exportadas para os PLYs de linhas. O grafo inteiro costuma ter "
                             "milhões de arestas e trava qualquer viewer.")
    parser.add_argument("--graph_debug_max_edges_plot", default=30000, type=int,
                        help="Máximo de arestas desenhadas nas projeções 2D (graph_edges_proj.png).")

    args = get_combined_args(parser)

    # get_combined_args() funde o cfg_args do modelo com a linha de comando
    # usando `if v != None: merged_dict[k] = v` (ver arguments/__init__.py).
    # O efeito colateral é que TODO argumento deixado no default None é
    # DESCARTADO do Namespace — não vira None, deixa de existir —, e o primeiro
    # acesso a ele estoura com AttributeError em vez de simplesmente ler None.
    # Isso atinge qualquer argumento opcional cujo default seja None
    # (--bbox_center, --bbox_width, --bbox_rotation, --scene, --render_frames,
    # --deva_json). Repor os defaults do parser aqui resolve a classe inteira do
    # problema, em vez de espalhar getattr(args, ..., None) por cada uso.
    for _acao in parser._actions:
        if _acao.dest != "help" and not hasattr(args, _acao.dest):
            setattr(args, _acao.dest, _acao.default)

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ==========================================================
    # CONFIGURAÇÃO DA ESTRUTURA DE DIRETÓRIOS
    # ==========================================================
    # `--experiment_name auto` (ou None) devolve o nome com timestamp das versões
    # anteriores, em que cada rodada ganha a sua própria pasta. O default fixo
    # ('teatime') encurta o comando, mas faz rodadas consecutivas gravarem no
    # mesmo lugar — 'auto' é a saída para quando estiver comparando configurações.
    if args.experiment_name is None or str(args.experiment_name).lower() == "auto":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = f"gcn_{args.gcn_depth}" if args.model_type_choice == "gcn" else args.model_type_choice
        exp_name = f"{model_suffix}_{timestamp}"
    else:
        exp_name = args.experiment_name
    
    base_dir = args.output
    exp_dir = os.path.join(base_dir, exp_name)
    model_dir = os.path.join(exp_dir, args.model_type_choice)
    images_dir = os.path.join(model_dir, "images")
    pointclouds_dir = os.path.join(model_dir, "pointclouds")
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(pointclouds_dir, exist_ok=True)
    
    args.output = model_dir
    
    print(f"\n{'='*80}")
    print(f"🚀 EXPERIMENTO: {exp_name}")
    print(f"📁 Diretório: {exp_dir}")
    if args.model_type_choice == "gcn":
        print(f"📁 Modelo: GCN-{args.gcn_depth.upper()} ({'2 camadas' if args.gcn_depth == 'v2' else '4 camadas com skip connections'})")
    else:
        print(f"📁 Modelo: {args.model_type_choice.upper()}")
    print(f"{'='*80}\n")
    
    safe_state(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Dispositivo: {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # ==========================================================
    # LOAD GAUSSIANS
    # ==========================================================
    gaussians = GaussianModel(model_params.extract(args).sh_degree)
    scene = Scene(model_params.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)

    xyz_full_original = gaussians._xyz.detach().cpu().numpy()
    
    total_gaussians_raw = len(xyz_full_original)
    
    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().squeeze(1).numpy()

    # FILTRAGEM GEOMÉTRICA E DE ROTULAÇÃO
    opacity = torch.sigmoid(gaussians._opacity).detach().cpu().numpy().squeeze()
    scaling = torch.exp(gaussians._scaling).detach().cpu().numpy()
    max_scaling = np.max(scaling, axis=1)

    labels_deva = build_labels_by_majority_vote(
        scene,
        xyz,
        args.masks_path,
        min_votes=args.min_votes,
    )

    mask_labels_validos = (labels_deva != -1)

    if args.use_bbox:
        box_center, half_width, box_rot, box_desc = resolver_bbox(args)
    else:
        box_center = None

    if box_center is not None:
        mask_box = pontos_dentro_da_bbox(xyz, box_center, half_width, box_rot)

        base_mask = mask_labels_validos & mask_box

        tipo = "OBB, orientada" if box_rot is not None else "AABB, alinhada aos eixos"
        print(f"\n📦 Bounding box aplicada ({tipo}) — {box_desc}:")
        print(f"   centro={box_center} | largura={half_width * 2.0}")
        if box_rot is not None:
            print(f"   orientação (colunas = eixos da caixa no mundo):")
            for linha in box_rot:
                print("     " + "  ".join(f"{v: .8f}" for v in linha))
        print(f"   Gaussianas dentro da caixa: {int(mask_box.sum()):,}/{len(xyz):,} "
              f"({100 * mask_box.sum() / len(xyz):.1f}%)")
        print(f"   Com label DEVA válido E dentro da caixa: {int(base_mask.sum()):,} "
              f"(sem a caixa seriam {int(mask_labels_validos.sum()):,})")
        if base_mask.sum() == 0:
            print("   ⚠️  A caixa não contém NENHUMA gaussiana rotulada — "
                  "verifique se os valores correspondem a esta cena.")
    else:
        base_mask = mask_labels_validos


    target_count = args.target_gaussians
    current_target = target_count
    mask_filter = base_mask.copy()
    final_count = 0

    for iteration in range(4):
        opacity_valid = opacity[base_mask]
        max_scaling_valid = max_scaling[base_mask]

        dyn_op_thresh, dyn_sc_thresh, _, mask_sel = compute_dynamic_filters(
            opacity_valid,
            max_scaling_valid,
            target_count=min(current_target, int(base_mask.sum()))
        )

        args.opacity_threshold = dyn_op_thresh
        args.max_scale_threshold = dyn_sc_thresh

        # A seleção vem da máscara devolvida pela função, e não de reaplicar os
        # limiares: com empates na opacidade (sigmoide saturada em 1.0) nenhum
        # par de limiares consegue expressar a escolha correta — reaplicar
        # `opacity > op_thresh` descartaria justamente o bloco empatado.
        # `mask_sel` é relativa ao subconjunto `base_mask`, então é espalhada
        # de volta para os índices originais.
        candidate_mask = np.zeros_like(base_mask)
        candidate_mask[np.where(base_mask)[0][mask_sel]] = True

        # Opacidade/escala/label/bbox não veem DENSIDADE: um floater isolado no
        # espaço pode passar em todos esses filtros. Remove quem sobra
        # sozinho.
        mask_filter = remove_spatial_outliers(xyz, candidate_mask, k=16, std_ratio=2.0)
        final_count = int(mask_filter.sum())

        print(f"   🔁 Iteração {iteration + 1}: target pedido={target_count:,} | "
              f"antes da remoção de outliers={int(candidate_mask.sum()):,} | "
              f"REAL após {'bbox+' if args.use_bbox else ''}outliers={final_count:,}")

        if final_count == 0:
            break

        error_ratio = abs(final_count - target_count) / target_count
        if error_ratio <= 0.02:
            break

        current_target = int(current_target * target_count / max(final_count, 1))
        current_target = min(current_target, int(base_mask.sum()))

    labels = labels_deva[mask_filter]
    xyz = xyz[mask_filter]
    rgb = rgb[mask_filter]
    # Maior eixo de cada gaussiana que sobreviveu ao filtro, na mesma unidade de
    # xyz (o _scaling é guardado em log). É o "raio" usado pelo modo
    # --edge_distance_mode gaussian_scale.
    node_scale = max_scaling[mask_filter]
    pipe = pipeline_params.extract(args)
    background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
    render_clusters(labels, "debug_labels_2d", gaussians, scene, pipe, background, args, device, mask_filter)

    total_gaussians_filtered = len(xyz)

    print(f"📦 Total de Gaussianas na cena (Original): {total_gaussians_raw:,}")
    print(f"🎯 Total de Gaussianas rotuladas (DEVA != -1): {np.sum(mask_labels_validos):,}")
    print(f"✅ Total de Gaussianas final para treinamento (Filtradas): {total_gaussians_filtered:,}")

    xyz_centered = xyz - np.mean(xyz, axis=0)
    spatial_std = np.std(xyz)
    xyz_norm = xyz_centered / (spatial_std + 1e-8)

    # ----------------------------------------------------------
    # COR EM ESPAÇO PERCEPTUAL (CIELAB), USADA EM TODA A PIPELINE
    # ----------------------------------------------------------
    # `rgb` guarda os coeficientes DC de esféricos harmônicos, NÃO cor: a cor
    # sRGB de fato é feat_dc * C0 + 0.5 (mesma conversão de
    # da montagem de rótulos). Converter uma única vez aqui deixa
    # o mesmo `lab` disponível para os três lugares onde a cor entra no
    # treinamento: features dos nós, corte/peso das arestas e termo de cor da
    # loss — todos passam a falar a mesma unidade (ΔE).
    C0_SH = 0.28209479177387814
    rgb_srgb = np.clip(rgb.astype(np.float64) * C0_SH + 0.5, 0.0, 1.0)
    lab = rgb2lab(rgb_srgb.reshape(1, -1, 3)).reshape(-1, 3)

    # Normalização das features de cor. Padronizar Lab por canal (StandardScaler)
    # DESTRUIRIA a uniformidade perceptual: cada eixo seria esticado pela sua
    # própria variância, e a distância no bloco de cor deixaria de ser ΔE. Por
    # isso a escala é ISOTRÓPICA — um único escalar para L, a e b —, o que
    # preserva as razões de ΔE e mantém a distância euclidiana nas features
    # proporcional à diferença percebida.
    lab_centered = lab - lab.mean(axis=0, keepdims=True)
    lab_global_std = float(lab_centered.std())
    if args.feature_lab_scale > 0:
        lab_scale = float(args.feature_lab_scale)
        scale_desc = "escala fixa do usuário"
    else:
        # "auto": iguala a magnitude do bloco de cor à da versão anterior
        # (StandardScaler * 3.0 dava desvio 3.0 por canal), sem abrir mão da
        # isotropia. Assim trocar RGB->Lab não muda o PESO relativo entre
        # geometria e cor nas features, só o espaço em que a cor é medida.
        lab_scale = 3.0 / (lab_global_std + 1e-8)
        scale_desc = "auto (mesma magnitude do bloco de cor da versão RGB)"
    lab_norm = lab_centered * lab_scale

    color_feat = lab_norm
    print(f"\n🎨 Features de cor: CIELAB isotrópico | escala={lab_scale:.4f} "
          f"({scale_desc}) | std global do Lab={lab_global_std:.2f}")
    print(f"   1 unidade de feature de cor = {1.0 / (lab_scale + 1e-8):.2f} ΔE(76)")

    # ==========================================================
    # CONSTRUÇÃO DO GRAFO
    # ==========================================================
    t_graph_start = time.time()
    
    x_input = np.concatenate([xyz_norm, color_feat], axis=1)
    x = torch.tensor(x_input, dtype=torch.float)

    print("\n🔗 Construindo grafo geométrico...")
    N_NEIGHBORS = args.graph_knn
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="auto").fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)

    neighbor_idx = indices[:, 1:]
    neighbor_dist = distances[:, 1:]

    sigma = np.median(neighbor_dist)
    print(f"  Vizinhos candidatos por nó: {neighbor_idx.shape[1]} (--graph_knn={N_NEIGHBORS})")
    print(f"  Sigma geométrico: {sigma:.4f}")

    # ----------------------------------------------------------
    # SIMILARIDADE DE COR EM ESPAÇO PERCEPTUAL (CIELAB + ΔE)
    # ----------------------------------------------------------
    # `lab` já foi calculado acima (SH DC -> sRGB -> CIELAB). A versão anterior
    # media distância euclidiana direto nos coeficientes SH, o que é duas vezes
    # não-perceptual: não é sequer sRGB, e RGB por si só já não é uniforme — o
    # mesmo Δ numérico é imperceptível em regiões claras e óbvio em
    # escuras/saturadas. Por isso o limiar antigo precisava ser um percentil
    # (relativo, sem significado físico e diferente a cada cena).
    #
    # Em CIELAB a distância aproxima diferença percebida, e a diferença é
    # medida em ΔE — com limiar absoluto e interpretável (ver
    # --edge_delta_e_threshold).
    lab_i = np.broadcast_to(lab[:, None, :], (lab.shape[0], neighbor_idx.shape[1], 3))
    lab_j = lab[neighbor_idx]

    if args.edge_delta_e_metric == "ciede2000":
        # CIEDE2000 corrige a não-uniformidade residual do Lab (azuis, baixa
        # saturação, dependência de tom/croma). Vetorizado: ~N*(k-1) pares.
        color_dist = deltaE_ciede2000(lab_i, lab_j)
    else:
        # CIE76: euclidiana no Lab. Já é perceptual o bastante para diferenças
        # grandes e custa uma fração do CIEDE2000.
        color_dist = np.linalg.norm(lab_i - lab_j, axis=-1)
    color_dist = np.ascontiguousarray(color_dist, dtype=np.float64)

    # O peso da aresta também passa a viver em ΔE: sigma em unidades
    # perceptuais, e não mais a mediana empírica de uma distância sem unidade.
    color_sigma = float(args.edge_delta_e_sigma)
    color_weight_all = np.exp(-(color_dist ** 2) / (2 * color_sigma ** 2 + 1e-8))
    # ----------------------------------------------------------
    # CRITÉRIO DE DISTÂNCIA: global (mean+std) ou local (por par)
    # ----------------------------------------------------------
    # O modo 'global' é uma estatística da nuvem inteira aplicada a um campo de
    # densidade que em 3DGS varia por ordens de grandeza: superfície texturizada
    # é densa, fundo e floater são esparsos. O mesmo limiar deixa passar os 6
    # vizinhos na região densa e não deixa passar nenhum na esparsa — daí nó
    # isolado e objeto estilhaçado no mesmo grafo. Os modos locais medem a
    # distância na régua da vizinhança de cada par, e por isso não têm constante
    # empírica dependente da cena.
    if args.edge_distance_mode == "global":
        spatial_weight_all = np.exp(-(neighbor_dist ** 2) / (2 * sigma ** 2 + 1e-8))
        distance_threshold = float(neighbor_dist.mean() + neighbor_dist.std())
        dist_ok = neighbor_dist <= distance_threshold
        dist_desc = "global (média + desvio)"

    elif args.edge_distance_mode == "local_scaling":
        # Zelnik-Manor & Perona: sigma_i = distância de i ao seu k-ésimo vizinho.
        # A afinidade exp(-d²/(sigma_i*sigma_j)) e o limiar c*sqrt(sigma_i*sigma_j)
        # ficam invariantes a uma mudança de escala da nuvem.
        k_ls = int(np.clip(args.edge_local_scaling_k, 1, neighbor_dist.shape[1]))
        sigma_local = np.maximum(neighbor_dist[:, k_ls - 1], 1e-12)
        sig_prod = sigma_local[:, None] * sigma_local[neighbor_idx]
        thr_pair = args.edge_local_scale_c * np.sqrt(sig_prod)
        spatial_weight_all = np.exp(-(neighbor_dist ** 2) / (sig_prod + 1e-12))
        dist_ok = neighbor_dist <= thr_pair
        distance_threshold = float(np.median(thr_pair))
        dist_desc = (f"local_scaling c={args.edge_local_scale_c:g} k={k_ls} "
                     f"(limiar por par; mediana mostrada)")
        print(f"  Sigma local (dist. ao {k_ls}º vizinho): mediana={np.median(sigma_local):.4f} "
              f"p5={np.percentile(sigma_local, 5):.4f} p95={np.percentile(sigma_local, 95):.4f} "
              f"— razão p95/p5 = {np.percentile(sigma_local, 95)/max(np.percentile(sigma_local, 5), 1e-12):.1f}x "
              f"(é essa variação de densidade que um limiar global ignora)")

    else:  # gaussian_scale
        # Distância medida em unidades do raio da própria gaussiana: com c=1 a
        # aresta existe quando os dois elipsoides se tocam. Usa o maior eixo,
        # que é o mesmo critério de escala já usado no filtro dinâmico.
        s_i = node_scale[:, None]
        s_j = node_scale[neighbor_idx]
        thr_pair = args.edge_local_scale_c * (s_i + s_j)
        s_pair = 0.5 * (s_i + s_j)
        spatial_weight_all = np.exp(-(neighbor_dist ** 2) / (2 * s_pair ** 2 + 1e-12))
        dist_ok = neighbor_dist <= thr_pair
        distance_threshold = float(np.median(thr_pair))
        dist_desc = (f"gaussian_scale c={args.edge_local_scale_c:g} "
                     f"(limiar = c*(s_i+s_j); mediana mostrada)")
        print(f"  Escala das gaussianas (maior eixo): mediana={np.median(node_scale):.4f} "
              f"p5={np.percentile(node_scale, 5):.4f} p95={np.percentile(node_scale, 95):.4f}")
    if args.edge_color_threshold_mode == "percentile":
        color_dist_threshold = float(np.percentile(color_dist, 75))
        threshold_desc = "percentil 75 (relativo à cena)"
    else:
        color_dist_threshold = float(args.edge_delta_e_threshold)
        threshold_desc = "limiar perceptual absoluto"

    de_p = np.percentile(color_dist, [25, 50, 75, 90])
    print(f"  Cor: CIELAB / ΔE {args.edge_delta_e_metric} | sigma_ΔE={color_sigma:.2f}")
    print(f"  ΔE entre vizinhos — p25={de_p[0]:.2f} | mediana={de_p[1]:.2f} | "
          f"p75={de_p[2]:.2f} | p90={de_p[3]:.2f}")
    print(f"  Distance threshold: {distance_threshold:.4f} ({dist_desc}) | "
          f"ΔE threshold: {color_dist_threshold:.2f} ({threshold_desc})")
    print(f"  Vizinhos aceitos pelo critério de distância: {100 * dist_ok.mean():.1f}%")

    frac_ok = float((color_dist <= color_dist_threshold).mean())
    print(f"  Vizinhos aceitos pelo critério de cor: {100 * frac_ok:.1f}%")
    if frac_ok > 0.98:
        print("  ⚠️ O limiar de ΔE quase não está podando arestas — considere "
              "baixar --edge_delta_e_threshold (ex.: 5.0) se quiser separar mais "
              "por cor.")
    elif frac_ok < 0.30:
        print("  ⚠️ O limiar de ΔE está podando a maior parte das arestas; muitos nós "
              "vão ficar com grau baixo ou isolados. Considere subir "
              "--edge_delta_e_threshold.")

    # ÚNICO critério de aresta: distância espacial E semelhança de cor (ΔE).
    # Não existe mais nenhum mecanismo que reponha aresta depois desta linha.
    #
    # A versão anterior tinha um MIN_DEGREE=2 que, para todo nó com menos de
    # dois vizinhos aprovados, ligava à força os dois mais próximos ignorando
    # os DOIS limiares. Medido na cena teatime, isso criava 70.057 arestas
    # (17% do total), com ΔE de até ~100 (contra o limiar de 12) e comprimento
    # de até 2,7 (contra ~0,23) — fios atravessando o vazio da sala, ligando
    # objetos sem nenhuma relação. Pior: no GAT o edge_weight não entra na
    # propagação, então essas arestas passavam mensagem com força total apesar
    # do peso ≈ 0, e o GATConv já cria self-loops por conta própria, de modo
    # que o colapso de embedding que o mecanismo pretendia evitar nem existia
    # nesse caminho. Sem o forçamento, o grafo pode ter nós isolados: no GCN
    # (add_self_loops=False) esses nós não recebem mensagem e o embedding
    # colapsa no bias, então o aviso abaixo importa nesse modo.
    edge_mask = dist_ok & (color_dist <= color_dist_threshold)

    src, k_idx = np.where(edge_mask)
    dst = neighbor_idx[src, k_idx]

    weight = 0.4 * spatial_weight_all[src, k_idx] + 1.2* color_weight_all[src, k_idx]

    edges_np = np.concatenate([
        np.stack([src, dst], axis=1),
        np.stack([dst, src], axis=1),
    ], axis=0)
    weights_np = np.concatenate([weight, weight], axis=0)

    edges_np, dedup_idx = np.unique(edges_np, axis=0, return_index=True)
    weights_np = weights_np[dedup_idx]

    edge_index = torch.tensor(edges_np.T, dtype=torch.long)
    edge_weights_t = torch.tensor(weights_np, dtype=torch.float, device=device)

    t_graph_end = time.time()
    graph_build_time = t_graph_end - t_graph_start

    print(f"  Arestas no grafo: {len(edges_np):,}")

    # Sem o grau mínimo forçado, nó isolado é um resultado possível — e no GCN
    # ele é silenciosamente destrutivo, então é medido e avisado sempre.
    _deg_final = np.bincount(edges_np[:, 0], minlength=len(xyz))
    _n_iso = int((_deg_final == 0).sum())
    _n_deg1 = int((_deg_final == 1).sum())
    print(f"  Grau: média={_deg_final.mean():.2f} mediana={np.median(_deg_final):.0f} "
          f"máx={_deg_final.max()} | nós isolados: {_n_iso:,} ({100.0*_n_iso/len(xyz):.2f}%) "
          f"| grau 1: {_n_deg1:,} ({100.0*_n_deg1/len(xyz):.2f}%)")
    if _n_iso > 0 and args.model_type_choice == "gcn":
        print(f"  ⚠️ {_n_iso:,} nós isolados no modo GCN (add_self_loops=False): eles não "
              f"recebem mensagem nenhuma e o embedding colapsa no bias. Suba "
              f"--edge_delta_e_threshold para religá-los por um critério real, em vez "
              f"de forçar arestas arbitrárias.")
    print(f"⏱️ Tempo de construção do grafo: {graph_build_time:.2f}s")

    # O diagnóstico roda FORA da medição de tempo do grafo, para não poluir a
    # métrica de graph_build_time que vai para o relatório do experimento.
    if args.graph_debug:
        debug_graph_connectivity(
            xyz=xyz,
            labels=labels,
            edges_np=edges_np,
            weights_np=weights_np,
            point_rgb01=rgb_srgb,
            neighbor_idx=neighbor_idx,
            neighbor_dist=neighbor_dist,
            color_dist=color_dist,
            edge_mask=edge_mask,
            edge_mask_pre_force=None,  # não há mais forçamento de arestas
            dist_ok=dist_ok,
            distance_threshold=distance_threshold,
            distance_desc=dist_desc,
            color_dist_threshold=color_dist_threshold,
            out_dir=os.path.join(args.output, "graph_debug"),
            max_edges_export=args.graph_debug_max_edges,
            max_edges_plot=args.graph_debug_max_edges_plot,
            show=args.graph_debug_show,
            seed=args.seed,
        )
        if args.graph_debug_only:
            print("\n🛑 --graph_debug_only: encerrando antes do treino.")
            raise SystemExit(0)

    # ==========================================================
    # MODELO (GAT ou GCN)
    # ==========================================================
    in_dim = x.shape[1]

    if args.model_type_choice == "gcn":
        w_min, w_max = weights_np.min(), weights_np.max()
        weights_norm = (weights_np - w_min) / (w_max - w_min + 1e-8)
        edge_weights_t = torch.tensor(weights_norm, dtype=torch.float, device=device)

        # ==========================================================
        # SELEÇÃO DO MODELO GCN (V2 ou DEEP)
        # ==========================================================
        if args.gcn_depth == "deep":
            print(f"🧠 Usando DeepGaussianGCN_ResNet (4 camadas com skip connections)")
            model = DeepGaussianGCN_ResNet(in_channels=in_dim, hidden_dim=64, out_dim=32).to(device)
        else:
            print(f"🧠 Usando GaussianGCNV2 (2 camadas)")
            model = GaussianGCNV2(in_channels=in_dim, out_dim=32).to(device)
        
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights_t).to(device)

    else:
        edge_weights_t = torch.tensor(weights_np, dtype=torch.float, device=device)
        model = GaussianGAT(in_channels=in_dim, heads=args.gat_heads).to(device)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights_t).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_color_sigma = float(args.loss_delta_e_sigma)
    colors_np = lab
    print(f"🎨 Loss de cor: CIELAB | sigma={loss_color_sigma:.2f} ΔE "
          f"(ΔE=2.3/JND -> similaridade "
          f"{np.exp(-(2.3 ** 2) / (2 * loss_color_sigma ** 2)):.2f}; "
          f"ΔE={args.edge_delta_e_threshold:.0f} -> "
          f"{np.exp(-(args.edge_delta_e_threshold ** 2) / (2 * loss_color_sigma ** 2)):.2f})")

    criterion = ColorAwareContrastiveLossV2(
        temperature=args.loss_temperature,
        pos_margin=args.loss_pos_margin,
        neg_margin=args.loss_neg_margin,
        color_weight=args.loss_color_weight,
        color_sigma=loss_color_sigma,
        color_exclude_diff_label=args.loss_color_exclude_diff_label,
        detach_centroids=args.loss_detach_centroids,
        nce_class_balanced=args.loss_nce_class_balanced,
        centroid_repulsion_weight=args.loss_centroid_repulsion_weight,
        pos_weight=args.loss_pos_weight,
        neg_weight=args.loss_neg_weight,
        nce_weight=args.loss_nce_weight,
    )
    print("🧮 InfoNCE contra centróides de rótulo (robusto a rótulo errado)")
    _fixes_on = []
    if args.loss_color_exclude_diff_label:
        _fixes_on.append("color_exclude_diff_label")
    if args.loss_detach_centroids:
        _fixes_on.append("detach_centroids")
    if args.loss_nce_class_balanced:
        _fixes_on.append("nce_class_balanced")
    if args.loss_centroid_repulsion_weight > 0:
        _fixes_on.append(f"centroid_repulsion_weight={args.loss_centroid_repulsion_weight}")
    print(f"🔧 Correções opcionais de loss (PCA borrado entre objetos): "
          f"{', '.join(_fixes_on) if _fixes_on else 'nenhuma ligada (comportamento anterior)'}")

    labels_t = torch.tensor(labels, device=device)
    colors_t = torch.tensor(colors_np, dtype=torch.float, device=device)

    # --- DIAGNÓSTICO TEMPORÁRIO: composição do grafo estrutural ---
    _src, _dst = data.edge_index
    _valid_e = (labels_t[_src] != -1) & (labels_t[_dst] != -1)
    _same_e = ((labels_t[_src] == labels_t[_dst]) & _valid_e).sum().item()
    _diff_e = ((labels_t[_src] != labels_t[_dst]) & _valid_e).sum().item()
    print(f"🔬 [DIAG] Arestas do grafo: {_same_e:,} same-label | {_diff_e:,} diff-label "
          f"({100*_diff_e/(_same_e+_diff_e+1e-8):.2f}% diff)")
    _uniq_lbl, _lbl_counts = torch.unique(labels_t, return_counts=True)
    print(f"🔬 [DIAG] {len(_uniq_lbl)} labels distintos | tamanho: "
          f"min={_lbl_counts.min().item()} max={_lbl_counts.max().item()} "
          f"média={_lbl_counts.float().mean().item():.0f}")

    # ==========================================================
    # LOOP DE TREINAMENTO
    # ==========================================================
    model_name = f"{args.model_type_choice.upper()}-{args.gcn_depth.upper()}" if args.model_type_choice == "gcn" else args.model_type_choice.upper()
    print(f"\n🎓 Treinando {model_name} com Early Stopping...")
    t_train_start = time.time()

    # Amostra a VRAM real do processo (via NVML) só durante o laço de
    # treino, para poder comparar diretamente com o que aparece no
    # nvidia-smi enquanto a rede está treinando — ver docstring de
    # NvmlVRAMSampler para o porquê disso divergir de max_memory_reserved.
    vram_sampler = NvmlVRAMSampler(interval=0.2)
    vram_sampler.start()

    early_stopping = EarlyStopping(patience=20, min_delta=0.0001, verbose=True)
    loss_history = []
    menor_loss = float('inf')

    for epoch in range(args.train_epochs):
        model.train()
        optimizer.zero_grad()

        if args.model_type_choice == "gcn":
            embeddings = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            embeddings = model(data.x, data.edge_index)

        z = F.normalize(embeddings, dim=1)

        loss = criterion(
            z,
            labels_t,
            data.edge_index,
            colors=colors_t,
            edge_weights=data.edge_weight
        )

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if current_loss < menor_loss:
            menor_loss = current_loss

        if epoch % 10 == 0:
            _t = getattr(criterion, "last_terms", None)
            if _t is None:
                print(f"Epoch {epoch:4d} | Loss: {current_loss:.4f} | Best: {menor_loss:.4f}")
            else:
                _flag = "  <-- nce ACIMA do piso" if _t["nce"] > _t["nce_floor"] else ""
                print(f"Epoch {epoch:4d} | Loss: {current_loss:.4f} | Best: {menor_loss:.4f} || "
                      f"pos={_t['pos']:.4f} neg={_t['neg']:.4f} nce={_t['nce']:.4f} "
                      f"(piso={_t['nce_floor']:.3f}) cor={_t['color']:.4f} "
                      f"cent_rep={_t['cent_rep']:.4f} "
                      f"| ancoras={_t['n_anchors']}{_flag}")

        if early_stopping(current_loss, model):
            break

    early_stopping.restore_best_model(model)

    training_vram_gb = vram_sampler.stop_and_get_peak_gb()

    t_train_end = time.time()
    training_time = t_train_end - t_train_start
    print(f"⏱️ Tempo de treinamento do modelo: {training_time:.2f}s")
    if training_vram_gb is not None:
        print(f"💾 VRAM real (NVML) durante o treino: {training_vram_gb:.2f} GB")

    # ==========================================================
    # EXTRAÇÃO DE EMBEDDINGS
    # ==========================================================
    model.eval()
    with torch.no_grad():
        if args.model_type_choice == "gcn":
            out_emb = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            out_emb = model(data.x, data.edge_index)
            
        emb = F.normalize(out_emb, dim=1).cpu().numpy()

    #emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    # ABLAÇÃO: clusteriza as FEATURES DE ENTRADA (xyz normalizado + Lab) em vez
    # do embedding aprendido, descartando a GNN por completo. Existe para
    # responder uma pergunta que o sweep de pesos levantou: se variar os pesos
    # da loss não muda o mIoU, a loss (e a rede) está de fato contribuindo, ou
    # o resultado vem das features e do HDBSCAN? Se esta ablação empatar com o
    # pipeline treinado, a etapa de aprendizado é decorativa.
    if args.cluster_on_raw_features:
        emb = x.cpu().numpy().astype(np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        print(f"🧪 ABLAÇÃO: clusterizando as features de ENTRADA ({emb.shape[1]}-D, "
              f"xyz+Lab), ignorando a GNN.")

    # Modelo/otimizador/grafo (GPU) só existem para produzir `emb`, que já foi
    # copiado para numpy acima. Sem liberar aqui, essa memória fica presa até
    # o fim do script e some da margem que os dois passes de renderização
    # (render_clusters + render_clusters_coverage, no fim do pipeline)
    # precisam para as ~130 câmeras da cena — foi o que causou OOM ali mesmo
    # com o grafo de treino já reduzido.
    if args.diagnose_embedding:
        diagnose_embedding(emb, labels, args.min_cluster_size, seed=args.seed)

    del model, optimizer, data, out_emb, edge_weights_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================================
    # HDBSCAN CLUSTERING
    # ==========================================================
    print("\n📊 Clusterizando os novos embeddings com HDBSCAN...")
    t_hdbscan_start = time.time()

    # Estes parâmetros estavam hardcoded aqui enquanto o relatório gravava
    # `args.min_cluster_size` — ou seja, a tabela dizia um valor e a execução
    # usava outro. Agora vêm dos argumentos, para o que é reportado ser o que
    # de fato rodou.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        #cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        cluster_selection_method=args.hdbscan_cluster_selection_method,
        # 'eom' já é o padrão do HDBSCAN, mas deixamos explícito de propósito:
        # 'leaf' seleciona clusters mais miúdos/homogêneos e fragmentaria mais.

        prediction_data=False
    ).fit(emb)

    cluster_labels = clusterer.labels_

    cluster_labels = reassign_noise_knn_vectorized(
        emb, 
        cluster_labels, 
        k=args.knn_reassign_k, 
        min_agreement=0.8
    )

    t_hdbscan_end = time.time()
    hdbscan_time = t_hdbscan_end - t_hdbscan_start
    print(f"⏱️ Tempo do HDBSCAN (+ KNN Reassign): {hdbscan_time:.2f}s")

    # ==========================================================
    # AVALIAÇÃO
    # ==========================================================
    print("\n" + "="*60)
    print("📊 AVALIAÇÃO DA SEGMENTAÇÃO 3D")
    print("="*60)
    
    semi_supervised_metrics = evaluate_semi_supervised(
        initial_labels=labels,
        refined_labels=cluster_labels
    )
    
    cluster_validity_metrics = evaluate_cluster_validity(
        embeddings=emb,
        labels=cluster_labels,
        xyz=xyz
    )
    
    spatial_metrics = evaluate_spatial_consistency(
        xyz=xyz,
        labels=cluster_labels
    )
    
    n_initial_clusters = len(np.unique(labels[labels != -1]))
    n_final_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    
    all_metrics = {
        **semi_supervised_metrics,
        **cluster_validity_metrics,
        **spatial_metrics,
        'Model_Type': args.model_type_choice,
        'GCN_Depth': args.gcn_depth if args.model_type_choice == "gcn" else "N/A",
        'GAT_Heads': args.gat_heads,
        'Min_Cluster_Size': args.min_cluster_size,
        'KNN_Reassign_K': args.knn_reassign_k,
        'Raw_Gaussians': total_gaussians_raw,
        'Filtered_Gaussians': total_gaussians_filtered,
        'Initial_Clusters': n_initial_clusters,
        'Refined_Clusters': n_final_clusters,
        'HDBSCAN_Time': hdbscan_time,
        'Propagate_Full_Pointcloud': args.render_full_pointcloud,
        'Propagate_Max_Distance': args.propagate_max_distance,
        # Hiperparâmetros da loss. Sem isso o relatório não permitia mapear um
        # resultado de volta para a configuração que o produziu — num sweep,
        # as rodadas ficavam indistinguíveis umas das outras.
        'Loss_Temperature': args.loss_temperature,
        'Loss_Pos_Margin': args.loss_pos_margin,
        'Loss_Neg_Margin': args.loss_neg_margin,
        'Loss_Pos_Weight': args.loss_pos_weight,
        'Loss_Neg_Weight': args.loss_neg_weight,
        'Loss_NCE_Weight': args.loss_nce_weight,
        'Loss_Color_Weight': args.loss_color_weight,
        'Loss_Centroid_Repulsion_Weight': args.loss_centroid_repulsion_weight,
        'Loss_Detach_Centroids': args.loss_detach_centroids,
        'Loss_NCE_Class_Balanced': args.loss_nce_class_balanced,
        'Loss_Color_Exclude_Diff_Label': args.loss_color_exclude_diff_label,
        'HDBSCAN_Min_Samples': args.hdbscan_min_samples,
        'Target_Gaussians': args.target_gaussians,
    }

    # Paleta única compartilhada entre a nuvem de pontos 3D e a renderização 2D,
    # para que o MESMO cluster tenha SEMPRE a MESMA cor nos dois resultados,
    # com distância perceptual máxima entre clusters (evita repetição de cor
    # entre objetos diferentes, ex: perna do urso / guardanapo / cadeira).
    unique_cluster_labels = [l for l in np.unique(cluster_labels) if l != -1]
    shared_palette = generate_distinct_colors(len(unique_cluster_labels))
    shared_color_map = {label: shared_palette[i] for i, label in enumerate(unique_cluster_labels)}
    shared_color_map[-1] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # ==========================================================
    # RENDERIZAÇÃO 2D
    # ==========================================================
    if args.render_images:
        print("\n" + "="*50 + "\nRENDER 2D\n" + "="*50)
        orig_dc = gaussians._features_dc.clone()
        pipe = pipeline_params.extract(args)
        background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
        
        n_total_gaussians = gaussians._xyz.shape[0]

        if args.render_full_pointcloud:
            # 🔧 PROPAGAR labels para TODAS as Gaussianas
            print("  🔄 Propagando labels para todas as Gaussianas...")
            full_cluster_labels = propagate_labels_to_full_set(
                xyz, cluster_labels, xyz_full_original, k=3,
                max_distance=args.propagate_max_distance if args.propagate_max_distance > 0 else None
            )
            
            # 🔧 VERIFICAR tamanho
            if len(full_cluster_labels) != n_total_gaussians:
                print(f"  ⚠️ Ajustando tamanho: {len(full_cluster_labels)} -> {n_total_gaussians}")
                if len(full_cluster_labels) > n_total_gaussians:
                    full_cluster_labels = full_cluster_labels[:n_total_gaussians]
                else:
                    temp = np.full(n_total_gaussians, -1)
                    temp[:len(full_cluster_labels)] = full_cluster_labels
                    full_cluster_labels = temp
            
            render_labels = full_cluster_labels
            render_mask = None  # ⚠️ NÃO passar mask_filter para renderizar TODAS
            
            print(f"  Gaussianas com cluster: {np.sum(render_labels != -1):,}")
        else:
            # 🔧 CORREÇÃO: render_labels DEVE ter o MESMO tamanho que mask_filter
            # quando mask_filter for usado
            print("  🎯 Renderizando apenas Gaussianas filtradas...")
            
            # cluster_labels já tem o tamanho correto (102541)
            render_labels = cluster_labels  # ⚠️ NÃO criar array com tamanho total
            render_mask = mask_filter
            
            print(f"  Gaussianas com cluster: {np.sum(render_labels != -1):,}")
            print(f"  Tamanho das labels: {len(render_labels)}")
            print(f"  Tamanho do mask_filter: {len(mask_filter)}")
        
        # Renderizar (reaproveita a paleta compartilhada com a nuvem de pontos 3D)
        render_clusters(
            render_labels,
            "images",
            gaussians,
            scene,
            pipe,
            background,
            args,
            device,
            render_mask,
            color_map=shared_color_map
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # `render_clusters` já escreve direto em `images_dir` (args.output/"images"
        # é o mesmo caminho de `images_dir`) — não há nada para mover. O bloco que
        # existia aqui tentava mover o diretório para dentro dele mesmo e depois
        # remover uma pasta não-vazia, o que sempre lançava OSError bem no fim do
        # pipeline (depois de tudo já ter sido salvo corretamente).

        gaussians._features_dc.data = orig_dc
        images_raw_dir = os.path.join(model_dir, "images_raw")
        print(f"🖼️ Imagens padronizadas (cor homogênea por cluster) salvas em: {images_dir}")
        print(f"🖼️ Imagens cruas (sem padronização de cor) salvas em: {images_raw_dir}")

        # Máscara discreta via coverage (sem decodificar cor — ver docstring de
        # render_clusters_coverage). Serve de entrada direta para mIoU, sem
        # precisar de KMeans/CRF a jusante como o pipeline baseado em cor exige.
        render_clusters_coverage(
            render_labels,
            "images",
            gaussians,
            scene,
            pipe,
            args,
            device,
            render_mask
        )

        # Render do embedding cru (PCA -> RGB). Usa sempre `mask_filter` e não
        # `render_mask`: o embedding só existe para as gaussianas que passaram
        # pelo filtro e foram treinadas, mesmo quando os clusters são
        # propagados para a nuvem completa.
        if args.render_pca:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            render_embedding_pca(emb, gaussians, scene, pipe, background,
                                 args, device, mask_filter)
    
    # ==========================================================
    # MEDIÇÃO DE VRAM PICO
    # ==========================================================
    # `max_vram_gb` (torch.cuda.max_memory_reserved) é o pico do pool que o
    # allocator do PyTorch reserva, medido no fim do pipeline inteiro. Serve
    # de referência histórica, mas SUBESTIMA sistematicamente o valor real
    # do nvidia-smi — alocações feitas por extensões CUDA de terceiros (ex.:
    # kernels de scatter/sort do torch_geometric) ficam fora do allocator do
    # PyTorch e, portanto, invisíveis a essa estatística; a proporção dessas
    # alocações "invisíveis" varia por arquitetura, então essa métrica pode
    # até inverter a ordem real de consumo entre configurações.
    #
    # `training_vram_gb` é a métrica confiável: pico de memória do PROCESSO
    # reportado pela NVML (a mesma API que o nvidia-smi consulta),
    # amostrado especificamente durante o laço de treino da rede (ver
    # NvmlVRAMSampler, iniciado/parado ao redor do loop de épocas).
    if torch.cuda.is_available():
        max_vram_bytes = torch.cuda.max_memory_reserved(device)
        max_vram_gb = max_vram_bytes / (1024 ** 3)
    else:
        max_vram_gb = 0.0

    print(f"💾 Pico de VRAM reservada (allocator PyTorch, pipeline inteiro): {max_vram_gb:.2f} GB")
    if training_vram_gb is not None:
        print(f"💾 VRAM real (NVML, apenas durante o treino): {training_vram_gb:.2f} GB")

    all_metrics['Max_VRAM_GB'] = max_vram_gb
    all_metrics['Training_VRAM_NVML_GB'] = training_vram_gb

    # ==========================================================
    # SALVAR RESULTADOS
    # ==========================================================
    excel_path, txt_path = save_experiment_results(
        metrics=all_metrics,
        loss_history=loss_history,
        output_dir=model_dir,
        model_name=model_name,
        training_time=training_time,
        graph_build_time=graph_build_time,
        hdbscan_time=hdbscan_time,
        max_vram_gb=max_vram_gb,
        training_vram_gb=training_vram_gb
    )

    # Salvar nuvem de pontos
    pointcloud_path = os.path.join(pointclouds_dir, f"{args.model_type_choice}_clusters.ply")
    metadata = {
        'model_type': args.model_type_choice,
        'gcn_depth': args.gcn_depth if args.model_type_choice == "gcn" else "N/A",
        'gat_heads': args.gat_heads,
        'min_cluster_size': args.min_cluster_size,
        'n_clusters': n_final_clusters,
        'raw_gaussians': total_gaussians_raw,
        'filtered_gaussians': total_gaussians_filtered,
        'max_vram_gb': max_vram_gb,
        'training_vram_nvml_gb': training_vram_gb
    }
    save_pointcloud_with_metadata(xyz, cluster_labels, pointcloud_path, metadata, color_map=shared_color_map)

    # ==========================================================
    # VISUALIZAÇÃO 3D
    # ==========================================================
    if args.visualize_3d and OPEN3D_AVAILABLE:
        print("\n🌐 Visualizando nuvem de pontos 3D...")
        visualize_clusters_open3d(xyz, cluster_labels, 
                                  f"{model_name} + HDBSCAN")
    
    # ==========================================================
    # RESUMO FINAL
    # ==========================================================
    total_pipeline_time = graph_build_time + training_time + hdbscan_time
    print("\n" + "="*80)
    print("✅ EXPERIMENTO CONCLUÍDO")
    print("="*80)
    print(f"📁 Diretório do experimento: {exp_dir}")
    print(f"📁 Modelo: {model_name}")
    print(f"📦 Gaussianas Originais (Cena): {total_gaussians_raw:,}")
    print(f"🎯 Gaussianas Filtradas (Treino): {total_gaussians_filtered:,}")
    print(f"💾 VRAM Pico (allocator PyTorch, pipeline inteiro): {max_vram_gb:.2f} GB")
    if training_vram_gb is not None:
        print(f"💾 VRAM Real (NVML, durante o treino): {training_vram_gb:.2f} GB")
    print(f"⏱️ Tempo Grafo: {graph_build_time:.2f}s | Treino: {training_time:.2f}s | HDBSCAN: {hdbscan_time:.2f}s | Total: {total_pipeline_time:.2f}s")
    print(f"📊 Clusters iniciais (DEVA/SAM): {n_initial_clusters}")
    print(f"📊 Clusters finais (HDBSCAN): {n_final_clusters}")
    print(f"📊 ARI: {semi_supervised_metrics['ARI']:.4f}")
    print(f"📊 NMI: {semi_supervised_metrics['NMI']:.4f}")
    print(f"📊 Silhouette: {cluster_validity_metrics['Silhouette']:.4f}")
    print(f"📊 Davies-Bouldin: {cluster_validity_metrics['Davies_Bouldin']:.4f}")
    print("-"*80)
    print(f"📊 Tabela de métricas: {excel_path}")
    print(f"📄 Relatório: {txt_path}")
    print(f"📊 Curva de loss: {os.path.join(model_dir, f'{model_name}_loss_curve.png')}")
    print(f"☁️ Nuvem de pontos: {pointcloud_path}")
    print(f"🖼️ Imagens renderizadas: {images_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()