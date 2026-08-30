import os
import json
import cv2
import time
import torch.nn as nn

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mode
from scipy.stats import mode as scipy_mode
from skimage.color import rgb2lab
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
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not installed. Install with: pip install open3d")

# ==========================================================
# LOSS CONTRASTIVA PONDERADA POR OPACIDADE
# ==========================================================


class CIELABShadowRobustLoss(nn.Module):
    def __init__(self, color_sigma=0.15, l_weight=0.0, use_smooth_l1=False):
        """
        Loss de cor baseada no espaço CIELAB insensível a sombras e variações de luz.
        
        Args:
            color_sigma (float): Tolerância de diferença cromática (recomendado entre 0.10 e 0.20).
            l_weight (float): Peso do canal L (Luminância). 
                              - 0.0 = Ignora totalmente sombras/luz (só olha a cor real).
                              - 0.05 a 0.1 = Dá um peso mínimo para sombras leves.
            use_smooth_l1 (bool): Se True, reduz o impacto de outliers.
        """
        super().__init__()
        self.color_sigma = color_sigma
        self.use_smooth_l1 = use_smooth_l1
        
        # Pesos para os canais [L, a, b]
        # Por padrão: L=0.0 (Ignora Iluminação), a=1.0 (Cor), b=1.0 (Cor)
        self.register_buffer("channel_weights", torch.tensor([l_weight, 1.0, 1.0]))

    def forward(self, embeddings, edge_index, colors_lab, edge_weights=None):
        """
        Args:
            embeddings (Tensor): (N, D) Embeddings da GNN.
            edge_index (Tensor): (2, E) Arestas do grafo.
            colors_lab (Tensor): (N, 3) Cores no espaço CIELAB normalizado [L_norm, a_norm, b_norm].
            edge_weights (Tensor, optional): Pesos geométricos/espaciais das arestas.
        """
        src, dst = edge_index

        # 1. Similaridade de Coseno dos embeddings aprendidos
        sim = F.cosine_similarity(embeddings[src], embeddings[dst], dim=1)

        # 2. Distância cromática PONDERADA (Ponderando/Zerando a diferença de iluminação L)
        diff = (colors_lab[src] - colors_lab[dst]) * self.channel_weights
        color_dist = torch.norm(diff, p=2, dim=1)

        # 3. Target: Kernel Gaussiano Cromático (RBF)
        # Se a cor (a, b) for idêntica, color_sim ~ 1.0 mesmo se uma parte estiver no escuro!
        color_sim = torch.exp(- (color_dist ** 2) / (2 * (self.color_sigma ** 2) + 1e-8))

        # 4. Cálculo da Perda
        if self.use_smooth_l1:
            loss_per_edge = F.smooth_l1_loss(sim, color_sim, reduction='none')
        else:
            loss_per_edge = (sim - color_sim) ** 2

        if edge_weights is not None:
            loss_per_edge = loss_per_edge * edge_weights

        return loss_per_edge.mean()

class ColorAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, pos_margin=0.5, neg_margin=0.2, 
                 color_weight=0.3, color_sigma=0.15):
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.color_weight = color_weight
        self.color_sigma = color_sigma
    
    def forward(self, embeddings, labels, edge_index, colors, edge_weights=None):
        src, dst = edge_index
        valid = (labels[src] != -1) & (labels[dst] != -1)
        
        # Similaridade de Coseno dos Embeddings da GNN
        sim = F.cosine_similarity(embeddings[src], embeddings[dst], dim=1)
        
        # ==========================================================
        # 1. LOSS SEMÂNTICA (Sua implementação original)
        # ==========================================================
        same_label = (labels[src] == labels[dst]) & valid
        diff_label = (labels[src] != labels[dst]) & valid
        
        pos_loss = torch.tensor(0.0, device=embeddings.device)
        if same_label.any():
            raw_pos = torch.relu(1 - sim[same_label] - self.pos_margin)
            if edge_weights is not None:
                pos_loss = (raw_pos * edge_weights[same_label]).mean()
            else:
                pos_loss = raw_pos.mean()
        
        neg_loss = torch.tensor(0.0, device=embeddings.device)
        if diff_label.any():
            raw_neg = torch.relu(sim[diff_label] - self.neg_margin)
            if edge_weights is not None:
                neg_loss = (raw_neg * edge_weights[diff_label]).mean()
            else:
                neg_loss = raw_neg.mean()
        
        # ==========================================================
        # 2. LOSS DE CONSISTÊNCIA DE COR (NOVO TERMO)
        # ==========================================================
        # Distância euclidiana de cor entre os nós da aresta (N_edges,)
        color_dist = torch.norm(colors[src] - colors[dst], p=2, dim=1)
        
        # Kernel Gaussiano: varia de 1.0 (cores idênticas) a 0.0 (cores muito diferentes)
        color_sim = torch.exp(- (color_dist ** 2) / (2 * (self.color_sigma ** 2) + 1e-8))
        
        # Força os embeddings a respeitarem a transição de cores
        # Se as cores forem parecidas (color_sim alto), sim deve ser alto.
        color_loss = F.mse_loss(sim, color_sim)

        # ==========================================================
        # 3. INFO-NCE LOSS (Sua implementação original)
        # ==========================================================
        uniq_labels = torch.unique(labels[labels != -1])
        nce_loss = torch.tensor(0.0, device=embeddings.device)
        nce_count = 0
        
        for label in uniq_labels:
            pos_mask = (labels == label) & (labels != -1)
            pos_embs = embeddings[pos_mask]
            
            if pos_embs.shape[0] < 2:
                continue
                
            for i in range(min(5, pos_embs.shape[0] - 1)):
                anchor = pos_embs[i]
                pos = pos_embs[i+1]
                
                neg_mask = (labels != label) & (labels != -1)
                neg_embs = embeddings[neg_mask]
                
                if neg_embs.shape[0] > 0:
                    neg_sim = F.cosine_similarity(anchor.unsqueeze(0), neg_embs)
                    hard_negatives = neg_embs[neg_sim.topk(min(10, neg_embs.shape[0])).indices]
                    
                    pos_sim = F.cosine_similarity(anchor.unsqueeze(0), pos.unsqueeze(0))
                    logits = torch.cat([pos_sim, F.cosine_similarity(anchor.unsqueeze(0), hard_negatives)])
                    nce_loss = nce_loss + -torch.log(torch.exp(logits[0]/self.temperature) / 
                                          (torch.exp(logits/self.temperature).sum() + 1e-8))
                    nce_count += 1
        
        if nce_count > 0:
            nce_loss = nce_loss / nce_count
        
        # Combinação final de perdas
        total_loss = (1-self.color_weight) * (pos_loss + neg_loss + nce_loss) + (self.color_weight * color_loss)
        return total_loss

class ColorAwareContrastiveLossV2(nn.Module):
    def __init__(self, temperature=0.07, pos_margin=0.5, neg_margin=0.2, 
                 color_weight=0.3, color_sigma=0.15, max_neg_candidates=512):
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.color_weight = color_weight
        self.color_sigma = color_sigma
        self.max_neg_candidates = max_neg_candidates  # Teto de candidatos para prevenir OOM
    
    def forward(self, embeddings, labels, edge_index, colors, edge_weights=None):
        src, dst = edge_index
        device = embeddings.device
        
        # 1. Normalização L2 prévia (transforma cosseno em produto escalar leve)
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        sim = (emb_norm[src] * emb_norm[dst]).sum(dim=1)
        
        # ==========================================================
        # 1. LOSS SEMÂNTICA
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
        # 2. LOSS DE CONSISTÊNCIA DE COR
        # ==========================================================
        color_dist = torch.norm(colors[src] - colors[dst], p=2, dim=1)
        color_sim = torch.exp(- (color_dist ** 2) / (2 * (self.color_sigma ** 2) + 1e-8))
        color_loss = F.mse_loss(sim, color_sim)

        # ==========================================================
        # 3. INFO-NCE LOSS (Economia de VRAM)
        # ==========================================================
        uniq_labels = torch.unique(labels[labels != -1])
        nce_loss = torch.tensor(0.0, device=device)
        nce_count = 0
        
        for label in uniq_labels:
            pos_mask = (labels == label)
            pos_indices = torch.where(pos_mask)[0]
            
            if pos_indices.shape[0] < 2:
                continue
                
            neg_mask = (labels != label) & (labels != -1)
            neg_indices = torch.where(neg_mask)[0]
            
            anchors_idx = pos_indices[torch.randperm(pos_indices.shape[0])[:min(5, pos_indices.shape[0])]]
            
            for anchor_idx in anchors_idx:
                anchor_emb = emb_norm[anchor_idx].unsqueeze(0)
                anchor_color = colors[anchor_idx].unsqueeze(0)
                
                # --- HARD POSITIVE MINING ---
                other_pos_indices = pos_indices[pos_indices != anchor_idx]
                with torch.no_grad():
                    pos_sims_no_grad = (anchor_emb * emb_norm[other_pos_indices]).sum(dim=1)
                    hard_pos_idx = other_pos_indices[pos_sims_no_grad.argmin()]
                
                # Retém gradiente apenas do positivo selecionado
                pos_sim = (anchor_emb * emb_norm[hard_pos_idx].unsqueeze(0)).sum(dim=1)
                
                # --- COLOR-AWARE HARD NEGATIVE MINING ---
                if neg_indices.shape[0] > 0:
                    # Amostragem limite de candidatos para evitar picos de memória
                    if neg_indices.shape[0] > self.max_neg_candidates:
                        perm = torch.randperm(neg_indices.shape[0], device=device)[:self.max_neg_candidates]
                        curr_neg_indices = neg_indices[perm]
                    else:
                        curr_neg_indices = neg_indices

                    # Mineração SEM GRADIENTE (libera ~90% de VRAM)
                    with torch.no_grad():
                        cand_embs = emb_norm[curr_neg_indices]
                        cand_colors = colors[curr_neg_indices]
                        
                        neg_emb_sim = (anchor_emb * cand_embs).sum(dim=1)
                        neg_color_dist = torch.norm(anchor_color - cand_colors, p=2, dim=1)
                        neg_color_sim = torch.exp(- (neg_color_dist ** 2) / (2 * (self.color_sigma ** 2) + 1e-8))
                        
                        combined_hardness = neg_emb_sim + self.color_weight * neg_color_sim
                        k_negs = min(10, curr_neg_indices.shape[0])
                        top_hard_rel_indices = combined_hardness.topk(k_negs).indices
                        selected_neg_indices = curr_neg_indices[top_hard_rel_indices]
                    
                    # Gradientes mantidos apenas para os Top-K Negativos finalistas
                    hard_neg_embs = emb_norm[selected_neg_indices]
                    hard_negatives_sim = (anchor_emb * hard_neg_embs).sum(dim=1)
                    
                    logits = torch.cat([pos_sim, hard_negatives_sim]).unsqueeze(0) / self.temperature
                    targets = torch.zeros(1, dtype=torch.long, device=device)
                    
                    nce_loss = nce_loss + F.cross_entropy(logits, targets)
                    nce_count += 1
        
        if nce_count > 0:
            nce_loss = nce_loss / nce_count
        
        total_loss = (1 - self.color_weight) * (pos_loss + neg_loss + nce_loss) + (self.color_weight * color_loss)
        return total_loss

class PureColorLoss(nn.Module):
    def __init__(self, color_sigma=0.15, use_smooth_l1=False):
        """
        Loss baseada puramente na consistência de cor sobre as arestas do grafo.
        
        Args:
            color_sigma (float): Tolerância de diferença de cor.
                                 Valores menores exigem cores mais idênticas.
            use_smooth_l1 (bool): Se True, usa Smooth L1 em vez de MSE para evitar
                                  que gradientes explodam em diferenças discrepantes.
        """
        super().__init__()
        self.color_sigma = color_sigma
        self.use_smooth_l1 = use_smooth_l1

    def forward(self, embeddings, edge_index, colors, edge_weights=None):
        """
        Args:
            embeddings (Tensor): (N, D) Embeddings gerados pela GNN.
            edge_index (Tensor): (2, E) Pares de arestas do grafo.
            colors (Tensor): (N, C) Cores das gaussianas (RGB ou CIELAB).
            edge_weights (Tensor, optional): Pesos geométricos/espaciais das arestas.
        """
        src, dst = edge_index

        # 1. Similaridade de Coseno entre os embeddings dos nós conectados (E,)
        sim = F.cosine_similarity(embeddings[src], embeddings[dst], dim=1)

        # 2. Distância Euclidiana de Cor entre os nós (E,)
        color_dist = torch.norm(colors[src] - colors[dst], p=2, dim=1)

        # 3. Target: Kernel Gaussiano de Cor (RBF)
        # Varia de 1.0 (cores idênticas) a 0.0 (cores muito distantes)
        color_sim = torch.exp(- (color_dist ** 2) / (2 * (self.color_sigma ** 2) + 1e-8))

        # 4. Cálculo da Perda (MSE ou Huber/Smooth L1)
        if self.use_smooth_l1:
            loss_per_edge = F.smooth_l1_loss(sim, color_sim, reduction='none')
        else:
            loss_per_edge = (sim - color_sim) ** 2

        # Aplica o peso da aresta se fornecido no grafo
        if edge_weights is not None:
            loss_per_edge = loss_per_edge * edge_weights

        return loss_per_edge.mean()

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, pos_margin=0.5, neg_margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
    
    def forward(self, embeddings, labels, edge_index, edge_weights=None):
        src, dst = edge_index
        valid = (labels[src] != -1) & (labels[dst] != -1)
        
        sim = F.cosine_similarity(embeddings[src], embeddings[dst], dim=1)
        
        same_label = (labels[src] == labels[dst]) & valid
        diff_label = (labels[src] != labels[dst]) & valid
        
        pos_loss = torch.tensor(0.0, device=embeddings.device)
        if same_label.any():
            raw_pos = torch.relu(1 - sim[same_label] - self.pos_margin)
            if edge_weights is not None:
                pos_loss = (raw_pos * edge_weights[same_label]).mean()
            else:
                pos_loss = raw_pos.mean()
        
        neg_loss = torch.tensor(0.0, device=embeddings.device)
        if diff_label.any():
            raw_neg = torch.relu(sim[diff_label] - self.neg_margin)
            if edge_weights is not None:
                neg_loss = (raw_neg * edge_weights[diff_label]).mean()
            else:
                neg_loss = raw_neg.mean()
        
        uniq_labels = torch.unique(labels[labels != -1])
        nce_loss = torch.tensor(0.0, device=embeddings.device)
        nce_count = 0
        
        for label in uniq_labels:
            pos_mask = (labels == label) & (labels != -1)
            pos_embs = embeddings[pos_mask]
            
            if pos_embs.shape[0] < 2:
                continue
                
            for i in range(min(5, pos_embs.shape[0] - 1)):
                anchor = pos_embs[i]
                pos = pos_embs[i+1]
                
                neg_mask = (labels != label) & (labels != -1)
                neg_embs = embeddings[neg_mask]
                
                if neg_embs.shape[0] > 0:
                    neg_sim = F.cosine_similarity(anchor.unsqueeze(0), neg_embs)
                    hard_negatives = neg_embs[neg_sim.topk(min(10, neg_embs.shape[0])).indices]
                    
                    pos_sim = F.cosine_similarity(anchor.unsqueeze(0), pos.unsqueeze(0))
                    logits = torch.cat([pos_sim, F.cosine_similarity(anchor.unsqueeze(0), hard_negatives)])
                    nce_loss = nce_loss + -torch.log(torch.exp(logits[0]/self.temperature) / 
                                          (torch.exp(logits/self.temperature).sum() + 1e-8))
                    nce_count += 1
        
        if nce_count > 0:
            nce_loss = nce_loss / nce_count
        
        return pos_loss + neg_loss + nce_loss


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
# FILTRO DINÂMICO
# ==========================================================
def compute_dynamic_filters(opacity, scaling, target_count=220000):
    """
    Calcula thresholds dinâmicos para atingir aproximadamente target_count gaussianas.
    Prioriza o ajuste de opacidade e GARANTE que count >= target_count.
    """
    n_total = len(opacity)
    print(f"\n🎯 Calculando filtros dinâmicos para ~{target_count:,} gaussianas...")
    print(f"   Total disponível: {n_total:,}")
    
    if n_total <= target_count:
        print(f"   ⚠️ Total ({n_total:,}) já é menor que o target ({target_count:,})")
        print(f"   Usando threshold mínimo (0.01) para manter todas as gaussianas")
        return 0, 10, n_total
    
    best_op_thresh = None
    best_sc_thresh = 0.7
    best_count = 0
    best_diff = float('inf')
    
    # 🔥 PRIORIDADE 1: Encontrar combinação com count >= target_count
    for op_thresh in np.linspace(0.0001, 0.5, 50):
        for sc_thresh in np.linspace(0.3, 50.0, 50):
            mask = (opacity > op_thresh) & (scaling < sc_thresh)
            count = np.sum(mask)
            
            # 🔥 Só considera se count >= target_count
            if count >= target_count:
                diff = count - target_count  # Queremos o mais próximo de target
                if diff < best_diff:
                    best_diff = diff
                    best_count = count
                    best_op_thresh = op_thresh
                    best_sc_thresh = sc_thresh
                    
                    # Se estiver muito próximo, para a busca
                    if diff < 500:
                        break
        if best_diff < 500:
            break
    
    # 🔥 PRIORIDADE 2: Se nenhuma combinação >= target, usa a mais próxima
    if best_op_thresh is None:
        print("   ⚠️ Nenhuma combinação atingiu o target. Usando a mais próxima...")
        best_diff = float('inf')
        for op_thresh in np.linspace(0.01, 0.5, 50):
            for sc_thresh in np.linspace(0.3, 2.0, 20):
                mask = (opacity > op_thresh) & (scaling < sc_thresh)
                count = np.sum(mask)
                diff = abs(count - target_count)
                if diff < best_diff:
                    best_diff = diff
                    best_count = count
                    best_op_thresh = op_thresh
                    best_sc_thresh = sc_thresh
    
    print(f"   ✅ Melhor combinação encontrada:")
    print(f"      Opacity threshold: {best_op_thresh:.3f}")
    print(f"      Scale threshold: {best_sc_thresh:.3f}")
    print(f"      Gaussianas resultantes: {best_count:,}")
    print(f"      {'✅ Target atingido!' if best_count >= target_count else f'⚠️ Faltam {target_count - best_count:,} para o target'}")
    
    return best_op_thresh, best_sc_thresh, best_count


# ==========================================================
# PROJECTION & LABELS
# ==========================================================
def project_points(xyz, view):
    P = view.full_proj_transform.detach().cpu().numpy()
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0],1))], axis=1)
    proj = xyz_h @ P.T
    proj = proj[:, :3] / (proj[:, 3:4] + 1e-8)
    u = ((proj[:,0]*0.5+0.5)*view.image_width).astype(int)
    v = ((proj[:,1]*0.5+0.5)*view.image_height).astype(int)
    return u, v

def project_points_with_depth(xyz, view, out_width=None, out_height=None):
    """
    Projeta pontos 3D no espaço de tela (pixels) e retorna profundidade (z_cam),
    seguindo a convenção exata do repositório oficial Gaussian Splatting (3DGS).
    """
    import torch

    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()

    N = xyz.shape[0]
    p_hom = np.hstack([xyz, np.ones((N, 1), dtype=np.float32)])

    # Matrizes de transformação da câmera COLMAP
    w2c = view.world_view_transform.detach().cpu().numpy()
    P = view.full_proj_transform.detach().cpu().numpy()

    # 1. Projeção para o espaço de câmera (obtenção do Z real)
    p_cam = p_hom @ w2c
    z_cam = p_cam[:, 2].copy()

    # 2. Projeção NDC
    proj = p_hom @ P
    w_vals = proj[:, 3:4]
    w_safe = np.where(np.abs(w_vals) < 1e-8, 1e-8, w_vals)
    proj_ndc = proj[:, :3] / w_safe

    # 3. Dimensões alvo da imagem
    W = out_width if out_width is not None else view.image_width
    H = out_height if out_height is not None else view.image_height

    # 4. Mapeamento NDC [-1, 1] -> Pixels [0, W-1 / 0, H-1]
    # O padrão oficial 3DGS utiliza np.floor no mapeamento discreto de pixels
    u = np.floor((proj_ndc[:, 0] * 0.5 + 0.5) * W).astype(np.int32)
    v = np.floor((0.5 - proj_ndc[:, 1] * 0.5) * H).astype(np.int32)

    # 5. Validação de frustum
    valid_ndc = (np.abs(proj_ndc[:, 0]) <= 1.0) & (np.abs(proj_ndc[:, 1]) <= 1.0)# & (proj_ndc[:, 2] <= 1.0)

    u[~valid_ndc] = -1
    v[~valid_ndc] = -1
    z_cam[~valid_ndc] = -1

    return u, v, z_cam



def smooth_labels_spatially(xyz, labels, k=12, min_agreement=0.6, n_passes=2):
    """
    Suaviza labels via votação de maioria entre vizinhos espaciais (k-NN).
    Só troca o label da gaussiana se uma fração >= min_agreement dos
    vizinhos concordar em outro label diferente do atual.
    """
    labels = labels.copy()
    valid_mask = labels != -1
    if valid_mask.sum() == 0:
        return labels

    for p in range(n_passes):
        valid_idx = np.where(labels != -1)[0]
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(xyz[valid_idx])
        _, nn_idx = nbrs.kneighbors(xyz[valid_idx])
        nn_idx = nn_idx[:, 1:]  # remove o próprio ponto (primeiro vizinho = ele mesmo)

        neighbor_labels = labels[valid_idx[nn_idx]]  # (M, k)
        new_labels = labels[valid_idx].copy()

        for i in range(len(valid_idx)):
            vals, counts = np.unique(neighbor_labels[i], return_counts=True)
            top = vals[np.argmax(counts)]
            top_frac = counts.max() / k
            if top != new_labels[i] and top_frac >= min_agreement:
                new_labels[i] = top

        changed = np.sum(new_labels != labels[valid_idx])
        labels[valid_idx] = new_labels
        print(f"  Passe {p+1}/{n_passes} de suavização: {changed:,} gaussianas trocaram de label")

    return labels


def build_labels_with_deva_json_v2_black(
    scene,
    xyz,
    masks_path,
    deva_json_path,
    min_score=0.3,
    min_area=20,
    max_area=22500,
    depth_tolerance=20,
    color_tolerance=0.25,
    use_color_validation=False,
    propagate_to_neighbors=False,
    k_propagate=1,
    depth_rel_tol=0.015,
    min_votes=5,
):
    N = xyz.shape[0]

    # -------------------------------------------------------------------------
    # 1. Carregamento do JSON do DEVA e indexação
    # -------------------------------------------------------------------------
    print(f"📖 Carregando anotações do DEVA em: {deva_json_path}")
    with open(deva_json_path, "r") as f:
        deva_data = json.load(f)

    deva_meta = {}
    total_segments_raw = 0
    total_segments_kept = 0

    for ann in deva_data.get("annotations", []):
        fname = ann["file_name"]
        deva_meta[fname] = {}
        for seg in ann.get("segments_info", []):
            total_segments_raw += 1
            sid = seg["id"]
            score = seg.get("score", 1.0)
            area = seg.get("area", 0)

            if score >= min_score:
                total_segments_kept += 1
                deva_meta[fname][sid] = {
                    "score": score,
                    "area": area,
                    "category_id": seg.get("category_id"),
                }

    print(f"  └─ Segmentos no JSON: {total_segments_raw} total | {total_segments_kept} mantidos (score>={min_score})")

    # -------------------------------------------------------------------------
    # 2. Estruturas para Acumulação Ponderada
    # -------------------------------------------------------------------------
    train_cameras = scene.getTrainCameras()
    total_cameras = len(train_cameras)

    votes_weight = [defaultdict(float) for _ in range(N)]
    votes_count = [defaultdict(int) for _ in range(N)]

    processed = 0
    frames_com_mascara_lida = 0
    total_pontos_antes_zbuffer = 0
    total_pontos_depois_zbuffer = 0

    print(f"🔍 Montando labels (projeção 1:1 na resolução do COLMAP com Z-buffer)...")

    for view in train_cameras:
        name = view.image_name
        json_frame_key = f"{name}.jpg" if not name.endswith((".jpg", ".png")) else name

        frame_seg_info = deva_meta.get(json_frame_key, {})
        if not frame_seg_info:
            continue

        mask_path = os.path.join(masks_path, f"{name}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        # 🔥 AJUSTE 1: Dimensões originais da câmera COLMAP
        W_orig = view.image_width
        H_orig = view.image_height

        # 🔥 AJUSTE 2: Redimensiona a máscara 1/4x para 1x usando INTER_NEAREST (preserva os IDs exatos)
        if mask.shape[1] != W_orig or mask.shape[0] != H_orig:
            mask = cv2.resize(mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

        H, W = H_orig, W_orig

        # Identificação de pixels não-pretos
        if mask.ndim == 3:
            is_not_black = np.any(mask != 0, axis=-1)
            mask_single = mask[..., 0]
        else:
            is_not_black = (mask != 0)
            mask_single = mask

        frames_com_mascara_lida += 1
        processed += 1
        if processed % 50 == 0:
            print(f"  Processando frame {processed}/{total_cameras}...")

        # 🔥 AJUSTE 3: Projeção de pontos com a resolução total (1x)
        u, v, z_cam = project_points_with_depth(xyz, view, out_width=W, out_height=H)

        valid = (z_cam > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue

        # Z-BUFFER (calculado no grid denso do COLMAP)
        u_valid = u[idx].astype(np.int64)
        v_valid = v[idx].astype(np.int64)
        z_valid = z_cam[idx]

        pixel_id = v_valid * W + u_valid

        order = np.lexsort((z_valid, pixel_id))
        pixel_id_sorted = pixel_id[order]
        z_sorted = z_valid[order]
        idx_sorted = idx[order]

        uniq_pixels, first_pos = np.unique(pixel_id_sorted, return_index=True)
        z_min_per_pixel = z_sorted[first_pos]

        pixel_lookup = np.searchsorted(uniq_pixels, pixel_id_sorted)
        z_min_broadcast = z_min_per_pixel[pixel_lookup]

        keep = z_sorted <= z_min_broadcast * (1.0 + depth_rel_tol)

        idx = idx_sorted[keep]
        u_kept = u[idx]
        v_kept = v[idx]

        total_pontos_antes_zbuffer += idx_sorted.size
        total_pontos_depois_zbuffer += idx.size

        if idx.size == 0:
            continue

        # Filtro de pixels de fundo (superfícies não-pretas)
        valid_surface = is_not_black[v_kept, u_kept]
        if not np.any(valid_surface):
            continue

        idx = idx[valid_surface]
        u_kept = u_kept[valid_surface]
        v_kept = v_kept[valid_surface]

        # Obtém os IDs das máscaras alinhadas
        sids = mask_single[v_kept, u_kept].astype(np.int64)

        cam_center = view.camera_center.detach().cpu().numpy() if hasattr(view, 'camera_center') else None

        for gi, sid in zip(idx, sids):
            if sid > 0 and sid in frame_seg_info:
                score = frame_seg_info[sid]["score"]
                area = frame_seg_info[sid]["area"]

                if cam_center is not None:
                    dist = np.linalg.norm(xyz[gi] - cam_center) + 1e-5
                    dist_weight = 1.0 / dist
                else:
                    dist_weight = 1.0

                area_weight = 1.0 / np.log1p(area)
                weight = score * dist_weight * area_weight

                votes_weight[gi][sid] += weight
                votes_count[gi][sid] += 1

    if total_pontos_antes_zbuffer > 0:
        reducao = 100 * (1 - total_pontos_depois_zbuffer / total_pontos_antes_zbuffer)
        print(f"\n📉 Z-buffer: {total_pontos_antes_zbuffer:,} → {total_pontos_depois_zbuffer:,} votos "
              f"({reducao:.1f}% descartados por oclusão)")

    # -------------------------------------------------------------------------
    # 3. Atribuição por Pontuação Média
    # -------------------------------------------------------------------------
    labels = np.full(N, -1, dtype=np.int64)

    for i in range(N):
        if votes_weight[i]:
            avg_scores = {
                sid: votes_weight[i][sid] / votes_count[i][sid]
                for sid in votes_weight[i]
                if votes_count[i][sid] >= min_votes
            }
            if avg_scores:
                labels[i] = max(avg_scores, key=avg_scores.get)

    assigned = np.sum(labels != -1)
    print(f"\n✅ Concluído:")
    print(f"  • Frames com máscara lida: {frames_com_mascara_lida}/{total_cameras}")
    print(f"  • Gaussianas rotuladas: {assigned}/{N} ({100 * assigned / N:.1f}%)")

    # -------------------------------------------------------------------------
    # 4. Propagação KNN Opcional
    # -------------------------------------------------------------------------
    if propagate_to_neighbors:
        valid_idx = np.where(labels != -1)[0]
        invalid_idx = np.where(labels == -1)[0]
        if len(valid_idx) > 0 and len(invalid_idx) > 0:
            print(f"  🔄 Propagando {len(invalid_idx):,} gaussianas via KNN (k={k_propagate})...")
            nbrs = NearestNeighbors(n_neighbors=k_propagate, algorithm="auto").fit(xyz[valid_idx])
            _, nn_idx = nbrs.kneighbors(xyz[invalid_idx])
            if k_propagate == 1:
                labels[invalid_idx] = labels[valid_idx[nn_idx.flatten()]]
            else:
                neigh_labels = labels[valid_idx[nn_idx]]
                labels[invalid_idx] = mode(neigh_labels, axis=1, keepdims=False).mode

            final_assigned = np.sum(labels != -1)
            print(f"  • Total após KNN: {final_assigned}/{N} ({100 * final_assigned / N:.1f}%)")

    return labels


# ==========================================================
# PROPAGAÇÃO DE LABELS PARA O CONJUNTO COMPLETO (SPAÇAL)
# ==========================================================
def propagate_labels_to_full_set(xyz_filtered, cluster_labels, xyz_full, k=1):
    """
    Propaga os labels de cluster do subconjunto filtrado para TODAS as gaussianas 
    originais usando o vizinho espacial mais próximo (corrige buracos brancos).
    """
    print(f"\n🔁 Propagando labels para o conjunto completo de gaussianas...")
    print(f"   Subconjunto filtrado: {len(xyz_filtered):,} | Conjunto completo: {len(xyz_full):,}")

    nbrs_full = NearestNeighbors(n_neighbors=k).fit(xyz_filtered)
    _, nn_idx = nbrs_full.kneighbors(xyz_full)

    if k == 1:
        full_labels = cluster_labels[nn_idx.flatten()]
    else:
        neighbor_labels = cluster_labels[nn_idx]  
        full_labels = np.array([
            np.bincount(row[row != -1]).argmax() if np.any(row != -1) else -1
            for row in neighbor_labels
        ])

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


def create_viewer_script(save_dir, output_dir):
    viewer_script = f'''#!/usr/bin/env python3
import open3d as o3d

def main():
    print("VISUALIZADOR DE SEGMENTAÇÃO GAT + HDBSCAN")
    pcd = o3d.io.read_point_cloud("{save_dir}/gat_clusters.ply")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
'''
    viewer_path = os.path.join(output_dir, "view_segmentation.py")
    with open(viewer_path, 'w') as f:
        f.write(viewer_script)
    os.chmod(viewer_path, 0o755)


import colorsys
import random

def generate_distinct_colors(n):
    """
    Gera N cores RGB visualmente MUITO distintas.
    Usa uma paleta base de cores puras e depois as embaralha 
    para evitar que clusters vizinhos fiquem com cores parecidas.
    """
    # Paleta de cores altamente contrastantes (Vermelho, Azul, Verde, Amarelo, Roxo, Laranja, etc.)
    base_colors = [
        [1.0, 0.0, 0.0],    # Vermelho Puro
        [0.0, 0.0, 1.0],    # Azul Puro
        [0.0, 1.0, 0.0],    # Verde Puro
        [1.0, 1.0, 0.0],    # Amarelo Puro
        [1.0, 0.0, 1.0],    # Magenta / Roxo
        [0.0, 1.0, 1.0],    # Ciano
        [1.0, 0.5, 0.0],    # Laranja
        [0.5, 0.0, 1.0],    # Violeta
        [0.0, 0.5, 1.0],    # Azul Claro
        [1.0, 0.0, 0.5],    # Rosa Forte
        [0.5, 1.0, 0.0],    # Verde Limão
        [0.0, 1.0, 0.5],    # Turquesa
        [0.5, 0.5, 1.0],    # Lavanda
        [1.0, 0.5, 0.5],    # Salmão
        [0.5, 1.0, 1.0],    # Gelado
    ]

    colors = []

    # Se o número de clusters for maior que a paleta, geramos variações 
    # mas ainda com saturação máxima (1.0) e luminosidade alta (0.9)
    if n <= len(base_colors):
        # Embaralha para que vizinhos não peguem cores sequenciais
        random.shuffle(base_colors)
        colors = np.array(base_colors[:n])
    else:
        # Para muitos clusters, usamos a roda de cores, mas com saturação e brilho fixos
        golden_ratio_conjugate = 0.618033988749895
        h = random.random()  # começa em ponto aleatório
        for _ in range(n):
            h = (h + golden_ratio_conjugate) % 1.0
            # Saturação e Valor FIXOS em 0.9 e 0.9 para garantir cores "vivas"
            rgb = colorsys.hsv_to_rgb(h, 0.95, 0.95) 
            colors.append(rgb)
        colors = np.array(colors)

    return colors

def render_clusters_tab20(labels, name, gaussians, scene, pipe, background, args, device, mask_filter=None):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(1, num_clusters)))[:, :3]
    np.random.shuffle(colors)
    
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    if -1 in color_map:
        color_map[-1] = np.array([0.0, 0.0, 0.0])  # Ruído/Sem classe fica estritamente PRETO

    gaussian_colors = np.array([color_map[l] for l in labels])
    colors_tensor_filtered = torch.tensor(gaussian_colors, dtype=torch.float32, device=device)

    orig_data = {
        'xyz': gaussians._xyz.data.clone(),
        'opacity': gaussians._opacity.data.clone(),
        'features_dc': gaussians._features_dc.data.clone(),
        'features_rest': gaussians._features_rest.data.clone(),
        'scaling': gaussians._scaling.data.clone(),
        'rotation': gaussians._rotation.data.clone()
    }

    if mask_filter is not None:
        mask = torch.tensor(mask_filter, device=device, dtype=torch.bool)
        gaussians._xyz.data = orig_data['xyz'][mask]
        gaussians._opacity.data = orig_data['opacity'][mask]
        gaussians._scaling.data = orig_data['scaling'][mask]
        gaussians._rotation.data = orig_data['rotation'][mask]
        
        gaussians._features_dc.data = (colors_tensor_filtered.unsqueeze(1) - 0.5) / 0.28209
        new_rest_shape = [gaussians._xyz.shape[0], orig_data['features_rest'].shape[1], 3]
        gaussians._features_rest.data = torch.zeros(new_rest_shape, device=device)
    else:
        gaussians._features_dc.data = (colors_tensor_filtered.unsqueeze(1) - 0.5) / 0.28209
        gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)

    out_dir = os.path.join(args.output, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering {num_clusters} discrete clusters to {out_dir}...")

    for view in scene.getTrainCameras():
        render_pkg = render(view, gaussians, pipe, background)
        img = render_pkg["render"]
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"), img_bgr)

    gaussians._xyz.data = orig_data['xyz']
    gaussians._opacity.data = orig_data['opacity']
    gaussians._features_dc.data = orig_data['features_dc']
    gaussians._features_rest.data = orig_data['features_rest']
    gaussians._scaling.data = orig_data['scaling']
    gaussians._rotation.data = orig_data['rotation']


def render_clusters(labels, name, gaussians, scene, pipe, background, args, device, mask_filter=None):
    """
    Renderiza Gaussianas coloridas por cluster.
    Gaussianas fora do mask_filter têm opacidade ZERO (invisíveis).
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    colors = generate_distinct_colors(num_clusters)

    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    if -1 in color_map:
        color_map[-1] = np.array([0.0, 0.0, 0.0])

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
    
    # 3c. Manter features_rest inalterado (tamanho original)
    # Só zerar se necessário para evitar artefatos
    if gaussians._features_rest.shape[1] > 0:
        # Pode manter original ou zerar completamente
        # gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)
        pass  # Manter original é mais seguro

    # ==========================================================
    # 4. Renderizar
    # ==========================================================
    out_dir = os.path.join(args.output, name)
    os.makedirs(out_dir, exist_ok=True)
    
    n_active = np.sum(mask_filter) if mask_filter is not None else n_total
    print(f"Rendering {num_clusters} discrete clusters to {out_dir}...")
    print(f"  Gaussianas totais: {n_total:,}")
    print(f"  Gaussianas ativas (opacity>0): {n_active:,}")
    print(f"  Gaussianas ocultas (opacity=0): {n_total - n_active:,}")

    for view in scene.getTrainCameras():
        render_pkg = render(view, gaussians, pipe, background)
        img = render_pkg["render"]
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"), img_bgr)

    # ==========================================================
    # 5. Restaurar dados originais
    # ==========================================================
    gaussians._xyz.data = orig_data['xyz']
    gaussians._opacity.data = orig_data['opacity']
    gaussians._features_dc.data = orig_data['features_dc']
    gaussians._features_rest.data = orig_data['features_rest']
    gaussians._scaling.data = orig_data['scaling']
    gaussians._rotation.data = orig_data['rotation']



def render_clusters_tab20(labels, name, gaussians, scene, pipe, background, args, device, mask_filter=None):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(1, num_clusters)))[:, :3]
    np.random.shuffle(colors)
    
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    if -1 in color_map:
        color_map[-1] = np.array([0.0, 0.0, 0.0])  # Ruído/Sem classe fica estritamente PRETO

    gaussian_colors = np.array([color_map[l] for l in labels])
    colors_tensor_filtered = torch.tensor(gaussian_colors, dtype=torch.float32, device=device)

    orig_data = {
        'xyz': gaussians._xyz.data.clone(),
        'opacity': gaussians._opacity.data.clone(),
        'features_dc': gaussians._features_dc.data.clone(),
        'features_rest': gaussians._features_rest.data.clone(),
        'scaling': gaussians._scaling.data.clone(),
        'rotation': gaussians._rotation.data.clone()
    }

    if mask_filter is not None:
        mask = torch.tensor(mask_filter, device=device, dtype=torch.bool)
        gaussians._xyz.data = orig_data['xyz'][mask]
        gaussians._opacity.data = orig_data['opacity'][mask]
        gaussians._scaling.data = orig_data['scaling'][mask]
        gaussians._rotation.data = orig_data['rotation'][mask]
        
        gaussians._features_dc.data = (colors_tensor_filtered.unsqueeze(1) - 0.5) / 0.28209
        new_rest_shape = [gaussians._xyz.shape[0], orig_data['features_rest'].shape[1], 3]
        gaussians._features_rest.data = torch.zeros(new_rest_shape, device=device)
    else:
        gaussians._features_dc.data = (colors_tensor_filtered.unsqueeze(1) - 0.5) / 0.28209
        gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)

    out_dir = os.path.join(args.output, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering {num_clusters} discrete clusters to {out_dir}...")

    for view in scene.getTrainCameras():
        render_pkg = render(view, gaussians, pipe, background)
        img = render_pkg["render"]
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"), img_bgr)

    gaussians._xyz.data = orig_data['xyz']
    gaussians._opacity.data = orig_data['opacity']
    gaussians._features_dc.data = orig_data['features_dc']
    gaussians._features_rest.data = orig_data['features_rest']
    gaussians._scaling.data = orig_data['scaling']
    gaussians._rotation.data = orig_data['rotation']

def generate_unique_colors(num_clusters, seed=42):
    """Gera N cores RGB (float32 entre [0.0, 1.0]) bem distintas no espaço HSV."""
    np.random.seed(seed)
    hues = np.linspace(0, 1, num_clusters, endpoint=False)
    np.random.shuffle(hues)
    
    colors = []
    for h in hues:
        s = np.random.uniform(0.65, 1.0)
        v = np.random.uniform(0.70, 1.0)
        colors.append(mcolors.hsv_to_rgb([h, s, v]))
        
    return np.array(colors, dtype=np.float32)

def render_clusters_uniform_cuda_tree(labels, name, gaussians, scene, pipe, background, args, device, mask_filter=None):
    """
    Renderiza cada cluster com uma cor única e uniforme usando o rasterizador CUDA.
    Substitui o mapa fixo por geração dinâmica HSV (sem repetição) e cKDTree para pós-processamento.
    """
    unique_labels = np.unique(labels)
    valid_labels = [l for l in unique_labels if l != -1]
    num_clusters = len(valid_labels)
    
    # 1. Gerar cores únicas sem limite de limite de tabela (sem repetição)
    unique_colors = generate_unique_colors(num_clusters)
    
    # 2. Mapeamento de rótulos para cores RGB
    color_map = {label: unique_colors[i] for i, label in enumerate(valid_labels)}
    if -1 in unique_labels:
        color_map[-1] = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Ruído fica preto
        
    # 3. Construir a árvore k-d com as cores dos clusters válidos
    valid_colors = np.array([color_map[l] for l in valid_labels], dtype=np.float32)
    color_tree = cKDTree(valid_colors)
    
    # 4. Converter cores para cada Gaussiana e transferir para GPU
    gaussian_colors = np.array([color_map[l] for l in labels], dtype=np.float32)
    colors_tensor = torch.tensor(gaussian_colors, dtype=torch.float32, device=device)
    
    # 5. Fazer backup dos parâmetros originais do modelo
    orig_data = {
        'xyz': gaussians._xyz.data.clone(),
        'opacity': gaussians._opacity.data.clone(),
        'features_dc': gaussians._features_dc.data.clone(),
        'features_rest': gaussians._features_rest.data.clone(),
        'scaling': gaussians._scaling.data.clone(),
        'rotation': gaussians._rotation.data.clone()
    }
    
    # 6. Atribuir cores no formato de harmônicos esféricos (DC) e zerar harmônicos superiores
    if mask_filter is not None:
        mask = torch.tensor(mask_filter, device=device, dtype=torch.bool)
        gaussians._xyz.data = orig_data['xyz'][mask]
        gaussians._opacity.data = orig_data['opacity'][mask]
        gaussians._scaling.data = orig_data['scaling'][mask]
        gaussians._rotation.data = orig_data['rotation'][mask]
        
        gaussians._features_dc.data = (colors_tensor[mask].unsqueeze(1) - 0.5) / 0.28209
        new_rest_shape = [gaussians._xyz.shape[0], orig_data['features_rest'].shape[1], 3]
        gaussians._features_rest.data = torch.zeros(new_rest_shape, device=device)
    else:
        gaussians._features_dc.data = (colors_tensor.unsqueeze(1) - 0.5) / 0.28209
        gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)
    
    out_dir = os.path.join(args.output, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering {num_clusters} unique uniform clusters (CUDA) to {out_dir}...")
    
    # 7. Renderização e pós-processamento por frame
    for view in scene.getTrainCameras():
        render_pkg = render(view, gaussians, pipe, background)
        img = render_pkg["render"]
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        
        h, w, c = img_np.shape
        flat_img = img_np.reshape(-1, 3)
        
        # Mascarar o fundo escuro/preto
        fg_mask = np.any(flat_img >= 0.05, axis=1)
        
        # Mapeamento vetorizado para a cor mais próxima via cKDTree
        if np.any(fg_mask):
            _, indices = color_tree.query(flat_img[fg_mask])
            flat_img[fg_mask] = valid_colors[indices]
        
        img_np = flat_img.reshape(h, w, c)
        
        # Salvar imagem
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"), img_bgr)
    
    # 8. Restaurar parâmetros originais das Gaussianas
    gaussians._xyz.data = orig_data['xyz']
    gaussians._opacity.data = orig_data['opacity']
    gaussians._features_dc.data = orig_data['features_dc']
    gaussians._features_rest.data = orig_data['features_rest']
    gaussians._scaling.data = orig_data['scaling']
    gaussians._rotation.data = orig_data['rotation']

def render_clusters_tab20_uniform_cuda(labels, name, gaussians, scene, pipe, background, args, device, mask_filter=None):
    """
    Renderiza cada cluster com UMA cor uniforme usando o rasterizador CUDA.
    Modifica as features dos objetos (OBJECTS) para terem cores uniformes por cluster.
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    # Gerar cores únicas para cada cluster
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(1, num_clusters)))[:, :3]
    np.random.shuffle(colors)
    
    # Mapear cada label para UMA cor específica
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    if -1 in color_map:
        color_map[-1] = np.array([0.0, 0.0, 0.0])  # Ruído fica preto
    
    # ====== CRIAR FEATURES UNIFORMES ======
    # Cada gaussiana recebe a cor do seu cluster (TODAS as gaussianas do mesmo cluster têm a MESMA cor)
    # Convertendo para o formato esperado pelo CUDA (features)
    gaussian_colors = np.array([color_map[l] for l in labels], dtype=np.float32)
    
    # Guardar dados originais
    orig_data = {
        'xyz': gaussians._xyz.data.clone(),
        'opacity': gaussians._opacity.data.clone(),
        'features_dc': gaussians._features_dc.data.clone(),
        'features_rest': gaussians._features_rest.data.clone(),
        'scaling': gaussians._scaling.data.clone(),
        'rotation': gaussians._rotation.data.clone()
    }
    
    # ====== MODIFICAR FEATURES DIRECTAMENTE ======
    # As features são armazenadas como SH coefficients.
    # Para cores uniformes, precisamos converter RGB para o formato SH.
    # A conversão é: SH_color = (RGB - 0.5) / 0.28209
    colors_tensor = torch.tensor(gaussian_colors, dtype=torch.float32, device=device)
    
    if mask_filter is not None:
        mask = torch.tensor(mask_filter, device=device, dtype=torch.bool)
        gaussians._xyz.data = orig_data['xyz'][mask]
        gaussians._opacity.data = orig_data['opacity'][mask]
        gaussians._scaling.data = orig_data['scaling'][mask]
        gaussians._rotation.data = orig_data['rotation'][mask]
        
        # Atribuir cores uniformes via features_dc
        gaussians._features_dc.data = (colors_tensor[mask].unsqueeze(1) - 0.5) / 0.28209
        # ZERAR features_rest para eliminar variações
        new_rest_shape = [gaussians._xyz.shape[0], orig_data['features_rest'].shape[1], 3]
        gaussians._features_rest.data = torch.zeros(new_rest_shape, device=device)
    else:
        # Atribuir cores uniformes via features_dc
        gaussians._features_dc.data = (colors_tensor.unsqueeze(1) - 0.5) / 0.28209
        # ZERAR features_rest para eliminar variações
        gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)
    
    # Renderizar
    out_dir = os.path.join(args.output, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering {num_clusters} uniform clusters (CUDA) to {out_dir}...")
    
    for view in scene.getTrainCameras():
        # Usar a função render ORIGINAL (já modificamos as features)
        render_pkg = render(view, gaussians, pipe, background)
        img = render_pkg["render"]
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        
        # ====== PÓS-PROCESSAMENTO: FORÇAR CORES PERFEITAMENTE UNIFORMES ======
        # Para cada pixel, mapear para a cor mais próxima do cluster
        for i in range(img_np.shape[0]):
            for j in range(img_np.shape[1]):
                pixel = img_np[i, j]
                # Ignorar fundo preto
                if np.all(pixel < 0.05):
                    continue
                # Encontrar a cor do cluster mais próxima
                distancias = []
                for label, cor in color_map.items():
                    if label == -1:
                        continue
                    dist = np.linalg.norm(pixel - cor)
                    distancias.append((dist, cor))
                
                if distancias:
                    _, cor_mais_proxima = min(distancias, key=lambda x: x[0])
                    img_np[i, j] = cor_mais_proxima
        
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"), img_bgr)
    
    # Restaurar dados originais
    gaussians._xyz.data = orig_data['xyz']
    gaussians._opacity.data = orig_data['opacity']
    gaussians._features_dc.data = orig_data['features_dc']
    gaussians._features_rest.data = orig_data['features_rest']
    gaussians._scaling.data = orig_data['scaling']
    gaussians._rotation.data = orig_data['rotation']
    """
    Renderiza clusters com cor 100% uniforme por pixel, usando o canal de
    'object features' (render_object) em vez do blending alfa de RGB.
    Isso evita halos/gradientes nas bordas dos clusters, permitindo
    avaliação correta com métricas tipo IoU/mIoU.
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    # Dimensão de saída de objetos do modelo (NUM_OBJECTS definido no CUDA/config)
    num_obj_channels = gaussians._objects_dc.shape[-1] if gaussians._objects_dc.dim() > 2 \
        else gaussians._objects_dc.shape[1]

    if num_clusters > num_obj_channels:
        raise ValueError(
            f"Número de clusters ({num_clusters}) excede os canais de objeto "
            f"disponíveis ({num_obj_channels}). Aumente NUM_OBJECTS no CUDA/config "
            f"ou reduza o número de clusters."
        )

    # Mapeia cada label -> índice de canal (0..num_clusters-1)
    label_to_channel = {label: i for i, label in enumerate(unique_labels)}
    NOISE_CHANNEL = label_to_channel.get(-1, None)

    # Cores fixas (tab20) só para visualização, aplicadas DEPOIS do argmax
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(1, num_clusters)))[:, :3]
    np.random.shuffle(colors)
    color_lut = {i: colors[i] for i in range(num_clusters)}
    if NOISE_CHANNEL is not None:
        color_lut[NOISE_CHANNEL] = np.array([0.0, 0.0, 0.0])  # ruído = preto

    # Monta o one-hot por gaussiana (não é mais RGB, é "objeto")
    channel_idx = np.array([label_to_channel[l] for l in labels])
    one_hot = np.zeros((len(labels), num_obj_channels), dtype=np.float32)
    one_hot[np.arange(len(labels)), channel_idx] = 1.0
    one_hot_tensor = torch.tensor(one_hot, dtype=torch.float32, device=device)

    # Backup do estado original
    orig_data = {
        'xyz': gaussians._xyz.data.clone(),
        'opacity': gaussians._opacity.data.clone(),
        'scaling': gaussians._scaling.data.clone(),
        'rotation': gaussians._rotation.data.clone(),
        'objects_dc': gaussians._objects_dc.data.clone(),
    }

    if mask_filter is not None:
        mask = torch.tensor(mask_filter, device=device, dtype=torch.bool)
        gaussians._xyz.data = orig_data['xyz'][mask]
        gaussians._opacity.data = orig_data['opacity'][mask]
        gaussians._scaling.data = orig_data['scaling'][mask]
        gaussians._rotation.data = orig_data['rotation'][mask]
        gaussians._objects_dc.data = one_hot_tensor.unsqueeze(1) \
            if orig_data['objects_dc'].dim() == 3 else one_hot_tensor
    else:
        gaussians._objects_dc.data = one_hot_tensor.unsqueeze(1) \
            if orig_data['objects_dc'].dim() == 3 else one_hot_tensor

    out_dir = os.path.join(args.output, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering {num_clusters} discrete clusters (hard/uniform) to {out_dir}...")

    for view in scene.getTrainCameras():
        render_pkg = render(view, gaussians, pipe, background)
        obj_render = render_pkg["render_object"]          # [OBJECTS, H, W]

        # Label duro por pixel: argmax sobre canais de objeto
        label_map = obj_render.argmax(dim=0).detach().cpu().numpy()  # [H, W]

        # Aplica cor fixa a partir do LUT — sem qualquer mistura alfa
        img_rgb = np.zeros((*label_map.shape, 3), dtype=np.float32)
        for ch, color in color_lut.items():
            img_rgb[label_map == ch] = color

        img_uint8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"), img_bgr)

    # Restaura estado original
    gaussians._xyz.data = orig_data['xyz']
    gaussians._opacity.data = orig_data['opacity']
    gaussians._scaling.data = orig_data['scaling']
    gaussians._rotation.data = orig_data['rotation']
    gaussians._objects_dc.data = orig_data['objects_dc']

def render_clusters_tab20_opacity0(labels, name, gaussians, scene, pipe, background, args, device, mask_filter=None):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(1, num_clusters)))[:, :3]
    np.random.shuffle(colors)
    
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    if -1 in color_map:
        color_map[-1] = np.array([0.0, 0.0, 0.0])  # Ruído/Sem classe fica estritamente PRETO

    gaussian_colors = np.array([color_map[l] for l in labels])
    colors_tensor_filtered = torch.tensor(gaussian_colors, dtype=torch.float32, device=device)

    orig_data = {
        'xyz': gaussians._xyz.data.clone(),
        'opacity': gaussians._opacity.data.clone(),
        'features_dc': gaussians._features_dc.data.clone(),
        'features_rest': gaussians._features_rest.data.clone(),
        'scaling': gaussians._scaling.data.clone(),
        'rotation': gaussians._rotation.data.clone()
    }

    # Criar máscara para gaussianas pretas (label -1) usando numpy (mais eficiente em memória)
    is_black_mask = np.array([l == -1 for l in labels])
    
    # Se há gaussianas pretas, modificar a opacidade diretamente
    if np.any(is_black_mask):
        # Converter máscara para tensor torch no mesmo dispositivo
        black_indices = torch.tensor(np.where(is_black_mask)[0], device=device, dtype=torch.long)
        
        # Modificar apenas as gaussianas pretas in-place (economiza memória)
        if mask_filter is not None:
            mask = torch.tensor(mask_filter, device=device, dtype=torch.bool)
            gaussians._xyz.data = orig_data['xyz'][mask]
            gaussians._opacity.data = orig_data['opacity'][mask]
            gaussians._scaling.data = orig_data['scaling'][mask]
            gaussians._rotation.data = orig_data['rotation'][mask]
            
            # Aplicar opacidade zero para gaussianas pretas no subset filtrado
            black_mask_filtered = torch.tensor(is_black_mask[mask_filter], device=device, dtype=torch.bool)
            if black_mask_filtered.any():
                gaussians._opacity.data[black_mask_filtered] = 0.0
            
            gaussians._features_dc.data = (colors_tensor_filtered[mask].unsqueeze(1) - 0.5) / 0.28209
            new_rest_shape = [gaussians._xyz.shape[0], orig_data['features_rest'].shape[1], 3]
            gaussians._features_rest.data = torch.zeros(new_rest_shape, device=device)
        else:
            # Para todas as gaussianas, modificar in-place
            # Usar indexação direta para modificar apenas as gaussianas pretas
            gaussians._opacity.data[black_indices] = 0.0
            
            # Modificar cores para todas as gaussianas
            gaussians._features_dc.data = (colors_tensor_filtered.unsqueeze(1) - 0.5) / 0.28209
            gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)
    else:
        # Se não há gaussianas pretas, proceder normalmente
        if mask_filter is not None:
            mask = torch.tensor(mask_filter, device=device, dtype=torch.bool)
            gaussians._xyz.data = orig_data['xyz'][mask]
            gaussians._opacity.data = orig_data['opacity'][mask]
            gaussians._scaling.data = orig_data['scaling'][mask]
            gaussians._rotation.data = orig_data['rotation'][mask]
            
            gaussians._features_dc.data = (colors_tensor_filtered[mask].unsqueeze(1) - 0.5) / 0.28209
            new_rest_shape = [gaussians._xyz.shape[0], orig_data['features_rest'].shape[1], 3]
            gaussians._features_rest.data = torch.zeros(new_rest_shape, device=device)
        else:
            gaussians._features_dc.data = (colors_tensor_filtered.unsqueeze(1) - 0.5) / 0.28209
            gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)

    out_dir = os.path.join(args.output, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Rendering {num_clusters} discrete clusters to {out_dir}...")
    print(f"Gaussianas pretas (opacidade 0): {np.sum(is_black_mask)}")

    for view in scene.getTrainCameras():
        render_pkg = render(view, gaussians, pipe, background)
        img = render_pkg["render"]
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{view.image_name}.png"), img_bgr)

    # Restaurar dados originais
    gaussians._xyz.data = orig_data['xyz']
    gaussians._opacity.data = orig_data['opacity']
    gaussians._features_dc.data = orig_data['features_dc']
    gaussians._features_rest.data = orig_data['features_rest']
    gaussians._scaling.data = orig_data['scaling']
    gaussians._rotation.data = orig_data['rotation']
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

def merge_adjacent_clusters_robust(
    xyz, 
    labels, 
    emb, 
    k_spatial=10, 
    centroid_sim_threshold=0.86, # Similaridade entre os CENTRÓIDES dos clusters
    min_touching_ratio=0.05,     # Exige que 5% dos PONTOS ÚNICOS do menor cluster estejam em contato
    max_merged_size_ratio=0.35   # TRAVA: Nenhum cluster pode ter mais de 35% dos pontos da cena
):
    """
    Versão robusta para fusão de clusters adjacentes.
    Evita o vazamento de embeddings da GNN nas bordas e previne o colapso global da cena.
    """
    unique_labels = np.array([l for l in np.unique(labels) if l != -1])
    if len(unique_labels) <= 1:
        return labels

    N_total = len(xyz)
    max_allowed_size = int(N_total * max_merged_size_ratio)

    print(f"\n🔍 Analisando fusão adjacente robusta (Threshold Centróide: {centroid_sim_threshold})...")

    # 1. Pré-calcula os centróides e tamanhos dos clusters inteiros (sem contaminação de borda)
    cluster_centroids = {}
    cluster_sizes = {}
    for l in unique_labels:
        mask = (labels == l)
        c_emb = emb[mask].mean(axis=0)
        c_emb = c_emb / (np.linalg.norm(c_emb) + 1e-8)
        cluster_centroids[l] = c_emb
        cluster_sizes[l] = np.sum(mask)

    # 2. Mapeia a fronteira 3D
    nbrs = NearestNeighbors(n_neighbors=k_spatial, algorithm="auto").fit(xyz)
    _, indices = nbrs.kneighbors(xyz)

    src_idx = np.repeat(np.arange(len(labels)), k_spatial)
    dst_idx = indices.flatten()

    src_l = labels[src_idx]
    dst_l = labels[dst_idx]

    valid_mask = (src_l != dst_l) & (src_l != -1) & (dst_l != -1)
    border_src = src_idx[valid_mask]
    border_dst = dst_idx[valid_mask]
    border_src_l = src_l[valid_mask]
    border_dst_l = dst_l[valid_mask]

    # Agrupa PONTOS ÚNICOS de interface (usando set para não inflacionar contagem)
    pair_touching_pts = {}
    for i in range(len(border_src_l)):
        l1, l2 = border_src_l[i], border_dst_l[i]
        pair = tuple(sorted((l1, l2)))
        
        if pair not in pair_touching_pts:
            pair_touching_pts[pair] = {"src": set(), "dst": set()}
            
        pair_touching_pts[pair]["src"].add(border_src[i])
        pair_touching_pts[pair]["dst"].add(border_dst[i])

    # 3. Seleciona candidatos válidos para fusão
    candidates = []
    for (l1, l2), pts_dict in pair_touching_pts.items():
        # Conta a quantidade de PONTOS ÚNICOS reais em contato
        unique_touching_count = min(len(pts_dict["src"]), len(pts_dict["dst"]))
        min_cluster_size = min(cluster_sizes[l1], cluster_sizes[l2])
        
        touching_ratio = unique_touching_count / min_cluster_size
        
        if touching_ratio < min_touching_ratio:
            continue  # Toque irrelevante / pontual

        # Similaridade entre as MÉDIAS GLOBAIS dos clusters
        sim = float(np.dot(cluster_centroids[l1], cluster_centroids[l2]))

        if sim >= centroid_sim_threshold:
            candidates.append((sim, l1, l2))

    # Ordena da MAIOR similaridade para a MENOR (Merge Guloso)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 4. Executa fusões controladas com trava de tamanho
    parent = {l: l for l in unique_labels}
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    current_sizes = cluster_sizes.copy()
    merged_count = 0

    for sim, l1, l2 in candidates:
        root1, root2 = find(l1), find(l2)
        if root1 != root2:
            new_size = current_sizes[root1] + current_sizes[root2]
            
            # Trava de tamanho: Não deixa formar um "super cluster" devorador
            if new_size <= max_allowed_size:
                parent[root1] = root2
                current_sizes[root2] = new_size
                merged_count += 1

    # 5. Atualiza o vetor de labels
    label_map = {l: find(l) for l in unique_labels}
    label_map[-1] = -1

    merged_labels = np.array([label_map[l] for l in labels])
    n_final = len(np.unique([l for l in merged_labels if l != -1]))

    print(f"  └─ Clusters reduzidos de {len(unique_labels)} para {n_final} ({merged_count} fusões autorizadas).")
    return merged_labels

def clean_micro_clusters_cluster_level(
    xyz, 
    labels, 
    emb=None, 
    min_cluster_size=50, 
    use_embeddings=True
):
    """
    Remove micro-clusters e ruídos reatribuindo O CLUSTER INTEIRO de uma vez
    ao cluster válido mais próximo.
    
    Args:
        xyz: Posições (N, 3) das gaussianas.
        labels: Vector de labels dos clusters gerados (HDBSCAN/KNN).
        emb: Embeddings (N, D) normalizados da GNN (opcional, mas recomendado).
        min_cluster_size: Clusters com contagem abaixo deste valor são absorvidos.
        use_embeddings: Se True e 'emb' for fornecido, mede distâncias no espaço latente.
    """
    cleaned_labels = labels.copy()
    
    unique_labels, counts = np.unique(cleaned_labels, return_counts=True)
    cluster_counts = dict(zip(unique_labels, counts))
    
    # 1. Separa clusters válidos (grandes) de pequenos/ruído
    valid_labels = [l for l, count in cluster_counts.items() if l != -1 and count >= min_cluster_size]
    small_labels = [l for l in unique_labels if l not in valid_labels]
    
    if not valid_labels or not small_labels:
        print("  └─ Nenhum micro-cluster para absorver ou nenhum cluster válido encontrado.")
        return cleaned_labels

    print(f"\n🧹 Absorvendo {len(small_labels)} micro-clusters/ruídos (nível de cluster completo)...")

    # Espaço de representação: Embeddings L2-normalized ou Posições XYZ
    feat_space = emb if (use_embeddings and emb is not None) else xyz

    # 2. Calcula os centróides de cada cluster VÁLIDO
    valid_centroids = np.array([feat_space[cleaned_labels == l].mean(axis=0) for l in valid_labels])
    
    # Indexador kNN para busca do centróide válido mais próximo
    nn_valid = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(valid_centroids)

    reassigned_count = 0
    
    # 3. Processa cada micro-cluster/ruído de uma só vez
    for s_label in small_labels:
        mask_s = (cleaned_labels == s_label)
        if not np.any(mask_s):
            continue
        
        # Centróide do micro-cluster atual
        s_centroid = feat_space[mask_s].mean(axis=0, keepdims=True)
        
        # Encontra o cluster válido mais próximo
        _, idx = nn_valid.kneighbors(s_centroid)
        best_valid_label = valid_labels[idx[0][0]]
        
        # Reatribuição GLOBAL das gaussianas do cluster
        cleaned_labels[mask_s] = best_valid_label
        reassigned_count += np.sum(mask_s)

    print(f"  └─ {reassigned_count:,} gaussianas (núcleos completos) reatribuídas com sucesso.")
    return cleaned_labels

def convert_rgb_to_normalized_cielab(rgb_array, device):
    """
    Converte um array RGB [0.0, 1.0] para CIELAB normalizado em [0.0, 1.0].
    """
    # Garante que os valores de RGB estejam no intervalo [0, 1]
    rgb_clamped = np.clip(rgb_array, 0.0, 1.0)
    
    # rgb2lab espera [0, 1] e retorna:
    # L: [0, 100]
    # a: [-128, 127]
    # b: [-128, 127]
    lab = rgb2lab(rgb_clamped)
    
    # Normalização dos canais para ficarem na mesma escala [0, 1]
    lab_norm = np.zeros_like(lab)
    lab_norm[:, 0] = lab[:, 0] / 100.0            # L (Luminância)
    lab_norm[:, 1] = (lab[:, 1] + 128.0) / 255.0   # a (Verde <-> Vermelho)
    lab_norm[:, 2] = (lab[:, 2] + 128.0) / 255.0   # b (Azul <-> Amarelo)
    
    return torch.tensor(lab_norm, dtype=torch.float, device=device)

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

def save_evaluation_report(metrics, output_dir, experiment_name="evaluation"):
    """
    Salva um relatório completo da avaliação em CSV e TXT
    
    Args:
        metrics: Dict com todas as métricas
        output_dir: Diretório para salvar
        experiment_name: Nome do experimento
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar em CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{experiment_name}_metrics_{timestamp}.csv")
    
    # Converter para DataFrame
    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)
    
    # Salvar em TXT (mais legível)
    txt_path = os.path.join(output_dir, f"{experiment_name}_report_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"RELATÓRIO DE AVALIAÇÃO - {experiment_name}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key:30s}: {value:.4f}\n")
            else:
                f.write(f"{key:30s}: {value}\n")
    
    print(f"\n✅ Relatório salvo em:\n  CSV: {csv_path}\n  TXT: {txt_path}")
    
    return csv_path, txt_path

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


def save_experiment_results(metrics, loss_history, output_dir, model_name="GAT", training_time=None, graph_build_time=None, hdbscan_time=None, max_vram_gb=None):
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
        max_vram_gb: Pico de VRAM alocada (opcional)
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
            'Pico de VRAM (GB)': round(max_vram_gb, 2) if max_vram_gb is not None else 'N/A',
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
def save_pointcloud_with_metadata(xyz, labels, output_path, metadata=None):
    """
    Salva a nuvem de pontos 3D com cores discretas para cada cluster e grava os metadados.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Fallback se Open3D não estiver instalado no ambiente de salvamento
    if not OPEN3D_AVAILABLE:
        npz_path = output_path.replace('.ply', '.npz')
        print(f"⚠️ Open3D não disponível. Salvando nuvem em NumPy: {npz_path}")
        np.savez(npz_path, points=xyz, labels=labels, metadata=metadata)
        return

    # Mapeamento discreto de cores categóricas para os clusters
    unique_labels = np.unique(labels)
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    cmap = plt.get_cmap("tab20")

    for label in unique_labels:
        mask = (labels == label)
        if label == -1:
            colors[mask] = [0.12, 0.12, 0.12]  # Ruído em cinza escuro
        else:
            # Seleciona cor categórica determinística
            colors[mask] = cmap(int(label) % 20)[:3]

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

def main():
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--masks_path", required=True)
    parser.add_argument("--deva_json", required=True)
    parser.add_argument("--output", default="exp_res")
    parser.add_argument("--visualize_3d", action="store_true", default=True)
    parser.add_argument("--save_pointclouds", action="store_true", default=True)
    parser.add_argument("--render_images", action="store_true", default=True)

    parser.add_argument("--model_type_choice", default="gat", choices=["gat", "gcn"])
    parser.add_argument("--gcn_depth", default="v2", choices=["v2", "deep"], 
                        help="v2: 2 camadas GCN | deep: 4 camadas com skip connections")
    parser.add_argument("--gat_heads", default=4, type=int)
    parser.add_argument("--opacity_threshold", default=0.05, type=float)
    parser.add_argument("--max_scale_threshold", default=10, type=float)
    parser.add_argument("--min_cluster_size", default=250, type=int)
    parser.add_argument("--target_gaussians", default=230000, type=int)

    parser.add_argument("--render_full_pointcloud", action="store_true", default=True)
    parser.add_argument("--knn_reassign_k", default=25, type=int)
    parser.add_argument("--experiment_name", default=None, type=str,
                        help="Nome do experimento (usado para organizar os resultados)")

    args = get_combined_args(parser)
    
    # ==========================================================
    # CONFIGURAÇÃO DA ESTRUTURA DE DIRETÓRIOS
    # ==========================================================
    if args.experiment_name is None:
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

    labels_deva = build_labels_with_deva_json_v2_black(
        scene,
        xyz,
        args.masks_path,
        args.deva_json,
        min_score=0.9,
    )

    first_candidate_len = gaussians._xyz.shape[0] * 0.8
    if 280000 < first_candidate_len:
        first_candidate_len = 280000

    mask_labels_validos = (labels_deva != -1)

# 2. Extrai a opacidade e a escala APENAS das gaussianas rotuladas
    opacity_valid = opacity[mask_labels_validos]
    max_scaling_valid = max_scaling[mask_labels_validos]

    # 3. Calcula os thresholds direcionados exatamente para as gaussianas do DEVA
    dyn_op_thresh, dyn_sc_thresh, final_count = compute_dynamic_filters(
        opacity_valid,
        max_scaling_valid,
        target_count=250000
    )

    args.opacity_threshold = dyn_op_thresh
    args.max_scale_threshold = dyn_sc_thresh



    # 3. Define um corte (ex: mantém apenas os 85% mais próximos, descarta 15% do fundo)

# ==========================================================
    # FILTRO POR BOUNDING BOX DO CLOUDCOMPARE (AABB)
    # ==========================================================
    # Valores extraídos diretamente do "Edit clipping box" do CloudCompare
    box_center = np.array([0.87771511, 1.94553256, 0.34934092])
    box_width  = np.array([7.83855629, 6.98039675, 7.24066257])

    # Calcula as metades das larguras para encontrar os limites min/max
    half_width = box_width / 2.0

    # Verifica se os pontos (xyz) estão dentro de [-half_width, +half_width] em relação ao centro
    mask_box = np.all(np.abs(xyz - box_center) <= half_width, axis=1)

    # 4. Gera a máscara final combinando os filtros (substituindo mask_depth por mask_box)
    mask_geom = (opacity > args.opacity_threshold) & (max_scaling < args.max_scale_threshold)
    mask_filter = mask_geom & mask_labels_validos & mask_box

    labels = labels_deva[mask_filter]
    xyz = xyz[mask_filter]
    rgb = rgb[mask_filter]
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

    scaler_rgb = StandardScaler()
    rgb_norm = scaler_rgb.fit_transform(rgb) * 3.0

    # ==========================================================
    # CONSTRUÇÃO DO GRAFO
    # ==========================================================
    t_graph_start = time.time()
    
    x_input = np.concatenate([xyz_norm, rgb_norm], axis=1)
    x = torch.tensor(x_input, dtype=torch.float)

    print("\n🔗 Construindo grafo geométrico...")
    N_NEIGHBORS = 9
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="auto").fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)

    neighbor_idx = indices[:, 1:]
    neighbor_dist = distances[:, 1:]

    sigma = np.median(neighbor_dist)
    print(f"  Sigma geométrico: {sigma:.4f}")

    rgb_i = rgb[:, None, :]
    rgb_j = rgb[neighbor_idx]
    color_dist = np.linalg.norm(rgb_i - rgb_j, axis=-1)
    color_sigma = np.median(color_dist)
    color_weight_all = np.exp(-(color_dist ** 2) / (2 * color_sigma ** 2 + 1e-8))
    spatial_weight_all = np.exp(-(neighbor_dist ** 2) / (2 * sigma ** 2 + 1e-8))

    distance_threshold = neighbor_dist.mean() + neighbor_dist.std()
    color_dist_threshold = np.percentile(color_dist, 75)
    print(f"  Distance threshold: {distance_threshold:.4f} | Color dist threshold: {color_dist_threshold:.4f}")

    edge_mask = (neighbor_dist <= distance_threshold) & (color_dist <= color_dist_threshold)
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
    print(f"⏱️ Tempo de construção do grafo: {graph_build_time:.2f}s")

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
    
    criterion = ColorAwareContrastiveLossV2(
        temperature=0.05,
        pos_margin=0.2,
        neg_margin=0.1,
        color_weight=0,
        color_sigma=0.25
    )
    
    labels_t = torch.tensor(labels, device=device)
    colors_t = torch.tensor(rgb_norm, dtype=torch.float, device=device)
    
    # ==========================================================
    # LOOP DE TREINAMENTO
    # ==========================================================
    model_name = f"{args.model_type_choice.upper()}-{args.gcn_depth.upper()}" if args.model_type_choice == "gcn" else args.model_type_choice.upper()
    print(f"\n🎓 Treinando {model_name} com Early Stopping...")
    t_train_start = time.time()

    early_stopping = EarlyStopping(patience=20, min_delta=0.0001, verbose=True)
    loss_history = []
    menor_loss = float('inf')

    for epoch in range(250):
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
            print(f"Epoch {epoch:4d} | Loss: {current_loss:.4f} | Best: {menor_loss:.4f}")

        if early_stopping(current_loss, model):
            break

    early_stopping.restore_best_model(model)
    
    t_train_end = time.time()
    training_time = t_train_end - t_train_start
    print(f"⏱️ Tempo de treinamento do modelo: {training_time:.2f}s")

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

    # ==========================================================
    # HDBSCAN CLUSTERING
    # ==========================================================
    print("\n📊 Clusterizando os novos embeddings com HDBSCAN...")
    t_hdbscan_start = time.time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=1,
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
    # MEDIÇÃO DE VRAM PICO
    # ==========================================================
    if torch.cuda.is_available():
        max_vram_bytes = torch.cuda.max_memory_allocated(device)
        max_vram_gb = max_vram_bytes / (1024 ** 3)
    else:
        max_vram_gb = 0.0

    print(f"💾 Pico de VRAM alocada: {max_vram_gb:.2f} GB")

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
        'Max_VRAM_GB': max_vram_gb,
    }
    
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
        max_vram_gb=max_vram_gb
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
        'max_vram_gb': max_vram_gb
    }
    save_pointcloud_with_metadata(xyz, cluster_labels, pointcloud_path, metadata)
    
    # ==========================================================
    # RENDERIZAÇÃO 2D
    # ==========================================================
    if args.render_images:
        print("\n" + "="*50 + "\nRENDER 2D\n" + "="*50)
        orig_dc = gaussians._features_dc.clone()
        pipe = pipeline_params.extract(args)
        background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
        
        n_total_gaussians = gaussians._xyz.shape[0]

        varvar = False
        if varvar:#args.render_full_pointcloud:
            # 🔧 PROPAGAR labels para TODAS as Gaussianas
            print("  🔄 Propagando labels para todas as Gaussianas...")
            full_cluster_labels = propagate_labels_to_full_set(
                xyz, cluster_labels, xyz_full_original, k=3
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
        
        # Renderizar
        render_clusters(
            render_labels, 
            "images", 
            gaussians, 
            scene, 
            pipe, 
            background, 
            args, 
            device, 
            render_mask
        )
        import shutil
        temp_dir = os.path.join(model_dir, "images")
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                src = os.path.join(temp_dir, file)
                dst = os.path.join(images_dir, file)
                shutil.move(src, dst)
            os.rmdir(temp_dir)
        
        gaussians._features_dc.data = orig_dc
        print(f"🖼️ Imagens renderizadas salvas em: {images_dir}")
    
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
    print(f"💾 VRAM Pico: {max_vram_gb:.2f} GB")
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