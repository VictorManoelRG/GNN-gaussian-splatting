import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mode
from scipy.stats import mode as scipy_mode

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
class GaussianGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=64, out_dim=32, dropout=0.1):
        super().__init__()
        # add_self_loops=True garante a preservação do próprio nó
        self.gcn1 = GCNConv(in_channels, hidden_dim, add_self_loops=True)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        
        self.gcn2 = GCNConv(hidden_dim, hidden_dim * 2, add_self_loops=True)
        self.norm2 = torch.nn.LayerNorm(hidden_dim * 2)

        # Projeção de atalho (Residual) para ajustar dimensões (in_channels -> hidden_dim * 2)
        self.proj_res = torch.nn.Linear(in_channels, hidden_dim * 2)

        self.linear = torch.nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # Mapeamento do caminho residual direto
        identity = self.proj_res(x)

        # Bloco 1
        h = self.gcn1(x, edge_index, edge_weight=edge_weight)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.dropout(h)

        # Bloco 2
        h = self.gcn2(h, edge_index, edge_weight=edge_weight)
        h = self.norm2(h)

        # Conexão Residual Crítica (impede o colapso de labels)
        h = h + identity
        h = F.elu(h)
        h = self.dropout(h)

        # Projeção de Saída
        x_out = self.linear(h)
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
        return 0.01, 1.0, n_total
    
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
    Projeta os pontos 3D garantindo compatibilidade de convenção de matrizes do 3DGS.
    """
    # Garantir matrizes no formato float32 do numpy
    w2c = view.world_view_transform.detach().cpu().numpy().T  # Atenção ao .T dependendo do framework
    P = view.full_proj_transform.detach().cpu().numpy().T     # Idem aqui

    p_hom = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)])
    
    # Transformação para o espaço da câmera
    p_cam = p_hom @ w2c
    z_cam = p_cam[:, 2]

    # Projeção NDC (-1 a 1)
    proj = p_hom @ P
    # Evita divisão por zero/valores atrás da câmera
    w = proj[:, 3:4]
    w_safe = np.where(np.abs(w) < 1e-6, 1e-6, w)
    proj_ndc = proj[:, :3] / w_safe

    W = out_width if out_width is not None else view.image_width
    H = out_height if out_height is not None else view.image_height

    # Pixel coords
    u = ((proj_ndc[:, 0] * 0.5 + 0.5) * W).astype(np.int32)
    v = ((proj_ndc[:, 1] * 0.5 + 0.5) * H).astype(np.int32)

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

def build_labels_with_deva_json(
    scene,
    xyz,
    masks_path,
    deva_json_path,
    min_score=0.3,
    min_area=20,
    max_area=10000,
    depth_tolerance=0.02,  # Mantido para compatibilidade, mas não será usado
    color_tolerance=0.25,
    use_color_validation=False,
    propagate_to_neighbors=True,
    k_propagate=1,
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

            # Filtro por limiares e área máxima (descarta fundo gigante)
            if score >= min_score and min_area <= area <= max_area:
                total_segments_kept += 1
                deva_meta[fname][sid] = {
                    "score": score,
                    "area": area,
                    "category_id": seg.get("category_id"),
                }

    print(f"  └─ Segmentos no JSON: {total_segments_raw} total | {total_segments_kept} mantidos (score>={min_score}, {min_area}<=area<={max_area})")

    # -------------------------------------------------------------------------
    # 2. Estruturas para Acumulação Ponderada e Média
    # -------------------------------------------------------------------------
    train_cameras = scene.getTrainCameras()
    total_cameras = len(train_cameras)

    # gi -> {sid: acumulado_dos_pesos}
    votes_weight = [defaultdict(float) for _ in range(N)]
    # gi -> {sid: contagem_de_observacoes}
    votes_count = [defaultdict(int) for _ in range(N)]

    processed = 0
    frames_com_mascara_lida = 0

    print(f"🔍 Montando labels (votação normalizada e por distância) em {masks_path}...")

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
        if mask.ndim == 3:
            mask = mask[..., 0]

        frames_com_mascara_lida += 1
        processed += 1
        if processed % 50 == 0:
            print(f"  Processando frame {processed}/{total_cameras}...")

        H, W = mask.shape

        # Projeta TODOS os pontos
        u, v, z_cam = project_points_with_depth(xyz, view, out_width=W, out_height=H)

        valid = (z_cam > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue

        # ════════════════════════════════════════════════════════════════════
        # 🔥 MODIFICAÇÃO PRINCIPAL: REMOVER O Z-BUFFER
        # Agora usamos TODOS os pontos projetados, sem filtrar por profundidade
        # ════════════════════════════════════════════════════════════════════
        
        # Obtém os IDs das máscaras para TODOS os pontos projetados
        sids = mask[v[idx], u[idx]].astype(np.int64)

        # Matriz de posições da câmera para ponderação por distância 3D
        cam_center = view.camera_center.detach().cpu().numpy() if hasattr(view, 'camera_center') else None

        for gi, sid in zip(idx, sids):
            if sid in frame_seg_info:
                score = frame_seg_info[sid]["score"]
                area = frame_seg_info[sid]["area"]

                # 1. Fator de distância: Visão de perto ganha mais peso
                if cam_center is not None:
                    dist = np.linalg.norm(xyz[gi] - cam_center) + 1e-5
                    dist_weight = 1.0 / dist
                else:
                    dist_weight = 1.0

                # 2. Fator de área: Ponderação inversamente proporcional ao tamanho
                area_weight = 1.0 / np.log1p(area)

                # Peso combinado para este frame
                weight = score * dist_weight * area_weight

                votes_weight[gi][sid] += weight
                votes_count[gi][sid] += 1

    # -------------------------------------------------------------------------
    # 3. Atribuição por Pontuação Média (Média Ponderada por Frame)
    # -------------------------------------------------------------------------
    labels = np.full(N, -1, dtype=np.int64)

    for i in range(N):
        if votes_weight[i]:
            # Calcula a média ponderada para cada SID em que esta gaussiana participou
            avg_scores = {
                sid: votes_weight[i][sid] / votes_count[i][sid]
                for sid in votes_weight[i]
            }
            # O label vencedor é o que teve maior score médio
            labels[i] = max(avg_scores, key=avg_scores.get)

    assigned = np.sum(labels != -1)
    print(f"\n✅ Concluído:")
    print(f"  • Frames com máscara lida: {frames_com_mascara_lida}/{total_cameras}")
    print(f"  • Gaussianas rotuladas (votação média/normalizada): {assigned}/{N} ({100 * assigned / N:.1f}%)")

    # --- suaviza ruído sal-e-pimenta antes de propagar pros -1 ---
    print(f"\n🧹 Suavizando labels espacialmente...")
    #labels = smooth_labels_spatially(xyz, labels, k=24, min_agreement=0.7, n_passes=2)
    
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


# [CORRIGIDO] Garante consistência tridimensional estrita de tamanhos de tensores
def render_clusters(labels, name, gaussians, scene, pipe, background, args, device, mask_filter=None):
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

# ==========================================================
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

def clean_micro_clusters_spatial_voting(
    xyz, 
    labels, 
    k_neighbors=20, 
    min_cluster_size=100, 
    iterations=2
):
    """
    Filtra pequenas manchas/micro-clusters isolados no espaço 3D, 
    fazendo com que o label dominante da vizinhança absorva as partes menores.
    
    Args:
        xyz: Posições (N, 3) das gaussianas filtradas.
        labels: Array de labels gerados pelo HDBSCAN/KNN.
        k_neighbors: Número de vizinhos 3D considerados para definir a maioria local.
        min_cluster_size: Clusters com total de gaussianas abaixo deste tamanho são absorvidos.
        iterations: Quantas passadas de suavização espacial aplicar.
    """
    print(f"\n🧹 Limpando micro-clusters (< {min_cluster_size} gaussianas) via Votação Espacial 3D...")
    cleaned_labels = labels.copy()
    
    # 1. Identifica os clusters que são "pequenos demais" para existirem sozinhos
    unique_labels, counts = np.unique(cleaned_labels, return_counts=True)
    small_clusters = set(unique_labels[counts < min_cluster_size])
    
    # O label -1 (ruído) também deve ser absorvido pela vizinhança
    small_clusters.add(-1)
    
    if not small_clusters:
        print("  └─ Nenhum micro-cluster para limpar.")
        return cleaned_labels

    # 2. Constrói a busca por vizinhos espaciais 3D
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto").fit(xyz)
    _, indices = nbrs.kneighbors(xyz)
    
    # 3. Iterações de Votação por Maioria Local
    for it in range(iterations):
        changed_count = 0
        for i in range(len(cleaned_labels)):
            current_label = cleaned_labels[i]
            
            # Se a gaussiana pertence a um micro-cluster ou é ruído (-1)
            if current_label in small_clusters:
                # Pega os labels dos k vizinhos 3D (excluindo a própria gaussiana)
                neighbor_indices = indices[i, 1:]
                neighbor_labels = cleaned_labels[neighbor_indices]
                
                # Filtra os vizinhos que também pertencem a micro-clusters para evitar contaminação
                valid_neighbor_labels = [l for l in neighbor_labels if l not in small_clusters]
                
                if valid_neighbor_labels:
                    # O label mais frequente na vizinhança "come" a gaussiana
                    most_frequent_label = mode(valid_neighbor_labels, keepdims=False).mode
                    cleaned_labels[i] = most_frequent_label
                    changed_count += 1
                    
        print(f"  └─ Passada {it + 1}/{iterations}: {changed_count:,} gaussianas reatribuídas ao label majoritário.")
        
        # Atualiza a contagem dos small_clusters para a próxima iteração
        u_lab, c_lab = np.unique(cleaned_labels, return_counts=True)
        small_clusters = set(u_lab[c_lab < min_cluster_size])
        small_clusters.add(-1)

    return cleaned_labels

def main():
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--masks_path", required=True)
    parser.add_argument("--deva_json", required=True)
    parser.add_argument("--output", default="output_seg")
    parser.add_argument("--visualize_3d", action="store_false")
    parser.add_argument("--save_pointclouds", action="store_false")

    # consolidado num único argumento (antes havia --model_type e
    # --model_type_choice conflitando; --model_type tinha choices=["gat"]
    # e por isso o branch "gcn" no código nunca era alcançado)
    parser.add_argument("--model_type_choice", default="gcn", choices=["gat", "gcn"])
    parser.add_argument("--gat_heads", default=2, type=int)
    parser.add_argument("--opacity_threshold", default=0.05, type=float)
    parser.add_argument("--max_scale_threshold", default=0.7, type=float, help="Filtro de escala contra elipsoides gigantes")
    parser.add_argument("--min_cluster_size", default=25, type=int)
    parser.add_argument("--target_gaussians", default=230000, type=int, help="Número alvo de gaussianas após filtragem")

    parser.add_argument("--render_full_pointcloud", action="store_true", default=True,
                         help="Propaga os labels para TODAS as gaussianas antes de renderizar (elimina buracos brancos)")
    parser.add_argument("--knn_reassign_k", default=25, type=int,
                         help="Número de vizinhos K nos embeddings para reatribuição rápida de ruído")

    args = get_combined_args(parser)
    os.makedirs(args.output, exist_ok=True)
    safe_state(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD GAUSSIANS
    gaussians = GaussianModel(model_params.extract(args).sh_degree)
    scene = Scene(model_params.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)

    xyz_full_original = gaussians._xyz.detach().cpu().numpy()
    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().squeeze(1).numpy()

    # FILTRAGEM GEOMÉTRICA (OPACIDADE + ESCALA)
    opacity = torch.sigmoid(gaussians._opacity).detach().cpu().numpy().squeeze()
    scaling = torch.exp(gaussians._scaling).detach().cpu().numpy()
    max_scaling = np.max(scaling, axis=1)

    dyn_op_thresh, dyn_sc_thresh, final_count = compute_dynamic_filters(
        opacity,
        max_scaling,
        target_count=180000
    )

    args.opacity_threshold = dyn_op_thresh
    args.max_scale_threshold = dyn_sc_thresh

    mask_filter = (opacity > args.opacity_threshold) & (max_scaling < args.max_scale_threshold)
    #mask_filter = None
    xyz = xyz[mask_filter]
    rgb = rgb[mask_filter]
    opacity_filtered = opacity[mask_filter]

    print(f"\n⚙️ Geometria Filtrada: {len(xyz):,} de {len(opacity):,} Gaussianas restantes.")

    labels = build_labels_with_deva_json(
        scene,
        xyz,
        args.masks_path,
        args.deva_json,
        min_score=0.7,
    )

    pipe = pipeline_params.extract(args)
    background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
    render_clusters(labels, "debug_labels_2d", gaussians, scene, pipe, background, args, device, mask_filter)

    unique, counts = np.unique(labels, return_counts=True)
    print("\n========== LABELS GERADOS ==========")
    print("Número de labels:", len(unique))
    for label, count in zip(unique[:30], counts[:30]):
        print(f"Label {label}: {count:,} Gaussianas")

    # ==========================================================
    # FEATURES
    # ==========================================================
    scaler = StandardScaler()
    x_input = np.concatenate([
        scaler.fit_transform(xyz) * 1,
        scaler.fit_transform(rgb) * 1.5,
    ], axis=1)
    x = torch.tensor(x_input, dtype=torch.float)

    # ==========================================================
    # CONSTRUÇÃO DO GRAFO GEOMÉTRICO (vetorizado)
    # ==========================================================
    print("\n🔗 Construindo grafo geométrico...")
    N_NEIGHBORS = 6
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="auto").fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)

    # remove o próprio ponto (primeiro vizinho = ele mesmo)
    neighbor_idx = indices[:, 1:]              # (N, k)
    neighbor_dist = distances[:, 1:]           # (N, k)

    sigma = np.median(neighbor_dist)
    print(f"  Sigma geométrico: {sigma:.4f}")

    # distância de cor euclidiana (não cosine — cosine trata sombra/luz
    # do mesmo objeto como "cores idênticas", o que gerava ruído nos labels)
    rgb_i = rgb[:, None, :]                    # (N, 1, 3)
    rgb_j = rgb[neighbor_idx]                  # (N, k, 3)
    color_dist = np.linalg.norm(rgb_i - rgb_j, axis=-1)   # (N, k)
    color_sigma = np.median(color_dist)
    color_weight_all = np.exp(-(color_dist ** 2) / (2 * color_sigma ** 2 + 1e-8))

    spatial_weight_all = np.exp(-(neighbor_dist ** 2) / (2 * sigma ** 2 + 1e-8))

    # thresholds adaptativos — calculados a partir da distribuição real
    # dos dados e agora efetivamente usados no filtro (antes eram
    # calculados e descartados em favor de números fixos)
    distance_threshold = neighbor_dist.mean() + neighbor_dist.std()
    color_dist_threshold = np.percentile(color_dist, 75)  # mantém os 75% mais parecidos
    print(f"  Distance threshold: {distance_threshold:.4f} | Color dist threshold: {color_dist_threshold:.4f}")

    edge_mask = (neighbor_dist <= distance_threshold) & (color_dist <= color_dist_threshold)
    src, k_idx = np.where(edge_mask)
    dst = neighbor_idx[src, k_idx]

    weight = 0.6 * spatial_weight_all[src, k_idx] + 1.2 * color_weight_all[src, k_idx]

    # arestas bidirecionais
    edges_np = np.concatenate([
        np.stack([src, dst], axis=1),
        np.stack([dst, src], axis=1),
    ], axis=0)
    weights_np = np.concatenate([weight, weight], axis=0)

    # dedup: kNN não é simétrico, então pares mutuamente vizinhos podem
    # gerar a mesma aresta duas vezes (uma de cada direção da iteração)
    edges_np, dedup_idx = np.unique(edges_np, axis=0, return_index=True)
    weights_np = weights_np[dedup_idx]

    print(f"  Arestas no grafo: {len(edges_np):,}")

    edge_index = torch.tensor(edges_np.T, dtype=torch.long)
    edge_weights_t = torch.tensor(weights_np, dtype=torch.float, device=device)

    # ==========================================================
    # MODELO (GAT ou GCN)
    # ==========================================================
    in_dim = x.shape[1]

    if args.model_type_choice == "gcn":
        # Normalização Min-Max exclusiva para a GCN
        w_min, w_max = weights_np.min(), weights_np.max()
        weights_norm = (weights_np - w_min) / (w_max - w_min + 1e-8)
        edge_weights_t = torch.tensor(weights_norm, dtype=torch.float, device=device)

        model = GaussianGCN(in_channels=in_dim, out_dim=32).to(device)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights_t).to(device)

    else:  # gat (Sua configuração original)
        edge_weights_t = torch.tensor(weights_np, dtype=torch.float, device=device)

        model = GaussianGAT(in_channels=in_dim, heads=args.gat_heads).to(device)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights_t).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss(
        temperature=0.10,
        pos_margin=0.6,
        neg_margin=0.25
    )
    labels_t = torch.tensor(labels, device=device)

    print(f"\n🎓 Treinando {args.model_type_choice.upper()} com Early Stopping...")

    early_stopping = EarlyStopping(patience=20, min_delta=0.0001, verbose=True)
    loss_history = []
    menor_loss = float('inf')

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        # A GCN consome edge_weight, a GAT roda sem ele no forward (como na sua original)
        if args.model_type_choice == "gcn":
            embeddings = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            embeddings = model(data.x, data.edge_index)

        z = F.normalize(embeddings, dim=1)

        loss = criterion(
            z,
            labels_t,
            data.edge_index,
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
        
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    # --- HDBSCAN COM PARÂMETROS OTIMIZADOS ---
    print("\n📊 Clusterizando os novos embeddings com HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=25,
        prediction_data=False  # Alterado para False para maximizar velocidade
    ).fit(emb)
    cluster_labels = clusterer.labels_

    # [NOVO] Execução do pipeline rápido de reatribuição por KNN vetorizado
    cluster_labels = reassign_noise_knn_vectorized(
        emb, 
        cluster_labels, 
        k=args.knn_reassign_k, 
        min_agreement=0.5
    )

    #cluster_labels = clean_micro_clusters_spatial_voting(
        #xyz=xyz, 
        #labels=cluster_labels, 
        #k_neighbors=25,          # Considera as 25 gaussianas 3D ao redor
        #min_cluster_size=30,    # Qualquer segmento menor que 200 gaussianas é "comido" pelo objeto principal
        #iterations=2             # 2 passadas garantem uma superfície contínua e limpa
    #)

    # Montagem final do vetor de cores adequado ao renderizador
    if args.render_full_pointcloud:
        full_cluster_labels = propagate_labels_to_full_set(
            xyz, cluster_labels, xyz_full_original, k=1
        )
        render_labels = full_cluster_labels
        render_xyz = xyz_full_original
    else:
        render_labels = np.full(len(xyz_full_original), -1)
        render_labels[mask_filter] = cluster_labels
        render_xyz = xyz_full_original

    # VISUALIZAÇÃO 3D
    if args.visualize_3d and OPEN3D_AVAILABLE:
        pointcloud_dir = os.path.join(args.output, "pointclouds") if args.save_pointclouds else None
        visualize_clusters_open3d(render_xyz, render_labels, "GNN + HDBSCAN + Rápido KNN")
        
        if args.save_pointclouds and pointcloud_dir:
            gat_path = os.path.join(pointcloud_dir, "gat_clusters.ply")
            visualize_clusters_open3d(render_xyz, render_labels, "GAT Clusters", save_path=gat_path)
            create_viewer_script(pointcloud_dir, args.output)

    # RENDER 2D
    print("\n" + "="*50 + "\nRENDER 2D\n" + "="*50)
    orig_dc = gaussians._features_dc.clone()
    pipe = pipeline_params.extract(args)
    background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)

    render_clusters(render_labels, args.model_type_choice, gaussians, scene, pipe, background, args, device, None)

    gaussians._features_dc.data = orig_dc
    print("\n✅ Concluído! Pipeline executado de forma otimizada.")

if __name__ == "__main__":
    main()