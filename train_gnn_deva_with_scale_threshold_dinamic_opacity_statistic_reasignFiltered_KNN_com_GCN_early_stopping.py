import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode  # [ADICIONADO] Para o voto majoritário rápido
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
        # Camada 1: Projeta de in_channels para hidden_dim
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        
        # Camada 2: Processa dentro do espaço oculto
        self.gcn2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * 2)
        
        # Projeção final para os embeddings do contraste/HDBSCAN
        self.linear = torch.nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_weight=None):
        # Primeira convolução em grafo
        x = self.gcn1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Segunda convolução em grafo
        x = self.gcn2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Mapeamento final de dimensão (embeddings de tamanho 32)
        x = self.linear(x)
        return x
    
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
    Prioriza o ajuste de opacidade.
    """
    n_total = len(opacity)
    print(f"\n🎯 Calculando filtros dinâmicos para ~{target_count:,} gaussianas...")
    print(f"   Total disponível: {n_total:,}")
    
    if n_total <= target_count:
        print(f"   ⚠️ Total ({n_total:,}) já é menor que o target ({target_count:,})")
        print(f"   Usando threshold mínimo (0.01) para manter todas as gaussianas")
        return 0, 1, n_total
    
    best_op_thresh = None
    best_count = 0
    best_diff = float('inf')
    
    for op_thresh in np.linspace(0.01, 0.5, 50):
        mask = (opacity > op_thresh) & (scaling < 10.0)
        count = np.sum(mask)
        diff = abs(count - target_count)
        
        if diff < best_diff:
            best_diff = diff
            best_count = count
            best_op_thresh = op_thresh
            
            if diff < 500:
                break
    
    if best_op_thresh is not None:
        for op_offset in np.linspace(-0.02, 0.02, 10):
            op_test = max(0.01, best_op_thresh + op_offset)
            for sc_thresh in np.linspace(0.3, 2.0, 10):
                mask = (opacity > op_test) & (scaling < sc_thresh)
                count = np.sum(mask)
                diff = abs(count - target_count)
                
                if diff < best_diff:
                    best_diff = diff
                    best_count = count
                    best_op_thresh = op_test
                    best_sc_thresh = sc_thresh
                    
                    if diff < 200:
                        break
            if best_diff < 200:
                break
    
    if 'best_sc_thresh' not in locals():
        best_sc_thresh = 0.7
    
    print(f"   ✅ Melhor combinação encontrada:")
    print(f"      Opacity threshold: {best_op_thresh:.3f}")
    print(f"      Scale threshold: {best_sc_thresh:.3f}")
    print(f"      Gaussianas resultantes: {best_count:,}")
    
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

def load_deva(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    ann_map = {}
    for ann in data["annotations"]:
        name = os.path.splitext(ann["file_name"])[0]
        ann_map[name] = ann["segments_info"]
    return ann_map

import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode

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


def build_labels(
    scene,
    xyz,
    masks_path,
    propagate_to_neighbors=False,
    depth_tolerance=0.03,   # Tolerância relativa de z-buffer por frame (3cm)
    k_propagate=3,         # Aumentado para 3 para suavizar KNN no final
):
    N = xyz.shape[0]
    labels = np.full(N, -1, dtype=np.int64)
    best_z = np.full(N, np.inf, dtype=np.float32) # Guarda a menor profundidade em que a gaussiana foi vista

    cameras = scene.getTrainCameras()
    total_frames = len(cameras)

    print(f"\n🔍 Montando labels com Z-Buffer Global a partir de {masks_path}...")

    for idx, view in enumerate(cameras):
        name = view.image_name
        mask_path = os.path.join(masks_path, f"{name}.png")
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]

        H, W = mask.shape

        # 1. Projeta TODOS os pontos no frame atual para montar um Z-Buffer real da cena
        u, v, z_cam = project_points_with_depth(xyz, view, out_width=W, out_height=H)

        # Filtro de frustum básico
        valid_frustum = (z_cam > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid_idx = np.where(valid_frustum)[0]

        if valid_idx.size == 0:
            continue

        # 2. Z-buffer do frame atual (para tratar oclusão entre as próprias gaussianas)
        u_v = u[valid_idx]
        v_v = v[valid_idx]
        z_v = z_cam[valid_idx]

        flat_pixel_idx = v_v * W + u_v
        frame_depth_buffer = np.full(H * W, np.inf, dtype=np.float32)
        np.minimum.at(frame_depth_buffer, flat_pixel_idx, z_v.astype(np.float32))

        # 3. Pega só as gaussianas na superfície visível deste frame
        min_z_in_frame = frame_depth_buffer[flat_pixel_idx]
        is_front = z_v <= (min_z_in_frame + depth_tolerance)
        front_idx = valid_idx[is_front]

        if front_idx.size == 0:
            continue

        # 4. Amostra a máscara para as gaussianas da frente
        sids = mask[v[front_idx], u[front_idx]].astype(np.int64)
        
        # Filtra apenas pixels com rótulos válidos (assumindo 0 como background/sem máscara)
        valid_mask = sids > 0
        front_idx = front_idx[valid_mask]
        sids = sids[valid_mask]
        z_front = z_cam[front_idx]

        # 5. ATUALIZAÇÃO INTELIGENTE:
        # Atualiza o label SE a gaussiana estiver mais perto da câmera do que nas vistas anteriores
        closer_than_before = z_front < best_z[front_idx]
        target_idx = front_idx[closer_than_before]
        
        labels[target_idx] = sids[closer_than_before]
        best_z[target_idx] = z_front[closer_than_before]

        if (idx + 1) % 50 == 0 or (idx + 1) == total_frames:
            assigned_so_far = np.sum(labels != -1)
            print(f"  Frame {idx+1}/{total_frames} | Gaussianas com label até agora: {assigned_so_far:,}/{N:,}")

    assigned = np.sum(labels != -1)
    print(f"\n  Gaussianas rotuladas por projeção direta: {assigned}/{N} ({100*assigned/N:.1f}%)")

    # --- PROPAGAÇÃO KNN PARA PONTOS NÃO VISTOS ---
    if propagate_to_neighbors:
        valid_idx = np.where(labels != -1)[0]
        invalid_idx = np.where(labels == -1)[0]

        if len(valid_idx) > 0 and len(invalid_idx) > 0:
            print(f"  🔄 Propagando {len(invalid_idx):,} gaussianas sem label via KNN (k={k_propagate})...")
            
            nbrs = NearestNeighbors(n_neighbors=k_propagate, algorithm="kd_tree", n_jobs=-1).fit(xyz[valid_idx])
            _, nn_idx = nbrs.kneighbors(xyz[invalid_idx])

            if k_propagate == 1:
                labels[invalid_idx] = labels[valid_idx[nn_idx.flatten()]]
            else:
                neigh_labels = labels[valid_idx[nn_idx]]
                # scipy mode retorna array de modos
                modes, _ = mode(neigh_labels, axis=1, keepdims=False)
                labels[invalid_idx] = modes

    final_valid = np.sum(labels != -1)
    n_clusters = len(np.unique(labels[labels != -1]))
    print(f"\n📊 RESUMO FINAL:")
    print(f"  Total com label: {final_valid}/{N} ({100*final_valid/N:.1f}%)")
    print(f"  Número de clusters/instâncias: {n_clusters}")

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
    
    parser.add_argument("--model_type", default="gat", choices=["gat"])
    parser.add_argument("--gat_heads", default=2, type=int)
    parser.add_argument("--opacity_threshold", default=0.05, type=float)
    parser.add_argument("--max_scale_threshold", default=0.7, type=float, help="Filtro de escala contra elipsoides gigantes")
    parser.add_argument("--min_cluster_size", default=25, type=int)
    parser.add_argument("--use_or_condition", action="store_true", default=True)
    parser.add_argument("--target_gaussians", default=230000, type=int, help="Número alvo de gaussianas após filtragem")

    parser.add_argument("--render_full_pointcloud", action="store_true", default=True,
                         help="Propaga os labels para TODAS as gaussianas antes de renderizar (elimina buracos brancos)")
    parser.add_argument("--knn_reassign_k", default=7, type=int,
                         help="Número de vizinhos K nos embeddings para reatribuição rápida de ruído")
    parser.add_argument("--model_type_choice", default="gcn", choices=["gat", "gcn"])

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
        target_count=args.target_gaussians
    )
    
    args.opacity_threshold = dyn_op_thresh
    args.max_scale_threshold = dyn_sc_thresh
    
    mask_filter = (opacity > args.opacity_threshold) & (max_scaling < args.max_scale_threshold)

    xyz = xyz[mask_filter]
    rgb = rgb[mask_filter]
    opacity_filtered = opacity[mask_filter]

    print(f"\n⚙️ Geometria Filtrada: {len(xyz):,} de {len(opacity):,} Gaussianas restantes.")

    ann_map = load_deva(args.deva_json)
    # Substitua a chamada no main por:

    labels = build_labels(scene, xyz, args.masks_path)
    pipe = pipeline_params.extract(args)
    background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
    #labels = build_labels_via_id_rendering(scene,gaussians,pipe,background,args.masks_path,ann_map,device)
    render_clusters(labels, "debug_labels_2d", gaussians, scene, pipe, background, args, device, mask_filter)
    unique, counts = np.unique(labels, return_counts=True)
        
    print("\n========== LABELS GERADOS ==========")
    print("Número de labels:", len(unique))
        
    for label, count in zip(unique[:30], counts[:30]):
        print(f"Label {label}: {count:,} Gaussianas")
    # FEATURES
    scaler = StandardScaler()
    x_input = np.concatenate([
        scaler.fit_transform(xyz) * 0.7,
        scaler.fit_transform(rgb) * 1.5,
    ], axis=1)
    x = torch.tensor(x_input, dtype=torch.float)

    # ==========================================================
    # CONSTRUÇÃO DO GRAFO GEOMÉTRICO
    # ==========================================================
    print("\n🔗 Construindo grafo geométrico...")
    N_NEIGHBORS = 6
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="auto").fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)

    edges = []
    edge_weights = []
    all_neighbor_distances = distances[:, 1:].reshape(-1)
    sigma = np.median(all_neighbor_distances)
    print(f"  Sigma geométrico: {sigma:.4f}")

    distance_threshold = np.mean(all_neighbor_distances) + 1.0 * np.std(all_neighbor_distances)
    all_color_sims = []

    for i in range(len(indices)):
        for k in range(1, len(indices[i])):
            j = indices[i][k]
            color_sim = F.cosine_similarity(
                torch.tensor(rgb[i]).unsqueeze(0),
                torch.tensor(rgb[j]).unsqueeze(0), dim=1
            ).item()
            all_color_sims.append(color_sim)

    all_color_sims = np.array(all_color_sims)
    color_threshold = np.percentile(all_color_sims, 25)

    for i in range(len(indices)):
        for k in range(1, len(indices[i])):
            j = indices[i][k]
            d = distances[i][k]
            color_sim = F.cosine_similarity(
                torch.tensor(rgb[i]).unsqueeze(0),
                torch.tensor(rgb[j]).unsqueeze(0), dim=1
            ).item()

            if d <=3 and color_sim >= 0.8:
                spatial_weight = np.exp(-(d ** 2) / (2 * sigma ** 2 + 1e-8))
                color_weight = (color_sim + 1.0) * 0.5
                weight = (0.6 * spatial_weight + 1.2 * color_weight)
                
                edges.append([i, j])
                edge_weights.append(weight)
                edges.append([j, i])
                edge_weights.append(weight)

            

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights_t = torch.tensor(edge_weights, dtype=torch.float, device=device)

    # MODELO GAT
    # MODELO (GAT ou GCN)
    in_dim = x.shape[1]
    
    if args.model_type == "gcn":
        model = GaussianGCN(in_dim=in_dim, out_dim=32).to(device)
    else:
        model = GaussianGAT(in_channels=in_dim, heads=args.gat_heads).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss()

    data = Data(x=x, edge_index=edge_index).to(device)
    labels_t = torch.tensor(labels, device=device)

    print("\n🎓 Treinando GAT/GCN com Early Stopping...")

    # Inicializa o early stopping
    early_stopping = EarlyStopping(patience=20, min_delta=0.0001, verbose=True)

    # Lista para guardar histórico de losses (opcional)
    loss_history = []
    menor_loss = float('inf')
    for epoch in range(1000):  # Aumente o máximo de épocas
        optimizer.zero_grad()

        if args.model_type == "gcn":
            embeddings = model(data.x, data.edge_index, edge_weight=edge_weights_t)
        else:
            embeddings = model(data.x, data.edge_index)

        z = F.normalize(embeddings, dim=1)

        loss = criterion(
            z,
            labels_t,
            data.edge_index,
            edge_weights=edge_weights_t
        )

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)
        
        # Atualiza o menor loss (tracking)
        if current_loss < menor_loss:
            menor_loss = current_loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Loss: {current_loss:.4f} | Best: {menor_loss:.4f}")

        # Verifica early stopping
        if early_stopping(current_loss, model):
            break

    # Restaura o melhor modelo encontrado
    early_stopping.restore_best_model(model)

    # Opcional: plota o histórico de losses
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.savefig(os.path.join(args.output, 'loss_history.png'))
        plt.close()
        print(f"  📊 Histórico de losses salvo em: {os.path.join(args.output, 'loss_history.png')}")
    except Exception as e:
        print(f"  ⚠️ Não foi possível salvar o gráfico de losses: {e}")

    # EXTRAÇÃO DE EMBEDDINGS
    model.eval()
    with torch.no_grad():
        if args.model_type == "gcn":
            out_emb = model(data.x, data.edge_index, edge_weight=edge_weights_t)
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

    render_clusters(cluster_labels, args.model_type, gaussians, scene, pipe, background, args, device, mask_filter)

    gaussians._features_dc.data = orig_dc
    print("\n✅ Concluído! Pipeline executado de forma otimizada.")

if __name__ == "__main__":
    main()