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
import hdbscan  # Reintroduzido

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
    
    # Se já temos menos que o target, não filtrar
    if n_total <= target_count:
        print(f"   ⚠️ Total ({n_total:,}) já é menor que o target ({target_count:,})")
        print(f"   Usando threshold mínimo (0.01) para manter todas as gaussianas")
        return 0.01, 10.0, n_total
    
    # Buscar threshold de opacidade que atinge o target (considerando escala máxima relaxada)
    best_op_thresh = None
    best_count = 0
    best_diff = float('inf')
    
    # Testar thresholds de opacidade de 0.01 a 0.5
    for op_thresh in np.linspace(0.01, 0.5, 50):
        # Primeiro com escala relaxada (10.0)
        mask = (opacity > op_thresh) & (scaling < 10.0)
        count = np.sum(mask)
        diff = abs(count - target_count)
        
        if diff < best_diff:
            best_diff = diff
            best_count = count
            best_op_thresh = op_thresh
            
            # Se já estamos muito próximos, podemos parar
            if diff < 500:
                break
    
    # Agora refinar com escala se necessário
    if best_op_thresh is not None:
        # Testar combinações próximas ao melhor threshold de opacidade
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
    
    # Se não encontrou um threshold de escala, usar valor padrão
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

def build_labels(scene, xyz, masks_path, ann_map, propagate_to_neighbors=True):
    N = xyz.shape[0]
    votes = [{} for _ in range(N)]
    has_vote = np.zeros(N, dtype=bool)

    for view in scene.getTrainCameras():
        name = view.image_name
        mask_path = os.path.join(masks_path, f"{name}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        segments = ann_map.get(name, [])
        id_to_weight = {}
        for seg in segments:
            sid = seg["id"]
            id_to_weight[sid] = seg["score"] * (1.0 / np.sqrt(seg["area"] + 1e-8))

        u, v = project_points(xyz, view)
        H, W = mask.shape

        for i in range(N):
            if 0 <= u[i] < W and 0 <= v[i] < H:
                sid = int(mask[v[i], u[i]])
                if sid == 0:
                    continue
                votes[i][sid] = votes[i].get(sid, 0) + id_to_weight.get(sid, 1.0)
                has_vote[i] = True

    labels = np.full(N, -1)
    for i in range(N):
        if len(votes[i]) > 0:
            labels[i] = max(votes[i], key=votes[i].get)

    print(f"\n📊 BUILD LABELS:")
    print(f"  Gaussianas com voto: {np.sum(has_vote)}/{N} ({100*np.sum(has_vote)/N:.1f}%)")
    
    if propagate_to_neighbors and np.sum(labels == -1) > 0:
        print("  Propagando labels para vizinhos...")
        nbrs = NearestNeighbors(n_neighbors=5).fit(xyz)
        _, indices = nbrs.kneighbors(xyz)
        new_labels = labels.copy()
        for i in range(N):
            if labels[i] == -1:
                for j in indices[i][1:]:
                    if labels[j] != -1:
                        new_labels[i] = labels[j]
                        break
        labels = new_labels
    return labels


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
        colors[labels == -1] = [0.1, 0.1, 0.1, 1]  # Ruído fica cinza escuro no visualizador 3D
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
def main():
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--masks_path", required=True)
    parser.add_argument("--deva_json", required=True)
    parser.add_argument("--output", default="output_seg")
    parser.add_argument("--visualize_3d", action="store_true")
    parser.add_argument("--save_pointclouds", action="store_true")
    
    parser.add_argument("--model_type", default="gat", choices=["gat"])
    parser.add_argument("--gat_heads", default=2, type=int)
    parser.add_argument("--opacity_threshold", default=0.05, type=float)
    parser.add_argument("--max_scale_threshold", default=0.7, type=float, help="Filtro de escala contra elipsoides gigantes")
    parser.add_argument("--min_cluster_size", default=40, type=int)
    parser.add_argument("--use_or_condition", action="store_true", default=True)
    parser.add_argument("--target_gaussians", default=230000, type=int, help="Número alvo de gaussianas após filtragem")

    args = get_combined_args(parser)
    os.makedirs(args.output, exist_ok=True)
    safe_state(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD GAUSSIANS
    gaussians = GaussianModel(model_params.extract(args).sh_degree)
    scene = Scene(model_params.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)

    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().squeeze(1).numpy()
    
    # FILTRAGEM GEOMÉTRICA (OPACIDADE + ESCALA)
    opacity = torch.sigmoid(gaussians._opacity).detach().cpu().numpy().squeeze()
    scaling = torch.exp(gaussians._scaling).detach().cpu().numpy()
    max_scaling = np.max(scaling, axis=1)

    # --- FILTRO DINÂMICO ---
    # Calcular thresholds dinâmicos para atingir ~target_gaussians
    dyn_op_thresh, dyn_sc_thresh, final_count = compute_dynamic_filters(
        opacity, 
        max_scaling, 
        target_count=args.target_gaussians
    )
    
    # Usar os thresholds dinâmicos (sobrescreve os argumentos)
    args.opacity_threshold = dyn_op_thresh
    args.max_scale_threshold = dyn_sc_thresh
    
    mask_filter = (opacity > args.opacity_threshold) & (max_scaling < args.max_scale_threshold)

    xyz = xyz[mask_filter]
    rgb = rgb[mask_filter]
    opacity_filtered = opacity[mask_filter]

    print(f"\n⚙️ Geometria Filtrada: {len(xyz):,} de {len(opacity):,} Gaussianas restantes.")

    ann_map = load_deva(args.deva_json)
    labels = build_labels(scene, xyz, args.masks_path, ann_map, propagate_to_neighbors=True)

    # FEATURES
    scaler = StandardScaler()
    x_input = np.concatenate([
        scaler.fit_transform(xyz) * 0.7,
        scaler.fit_transform(rgb) * 1.0,
    ], axis=1)
    x = torch.tensor(x_input, dtype=torch.float)

    # CONSTRUÇÃO DO GRAFO PONDERADO
    print("\n🔗 Construindo grafo...")
    nbrs = NearestNeighbors(n_neighbors=9).fit(x_input)
    distances, indices = nbrs.kneighbors(x_input)

    edges = []
    edge_weights = []

    for i in range(len(indices)):
        for idx, j in enumerate(indices[i]):
            if i == j:
                continue
            
            color_sim = F.cosine_similarity(
                torch.tensor(rgb[i]).unsqueeze(0), 
                torch.tensor(rgb[j]).unsqueeze(0)
            ).item()
            
            dist_cond = distances[i][idx] < 5.0
            color_cond = color_sim > 0.6
            
            if args.use_or_condition if (dist_cond or color_cond) else (dist_cond and color_cond):
                edges.append([i, j])
                weight = float(opacity_filtered[i] * opacity_filtered[j])
                edge_weights.append(weight)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights_t = torch.tensor(edge_weights, dtype=torch.float, device=device)

    # MODELO
    in_dim = x.shape[1]
    model = GaussianGAT(in_dim, heads=args.gat_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss(temperature=0.07, pos_margin=0.5, neg_margin=0.2)

    data = Data(x=x, edge_index=edge_index).to(device)
    labels_t = torch.tensor(labels, device=device)

    print("\n🎓 Treinando GAT Ponderada...")
    for epoch in range(200):
        optimizer.zero_grad()
        z = F.normalize(model(data.x, data.edge_index), dim=1)
        loss = criterion(z, labels_t, data.edge_index, edge_weights=edge_weights_t)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    # EXTRAÇÃO DE EMBEDDINGS
    model.eval()
    with torch.no_grad():
        emb = F.normalize(model(data.x, data.edge_index), dim=1).cpu().numpy()

    # --- REINTRODUÇÃO DO HDBSCAN APÓS O TREINO ---
    print("\n📊 Clusterizando os novos embeddings com HDBSCAN...")
    auto_min_size = max(10, int(args.min_cluster_size * 0.8))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=auto_min_size).fit(emb)
    cluster_labels = clusterer.labels_
    unique_gnn = np.unique(cluster_labels)
    print(f"  GAT + HDBSCAN: {len(unique_gnn[unique_gnn != -1])} clusters identificados.")

    # VISUALIZAÇÃO 3D
    if args.visualize_3d and OPEN3D_AVAILABLE:
        pointcloud_dir = os.path.join(args.output, "pointclouds") if args.save_pointclouds else None
        visualize_clusters_open3d(xyz, cluster_labels, f"GNN Modulada + HDBSCAN")
        
        if args.save_pointclouds and pointcloud_dir:
            gat_path = os.path.join(pointcloud_dir, "gat_clusters.ply")
            visualize_clusters_open3d(xyz, cluster_labels, "GAT Clusters", save_path=gat_path)
            create_viewer_script(pointcloud_dir, args.output)

    # RENDER 2D
    print("\n" + "="*50)
    print("RENDER 2D")
    print("="*50)
    orig_dc = gaussians._features_dc.clone()
    pipe = pipeline_params.extract(args)
    background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)

    render_clusters(cluster_labels, args.model_type, gaussians, scene, pipe, background, args, device, mask_filter)

    gaussians._features_dc.data = orig_dc
    print("\n✅ Concluído! Mapeamento de clusters discreto gerado no render 2D.")

if __name__ == "__main__":
    main()