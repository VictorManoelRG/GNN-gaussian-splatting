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
import hdbscan

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
# LOSS FOCADA EM RENDERIZAÇÃO 2D (SEM COORDENADAS 3D)
# ==========================================================
class RenderContrastiveLoss2D(torch.nn.Module):
    """
    Loss focada exclusivamente em renderização 2D.
    Não usa coordenadas 3D para o cálculo da perda.
    """
    def __init__(self, temperature=0.07, pos_margin=0.3, neg_margin=0.1):
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.num_views_sample = 8  # Número de views para amostrar
    
    def forward(self, embeddings, labels, views, gaussians, pipe, background):
        """
        Args:
            embeddings: [N, D] embeddings dos pontos 3D
            labels: [N] labels por ponto (do SAM)
            views: Lista de views da cena
            gaussians: Modelo Gaussian
            pipe: Pipeline params
            background: Cor de fundo
        """
        device = embeddings.device
        total_loss = torch.tensor(0.0, device=device)
        valid_views = 0
        
        # Seleciona views aleatórias para eficiência
        if len(views) > self.num_views_sample:
            import random
            sample_views = random.sample(views, self.num_views_sample)
        else:
            sample_views = views
        
        # Salva features originais para restaurar depois
        orig_dc = gaussians._features_dc.clone()
        orig_features_rest = gaussians._features_rest.clone()
        
        for view in sample_views:
            # 1. MAPEIA PONTOS 3D PARA PIXELS 2D
            xyz = gaussians._xyz.detach().cpu().numpy()
            u, v = self._project_points(xyz, view)
            
            H, W = view.image_height, view.image_width
            valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            
            if valid_mask.sum() < 10:
                continue
            
            # 2. CRIA MÁSCARA 2D A PARTIR DOS LABELS 3D
            mask_2d = self._create_2d_mask(labels, u, v, H, W, device)
            
            # 3. OBTÉM OS LABELS ÚNICOS NA MÁSCARA
            unique_labels = torch.unique(mask_2d[mask_2d != -1])
            
            if len(unique_labels) < 2:
                continue
            
            # 4. RENDERIZA A IMAGEM COM CORES POR CLASSE
            class_colors = self._get_class_colors(len(unique_labels) + 1, device)
            label_to_color = {int(label): class_colors[i] for i, label in enumerate(unique_labels)}
            label_to_color[-1] = torch.tensor([0.5, 0.5, 0.5], device=device)  # Fundo
            
            # Atribui cores aos pontos baseado nos labels
            point_colors = torch.zeros((len(xyz), 3), device=device)
            for label in unique_labels:
                mask = (labels == label)
                point_colors[mask] = label_to_color[int(label)]
            
            # Pontos sem label ficam cinza
            point_colors[labels == -1] = torch.tensor([0.5, 0.5, 0.5], device=device)
            
            # Renderiza com as cores das classes
            gaussians._features_dc.data = (point_colors.unsqueeze(1) - 0.5) / 0.28209
            gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)
            
            render_pkg = render(view, gaussians, pipe, background)
            rendered_img = render_pkg["render"]  # [3, H, W]
            
            # 5. EXTRAI EMBEDDINGS DOS PONTOS QUE PROJETAM NA IMAGEM
            valid_indices = torch.where(valid_mask)[0]
            valid_u = u[valid_mask]
            valid_v = v[valid_mask]
            
            point_embeddings = embeddings[valid_indices]  # [M, D]
            pixel_labels = mask_2d[valid_v, valid_u]  # [M]
            
            # 6. EXTRAI FEATURES DA IMAGEM RENDERIZADA NOS PIXELS CORRESPONDENTES
            pixel_features = rendered_img[:, valid_v, valid_u].T  # [M, 3]
            pixel_features = F.normalize(pixel_features, dim=1)
            
            # 7. CALCULA LOSS CONTRASTIVA EM 2D
            loss = self._compute_2d_contrastive_loss(
                point_embeddings, 
                pixel_features, 
                pixel_labels
            )
            
            if not torch.isnan(loss) and loss > 0:
                total_loss += loss
                valid_views += 1
        
        # Restaura features originais
        gaussians._features_dc.data = orig_dc
        gaussians._features_rest.data = orig_features_rest
        
        if valid_views > 0:
            return total_loss / valid_views
        else:
            return torch.tensor(0.0, device=device)
    
    def _project_points(self, xyz, view):
        """Projeta pontos 3D para coordenadas 2D"""
        P = view.full_proj_transform.detach().cpu().numpy()
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        proj = xyz_h @ P.T
        proj = proj[:, :3] / (proj[:, 3:4] + 1e-8)
        u = ((proj[:, 0] * 0.5 + 0.5) * view.image_width).astype(int)
        v = ((proj[:, 1] * 0.5 + 0.5) * view.image_height).astype(int)
        return u, v
    
    def _create_2d_mask(self, labels, u, v, H, W, device):
        """Cria máscara 2D a partir dos labels 3D projetados"""
        mask = torch.full((H, W), -1, dtype=torch.long, device=device)
        
        for i in range(len(u)):
            if 0 <= u[i] < W and 0 <= v[i] < H:
                mask[v[i], u[i]] = labels[i]
        
        return mask
    
    def _get_class_colors(self, num_classes, device):
        """Gera cores únicas para cada classe"""
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, max(num_classes, 1)))[:, :3]
        return torch.tensor(colors, dtype=torch.float32, device=device)
    
    def _compute_2d_contrastive_loss(self, point_embs, pixel_feats, labels):
        """Computa loss contrastiva entre pontos 3D e pixels 2D"""
        device = point_embs.device
        
        # Normaliza embeddings
        point_embs = F.normalize(point_embs, dim=1)
        pixel_feats = F.normalize(pixel_feats, dim=1)
        
        # Similaridade entre embeddings dos pontos e features dos pixels
        sim = F.cosine_similarity(point_embs, pixel_feats, dim=1)
        
        # Máscaras baseadas nos labels
        same_mask = (labels[:, None] == labels[None, :]) & (labels[:, None] != -1) & (labels[None, :] != -1)
        diff_mask = (labels[:, None] != labels[None, :]) & (labels[:, None] != -1) & (labels[None, :] != -1)
        
        # Loss positiva: pontos com mesmo label devem ser similares aos pixels
        pos_loss = torch.tensor(0.0, device=device)
        if same_mask.any():
            # Pega pares positivos
            pos_pairs = same_mask.nonzero(as_tuple=True)
            if len(pos_pairs[0]) > 0:
                pos_sim = sim[pos_pairs[0]]
                pos_loss = torch.relu(1 - pos_sim - self.pos_margin).mean()
        
        # Loss negativa: pontos com labels diferentes devem ser diferentes
        neg_loss = torch.tensor(0.0, device=device)
        if diff_mask.any():
            # Pega pares negativos
            neg_pairs = diff_mask.nonzero(as_tuple=True)
            if len(neg_pairs[0]) > 0:
                # Amostra alguns negativos para eficiência
                neg_indices = torch.randint(0, len(neg_pairs[0]), (min(1000, len(neg_pairs[0])),))
                neg_sim = sim[neg_pairs[0][neg_indices]]
                neg_loss = torch.relu(neg_sim - self.neg_margin).mean()
        
        # NCE Loss (InfoNCE) - versão simplificada
        nce_loss = torch.tensor(0.0, device=device)
        unique_labels = torch.unique(labels[labels != -1])
        
        for label in unique_labels[:5]:  # Limita para eficiência
            pos_mask = (labels == label)
            pos_embs = point_embs[pos_mask]
            pos_feats = pixel_feats[pos_mask]
            
            if len(pos_embs) < 2:
                continue
            
            # Para cada embedding positivo, encontra o negativo mais próximo
            for i in range(min(5, len(pos_embs))):
                anchor = pos_embs[i]
                positive = pos_feats[i]
                
                # Encontra negativos (pixels com labels diferentes)
                neg_mask = (labels != label) & (labels != -1)
                neg_feats = pixel_feats[neg_mask]
                
                if len(neg_feats) > 0:
                    # Seleciona negativos hard
                    neg_sim_all = F.cosine_similarity(anchor.unsqueeze(0), neg_feats)
                    hard_neg = neg_feats[neg_sim_all.topk(min(10, len(neg_feats))).indices]
                    
                    # Calcula NCE
                    pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
                    logits = torch.cat([pos_sim, F.cosine_similarity(anchor.unsqueeze(0), hard_neg)])
                    
                    loss_nce = -torch.log(
                        torch.exp(logits[0] / self.temperature) / 
                        (torch.exp(logits / self.temperature).sum() + 1e-8)
                    )
                    nce_loss += loss_nce
        
        if len(unique_labels) > 0:
            nce_loss = nce_loss / (len(unique_labels) * 5 + 1e-8)
        
        # Retorna combinação das losses
        return pos_loss + neg_loss + nce_loss * 0.5

# ==========================================================
# LOSS DE ENTROPIA CRUZADA 2D (ALTERNATIVA)
# ==========================================================
class CrossEntropyRenderLoss2D(torch.nn.Module):
    """
    Loss de entropia cruzada usando renderização de máscaras 2D
    """
    def __init__(self, num_classes, temperature=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
    
    def forward(self, embeddings, labels, views, gaussians, pipe, background):
        device = embeddings.device
        total_loss = 0.0
        valid_views = 0
        
        # Salva features originais
        orig_dc = gaussians._features_dc.clone()
        orig_rest = gaussians._features_rest.clone()
        
        # Gera cores para cada classe
        class_colors = self._get_class_colors(self.num_classes, device)
        
        for view in views[:8]:  # Usa até 8 views
            # Mapeia pontos para 2D
            xyz = gaussians._xyz.detach().cpu().numpy()
            u, v = self._project_points(xyz, view)
            
            H, W = view.image_height, view.image_width
            valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            
            if valid_mask.sum() < 10:
                continue
            
            # Cria máscara ground truth 2D
            gt_mask = torch.full((H, W), -1, dtype=torch.long, device=device)
            for i in range(len(u)):
                if valid_mask[i]:
                    gt_mask[v[i], u[i]] = labels[i] + 1  # +1 para evitar classe 0
            
            # Remove pixels sem label
            valid_pixels = gt_mask != -1
            if valid_pixels.sum() < 10:
                continue
            
            # Atribui cores aos pontos baseado nos labels
            point_colors = torch.zeros((len(xyz), 3), device=device)
            for i in range(self.num_classes):
                mask = (labels == i - 1)
                point_colors[mask] = class_colors[i]
            point_colors[labels == -1] = torch.tensor([0.5, 0.5, 0.5], device=device)
            
            # Renderiza
            gaussians._features_dc.data = (point_colors.unsqueeze(1) - 0.5) / 0.28209
            gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest)
            
            render_pkg = render(view, gaussians, pipe, background)
            rendered_img = render_pkg["render"]  # [3, H, W]
            
            # Converte imagem RGB para logits das classes
            logits = self._rgb_to_logits(rendered_img, class_colors)  # [H, W, num_classes]
            
            # Calcula cross entropy apenas nos pixels válidos
            loss = F.cross_entropy(
                logits[valid_pixels].reshape(-1, self.num_classes),
                gt_mask[valid_pixels]
            )
            
            total_loss += loss
            valid_views += 1
        
        # Restaura features
        gaussians._features_dc.data = orig_dc
        gaussians._features_rest.data = orig_rest
        
        return total_loss / max(1, valid_views)
    
    def _project_points(self, xyz, view):
        P = view.full_proj_transform.detach().cpu().numpy()
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        proj = xyz_h @ P.T
        proj = proj[:, :3] / (proj[:, 3:4] + 1e-8)
        u = ((proj[:, 0] * 0.5 + 0.5) * view.image_width).astype(int)
        v = ((proj[:, 1] * 0.5 + 0.5) * view.image_height).astype(int)
        return u, v
    
    def _get_class_colors(self, num_classes, device):
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, num_classes))[:, :3]
        return torch.tensor(colors, dtype=torch.float32, device=device)
    
    def _rgb_to_logits(self, rgb_img, class_colors):
        """Converte imagem RGB para logits das classes baseado na cor"""
        H, W = rgb_img.shape[1], rgb_img.shape[2]
        rgb_flat = rgb_img.permute(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
        
        # Calcula similaridade com cada cor de classe
        logits = []
        for color in class_colors:
            sim = F.cosine_similarity(rgb_flat, color.unsqueeze(0), dim=1)
            logits.append(sim / self.temperature)
        
        return torch.stack(logits, dim=1).reshape(H, W, -1)  # [H, W, num_classes]

# ==========================================================
# GAT ARCHITECTURE (MANTIDA IGUAL)
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
# PROJECTION & LABELS (MANTIDO IGUAL)
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
# VISUALIZATION & RENDER BY CLUSTER (MANTIDO IGUAL)
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

def render_clusters(labels, name, gaussians, scene, pipe, background, args, device, mask_filter=None):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(1, num_clusters)))[:, :3]
    np.random.shuffle(colors)
    
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    if -1 in color_map:
        color_map[-1] = np.array([0.0, 0.0, 0.0])

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
# MAIN MODIFICADO
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
    parser.add_argument("--opacity_threshold", default=0.1, type=float)
    parser.add_argument("--max_scale_threshold", default=0.7, type=float)
    parser.add_argument("--min_cluster_size", default=25, type=int)
    parser.add_argument("--max_edge_dist", default=0.15, type=float)
    parser.add_argument("--loss_type", default="contrastive_2d", 
                       choices=["contrastive_2d", "cross_entropy_2d"],
                       help="Tipo de loss focada em 2D")

    args = get_combined_args(parser)
    os.makedirs(args.output, exist_ok=True)
    safe_state(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD GAUSSIANS
    gaussians = GaussianModel(model_params.extract(args).sh_degree)
    scene = Scene(model_params.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)

    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().squeeze(1).numpy()
    
    # FILTRAGEM GEOMÉTRICA
    opacity = torch.sigmoid(gaussians._opacity).detach().cpu().numpy().squeeze()
    scaling = torch.exp(gaussians._scaling).detach().cpu().numpy()
    max_scaling = np.max(scaling, axis=1)

    mask_filter = (opacity > args.opacity_threshold) & (max_scaling < args.max_scale_threshold)

    xyz = xyz[mask_filter]
    rgb = rgb[mask_filter]
    opacity_filtered = opacity[mask_filter]
    scaling_filtered = scaling[mask_filter]

    print(f"\n⚙️ Geometria Filtrada: {len(xyz):,} de {len(opacity):,} Gaussianas restantes.")

    ann_map = load_deva(args.deva_json)
    labels = build_labels(scene, xyz, args.masks_path, ann_map, propagate_to_neighbors=True)

    # FEATURES (Mantido igual)
    scaler = StandardScaler()
    log_scaling = np.log(scaling_filtered + 1e-8)
    
    x_input = np.concatenate([
        scaler.fit_transform(xyz) * 1.2,
        scaler.fit_transform(rgb) * 0.7,
        scaler.fit_transform(log_scaling) * 0.2,
    ], axis=1)
    x = torch.tensor(x_input, dtype=torch.float)

    # CONSTRUÇÃO DO GRAFO
    print("\n🔗 Construindo grafo geométrico 3D...")
    nbrs_geo = NearestNeighbors(n_neighbors=12).fit(xyz)
    distances, indices = nbrs_geo.kneighbors(xyz)

    edges = []
    edge_weights = []

    for i in range(len(indices)):
        for idx, j in enumerate(indices[i]):
            if i == j:
                continue
            if distances[i][idx] < args.max_edge_dist:
                edges.append([i, j])
                weight = float(opacity_filtered[i] * opacity_filtered[j]) / (1.0 + distances[i][idx])
                edge_weights.append(weight)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights_t = torch.tensor(edge_weights, dtype=torch.float, device=device)

    # MODELO GAT
    in_dim = x.shape[1]
    model = GaussianGAT(in_dim, heads=args.gat_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # SELECIONA A LOSS FOCADA EM 2D
    if args.loss_type == "contrastive_2d":
        print("\n🎓 Treinando com LOSS CONTRASTIVA 2D (sem coordenadas 3D)...")
        criterion = RenderContrastiveLoss2D(
            temperature=0.07, 
            pos_margin=0.3, 
            neg_margin=0.1
        )
    else:
        num_unique = len(np.unique(labels[labels != -1]))
        print(f"\n🎓 Treinando com LOSS DE ENTROPIA CRUZADA 2D ({num_unique} classes)...")
        criterion = CrossEntropyRenderLoss2D(
            num_classes=num_unique + 1,
            temperature=0.1
        )

    data = Data(x=x, edge_index=edge_index).to(device)
    labels_t = torch.tensor(labels, device=device)
    
    # Preparação para renderização
    pipe = pipeline_params.extract(args)
    background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
    views = scene.getTrainCameras()[:20]  # Usa até 20 views para treino

    # LOOP DE TREINAMENTO
    print(f"\n🎓 Treinando GAT com foco em renderização 2D...")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        z = F.normalize(model(data.x, data.edge_index), dim=1)
        
        # Compute loss focada em 2D
        loss = criterion(
            z, 
            labels_t, 
            views,
            gaussians, 
            pipe, 
            background
        )
        
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Loss 2D: {loss.item():.4f}")

    # EXTRAÇÃO DE EMBEDDINGS
    model.eval()
    with torch.no_grad():
        emb = F.normalize(model(data.x, data.edge_index), dim=1).cpu().numpy()

    # CLUSTERIZAÇÃO COM HDBSCAN
    print("\n📊 Clusterizando os novos embeddings com HDBSCAN...")
    auto_min_size = max(10, int(args.min_cluster_size * 0.8))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=auto_min_size).fit(emb)
    cluster_labels = clusterer.labels_
    unique_gnn = np.unique(cluster_labels)
    print(f"  GAT + HDBSCAN: {len(unique_gnn[unique_gnn != -1])} clusters identificados.")

    # VISUALIZAÇÃO 3D
    if args.visualize_3d and OPEN3D_AVAILABLE:
        pointcloud_dir = os.path.join(args.output, "pointclouds") if args.save_pointclouds else None
        visualize_clusters_open3d(xyz, cluster_labels, f"GNN 2D + HDBSCAN")
        
        if args.save_pointclouds and pointcloud_dir:
            gat_path = os.path.join(pointcloud_dir, "gat_clusters.ply")
            visualize_clusters_open3d(xyz, cluster_labels, "GAT Clusters", save_path=gat_path)

    # RENDER 2D
    print("\n" + "="*50)
    print("RENDER 2D")
    print("="*50)
    orig_dc = gaussians._features_dc.clone()
    pipe = pipeline_params.extract(args)
    background = torch.tensor([1, 1, 1], dtype=torch.float, device=device)

    render_clusters(cluster_labels, args.model_type, gaussians, scene, pipe, background, args, device, mask_filter)

    gaussians._features_dc.data = orig_dc
    print("\n✅ Concluído! A segmentação focada em 2D foi gerada com sucesso.")

if __name__ == "__main__":
    main()