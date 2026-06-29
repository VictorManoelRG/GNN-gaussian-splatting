import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import os

from scene import Scene
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import hdbscan

# ==========================================================
# GNN MODEL
# ==========================================================
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

class GaussianGNN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 32, heads=2, dropout=0.2)
        self.conv2 = GATConv(32 * 2, 32, heads=1, dropout=0.2)
        self.conv3 = GATConv(32, 32, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

# ==========================================================
# UTILS
# ==========================================================
def labels_to_colors(labels):
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20")
    label_to_color = {}
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            label_to_color[label] = np.array([0, 0, 0])
        else:
            color = np.array(cmap(i % 20)[:3])
            label_to_color[label] = color
            
    return np.array([label_to_color[l] for l in labels])

# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = ArgumentParser(description="Gaussian GNN Full Segmentation")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output_path", default="output_full_seg", type=str)
    args = get_combined_args(parser)

    os.makedirs(args.output_path, exist_ok=True)
    safe_state(False)

    # 1. Carregar Dados
    gaussians = GaussianModel(model_params.extract(args).sh_degree)
    scene = Scene(model_params.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    
    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().squeeze(1).numpy()
    objects_dc = gaussians._objects_dc.detach().cpu().numpy().squeeze(1)

    print(f"Total de Gaussianas: {xyz.shape[0]}")

    # 2. Normalização
    scaler = StandardScaler()
    xyz_norm = scaler.fit_transform(xyz)
    rgb_norm = scaler.fit_transform(rgb)
    obj_norm = scaler.fit_transform(objects_dc)

    # 3. Grafo
    graph_features = np.concatenate([
        xyz_norm * 1.0, 
        rgb_norm * 0.2, 
        obj_norm * 1.2
    ], axis=1)

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree', n_jobs=-1).fit(graph_features)
    _, indices = nbrs.kneighbors(graph_features)

    obj_norm_unit = obj_norm / (np.linalg.norm(obj_norm, axis=1, keepdims=True) + 1e-8)

    edges = []
    for i in range(len(indices)):
        for j in indices[i]:
            if i == j: continue
            sem_sim = np.dot(obj_norm_unit[i], obj_norm_unit[j])
            if sem_sim > 0.8:
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    del edges

    # 4. Input
    x_input = np.concatenate([xyz_norm, rgb_norm, obj_norm], axis=1)
    x_gnn = torch.tensor(x_input, dtype=torch.float)

    # 5. Treino
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_gnn = GaussianGNN(in_channels=x_gnn.shape[1]).to(device)
    optimizer = torch.optim.Adam(model_gnn.parameters(), lr=0.001)
    data = Data(x=x_gnn, edge_index=edge_index).to(device)

    print("Treinando GNN...")
    model_gnn.train()

    for epoch in range(200):
        optimizer.zero_grad()
        z = F.normalize(model_gnn(data.x, data.edge_index), dim=1)

        src, dst = data.edge_index
        cos_pos = torch.sum(z[src] * z[dst], dim=1)
        loss_pos = 1 - cos_pos.mean()

        neg_idx = torch.randint(0, z.shape[0], (src.shape[0],), device=device)
        cos_neg = torch.sum(z[src] * z[neg_idx], dim=1)
        loss_neg = torch.clamp(cos_neg - 0.3, min=0).mean()

        loss = loss_pos + loss_neg
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # 6. Clustering
    model_gnn.eval()
    with torch.no_grad():
        embeddings = F.normalize(model_gnn(data.x, data.edge_index), dim=1).cpu().numpy()

    print("HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=150, min_samples=15, core_dist_n_jobs=-1).fit(embeddings)
    labels = clusterer.labels_

    print(f"Clusters: {len(np.unique(labels))}")

    # ==========================================================
    # 🔥 RENDER SEM BORRÃO
    # ==========================================================

    cluster_colors = labels_to_colors(labels)

    cluster_colors_tensor = torch.tensor(
        cluster_colors, dtype=torch.float32, device="cuda"
    )

    # evitar extremos
    cluster_colors_tensor = torch.clamp(cluster_colors_tensor, 0.05, 0.95)

    # 🔥 SHAPE CORRETO
    cluster_colors_tensor = cluster_colors_tensor.unsqueeze(1)  # [N,1,3]

    # backup
    orig_shs = gaussians._features_dc.clone()
    orig_rest = gaussians._features_rest.clone()
    orig_opacity = gaussians._opacity.clone()

    # aplicar cores
    gaussians._features_dc.data = (cluster_colors_tensor - 0.5) / 0.28209
    #gaussians._features_rest.data.zero_()

    # ⚠️ NÃO mexer no rest (por enquanto)
    # gaussians._features_rest.data = orig_rest

    # pipeline (VOCÊ ESQUECEU ISSO)
    pipe = pipeline_params.extract(args)

    background = torch.tensor([1, 1, 1], dtype=torch.float, device="cuda")

    print("Renderizando...")
    for idx, view in enumerate(scene.getTrainCameras()):
        render_pkg = render(view, gaussians, pipe, background)
        img = render_pkg["render"]

        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(args.output_path, f"seg_{view.image_name}.png"), img_bgr)

    # restore
    gaussians._features_dc.data = orig_shs
    gaussians._features_rest.data = orig_rest
    gaussians._opacity.data = orig_opacity


if __name__ == "__main__":
    main()