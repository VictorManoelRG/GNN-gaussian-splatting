import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

from argparse import ArgumentParser

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ==========================================================
# GNN
# ==========================================================

class GaussianGNN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = ArgumentParser(description="Gaussian GNN Clustering")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)

    args = get_combined_args(parser)

    safe_state(False)

    # ==========================================================
    # GAUSSIANAS
    # ==========================================================

    gaussians = GaussianModel(model.extract(args).sh_degree)

    scene = Scene(
        model.extract(args),
        gaussians,
        load_iteration=args.iteration,
        shuffle=False
    )

    # ==========================================================
    # EXTRAÇÃO
    # ==========================================================

    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().squeeze(1).numpy()
    opacity = gaussians._opacity.detach().cpu().numpy()

    # 🔥 NOVO: objects_dc (semântica do Gaussian Grouping)
    objects_dc = gaussians._objects_dc.detach().cpu().numpy()

    print("XYZ:", xyz.shape)
    print("RGB:", rgb.shape)
    print("Opacity:", opacity.shape)
    print("Objects_DC:", objects_dc.shape)

    # ==========================================================
    # CROP
    # ==========================================================

    xmin = -4.75142860
    xmax = 1.62931848

    ymin = -4.88524866
    ymax = 4.03019190

    zmin = 1.87001801
    zmax = 13.04788589

    # xmin = -2.97541118
    # xmax = 3.82262826

    # ymin = -5.42769956
    # ymax = 1.03320932

    # zmin = -11.80160141
    # zmax = -4.22889137

    mask = (
        (xyz[:, 0] >= xmin) &
        (xyz[:, 0] <= xmax) &
        (xyz[:, 1] >= ymin) &
        (xyz[:, 1] <= ymax) &
        (xyz[:, 2] >= zmin) &
        (xyz[:, 2] <= zmax)
    )

    xyz = xyz[mask]
    rgb = rgb[mask]
    opacity = opacity[mask]
    objects_dc = objects_dc[mask]

    print("\nApós crop:", xyz.shape)

    # ==========================================================
    # NORMALIZAÇÃO
    # ==========================================================

    xyz_norm = StandardScaler().fit_transform(xyz)
    rgb_norm = StandardScaler().fit_transform(rgb)
    opacity_norm = StandardScaler().fit_transform(opacity)
    objects_dc = objects_dc.squeeze(1)
    obj_norm = StandardScaler().fit_transform(objects_dc)

    # ==========================================================
    # NODE FEATURES (GNN INPUT)
    # ==========================================================

    features = np.concatenate([
        xyz_norm,
        rgb_norm,
        obj_norm  # 🔥 NOVO
    ], axis=1)

    features = StandardScaler().fit_transform(features)

    x = torch.tensor(features, dtype=torch.float)

    # ==========================================================
    # GRAFO kNN
    # ==========================================================

    print("\nConstruindo grafo kNN...")

    k = 10

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')

    # 🔥 grafo agora usa SEMÂNTICA + GEOMETRIA
    graph_features = np.concatenate([
        xyz_norm * 1.0,
        rgb_norm * 0.1,
        obj_norm * 0.6   # 🔥 MUITO IMPORTANTE
    ], axis=1)

    nbrs.fit(graph_features)

    _, indices = nbrs.kneighbors(graph_features)

    edges = []
    for i in range(len(indices)):
        for j in indices[i]:
            edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    print("Edge index:", edge_index.shape)

    # ==========================================================
    # DATASET
    # ==========================================================

    data = Data(
        x=x,
        edge_index=edge_index
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # ==========================================================
    # MODELO
    # ==========================================================

    model_gnn = GaussianGNN(in_channels=x.shape[1]).to(device)

    optimizer = torch.optim.Adam(model_gnn.parameters(), lr=0.001)

    # ==========================================================
    # TREINO
    # ==========================================================

    print("\nTreinando GNN...\n")

    for epoch in range(100):

        model_gnn.train()
        optimizer.zero_grad()

        embeddings = model_gnn(data.x, data.edge_index)

        src = data.edge_index[0]
        dst = data.edge_index[1]

        emb_src = embeddings[src]
        emb_dst = embeddings[dst]

        # loss local (vizinhos próximos no grafo)
        loss_pos = F.mse_loss(emb_src, emb_dst)

        # negativos aleatórios
        rand_idx = torch.randint(
            0,
            embeddings.shape[0],
            (src.shape[0],),
            device=device
        )

        emb_neg = embeddings[rand_idx]

        dist_neg = torch.norm(emb_src - emb_neg, dim=1)
        loss_neg = torch.mean(1.0 / (dist_neg + 1e-6))

        loss = loss_pos + 0.1 * loss_neg

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # ==========================================================
    # EMBEDDINGS
    # ==========================================================

    print("\nGerando embeddings finais...")

    model_gnn.eval()

    with torch.no_grad():
        embeddings = model_gnn(data.x, data.edge_index)

    embeddings_np = embeddings.cpu().numpy()

    # ==========================================================
    # CLUSTERING
    # ==========================================================

    print("\nAplicando DBSCAN...")

    clustering = DBSCAN(eps=0.7, min_samples=20)
    labels = clustering.fit_predict(embeddings_np)

    print("Clusters:", np.unique(labels))

    # ==========================================================
    # COLORAÇÃO
    # ==========================================================

    colors = plt.get_cmap("tab20")(
        (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
    )

    colors[labels == -1] = [0, 0, 0, 1]

    # ==========================================================
    # VISUALIZAÇÃO
    # ==========================================================

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd])


# ==========================================================
# ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    main()