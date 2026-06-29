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

        self.conv1 = GCNConv(
            in_channels,
            64
        )

        self.conv2 = GCNConv(
            64,
            32
        )

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = ArgumentParser(
        description="Gaussian GNN Clustering"
    )

    model = ModelParams(
        parser,
        sentinel=True
    )

    pipeline = PipelineParams(parser)

    parser.add_argument(
        "--iteration",
        default=-1,
        type=int
    )

    args = get_combined_args(parser)

    safe_state(False)

    # ==========================================================
    # CARREGAR GAUSSIANAS
    # ==========================================================

    gaussians = GaussianModel(
        model.extract(args).sh_degree
    )

    scene = Scene(
        model.extract(args),
        gaussians,
        load_iteration=args.iteration,
        shuffle=False
    )

    # ==========================================================
    # EXTRAIR FEATURES
    # ==========================================================

    xyz = gaussians._xyz.detach().cpu()

    rgb = gaussians._features_dc.detach().cpu().squeeze(1)

    opacity = gaussians._opacity.detach().cpu()

    print("XYZ:", xyz.shape)
    print("RGB:", rgb.shape)
    print("Opacity:", opacity.shape)

    # ==========================================================
    # CROP ESPACIAL
    # ==========================================================

    xmin = -2.97541118
    xmax = 3.82262826

    ymin = -5.42769956
    ymax = 1.03320932

    zmin = -11.80160141
    zmax = -4.22889137

    mask = (
        (xyz[:, 0] >= xmin) &
        (xyz[:, 0] <= xmax) &
        (xyz[:, 1] >= ymin) &
        (xyz[:, 1] <= ymax) &
        (xyz[:, 2] >= zmin) &
        (xyz[:, 2] <= zmax)
    )

    cropped_xyz = xyz[mask]

    cropped_rgb = rgb[mask]

    cropped_opacity = opacity[mask]

    print("\nGaussianas após crop:")
    print(cropped_xyz.shape)

    # ==========================================================
    # NORMALIZAR XYZ
    # ==========================================================

    cropped_xyz = (
        cropped_xyz - cropped_xyz.mean(dim=0)
    )

    # ==========================================================
    # FEATURES
    # ==========================================================

    features = torch.cat([
        cropped_xyz,
        cropped_rgb,
        cropped_opacity
    ], dim=1)

    features_np = features.numpy()

    scaler = StandardScaler()

    features_np = scaler.fit_transform(
        features_np
    )

    # ==========================================================
    # CONSTRUIR GRAFO kNN
    # ==========================================================

    print("\nConstruindo grafo kNN...")

    k = 10

    nbrs = NearestNeighbors(
        n_neighbors=k,
        algorithm='ball_tree'
    )

    xyz_norm = StandardScaler().fit_transform(
        cropped_xyz.numpy()
    )

    rgb_norm = StandardScaler().fit_transform(
        cropped_rgb.numpy()
    )

    graph_features = np.concatenate([
        xyz_norm * 1.0,
        rgb_norm * 0.3
    ], axis=1)

    nbrs.fit(graph_features)

    distances, indices = nbrs.kneighbors(
        graph_features
    )

    edges = []

    for i in range(len(indices)):

        for j in indices[i]:

            edges.append([i, j])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    print("Edge index shape:")
    print(edge_index.shape)

    # ==========================================================
    # OBJETO DO GRAFO
    # ==========================================================

    x = torch.tensor(
        features_np,
        dtype=torch.float
    )

    data = Data(
        x=x,
        edge_index=edge_index
    )

    # ==========================================================
    # DEVICE
    # ==========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    data = data.to(device)

    # ==========================================================
    # GNN
    # ==========================================================

    model_gnn = GaussianGNN(
        in_channels=x.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(
        model_gnn.parameters(),
        lr=0.001
    )

    # ==========================================================
    # TREINO
    # ==========================================================

    print("\nTreinando GNN...\n")

    for epoch in range(100):

        model_gnn.train()

        optimizer.zero_grad()

        embeddings = model_gnn(
            data.x,
            data.edge_index
        )

        src = data.edge_index[0]

        dst = data.edge_index[1]

        emb_src = embeddings[src]

        emb_dst = embeddings[dst]

        # ======================================================
        # LOSS:
        # embeddings de vizinhos
        # devem ser parecidos
        # ======================================================

        loss_pos = F.mse_loss(emb_src, emb_dst)

        random_idx = torch.randint(
            0,
            embeddings.shape[0],
            (src.shape[0],),
            device=device
        )

        emb_neg = embeddings[random_idx]

        dist_neg = torch.norm(
            emb_src - emb_neg,
            dim=1
        )

        # penaliza embeddings negativos muito próximos
        loss_neg = torch.mean(
            1.0 / (dist_neg + 1e-6)
        )

        loss = loss_pos + 0.1 * loss_neg

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch} | Loss: {loss.item():.6f}"
            )

    # ==========================================================
    # GERAR EMBEDDINGS FINAIS
    # ==========================================================

    print("\nGerando embeddings finais...")

    model_gnn.eval()

    with torch.no_grad():

        embeddings = model_gnn(
            data.x,
            data.edge_index
        )

    embeddings_np = embeddings.cpu().numpy()

    print("Embeddings shape:")
    print(embeddings_np.shape)

    # ==========================================================
    # CLUSTERING NOS EMBEDDINGS
    # ==========================================================

    print("\nAplicando DBSCAN...")

    clustering = DBSCAN(
        eps=0.7,
        min_samples=20
    )

    labels = clustering.fit_predict(
        embeddings_np
    )

    unique_labels = np.unique(labels)

    print("Clusters encontrados:")
    print(unique_labels)

    # ==========================================================
    # COLORIR CLUSTERS
    # ==========================================================

    colors = plt.get_cmap("tab20")(
        labels / (labels.max() + 1)
    )

    # ruído = preto
    colors[labels == -1] = [0, 0, 0, 1]

    # ==========================================================
    # POINT CLOUD
    # ==========================================================

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        cropped_xyz.numpy()
    )

    pcd.colors = o3d.utility.Vector3dVector(
        colors[:, :3]
    )

    # ==========================================================
    # VISUALIZAR
    # ==========================================================

    o3d.visualization.draw_geometries([pcd])


# ==========================================================
# ENTRYPOINT
# ==========================================================

if __name__ == "__main__":

    main()