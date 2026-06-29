import torch
import numpy as np
import open3d as o3d

from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

from argparse import ArgumentParser

from sklearn.neighbors import NearestNeighbors

from torch_geometric.data import Data


def main():

    parser = ArgumentParser(description="Gaussian Crop + Graph Builder")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)

    args = get_combined_args(parser)

    safe_state(False)

    # ==========================================================
    # CARREGAR GAUSSIANAS
    # ==========================================================

    gaussians = GaussianModel(model.extract(args).sh_degree)

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
    # VISUALIZAR NUVEM
    # ==========================================================

    points = xyz.numpy()

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    colors = rgb.numpy()

    colors = (
        colors - colors.min()
    ) / (
        colors.max() - colors.min()
    )

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

    # ==========================================================
    # DEFINIR CROP ESPACIAL
    #
    # SUBSTITUA PELOS VALORES
    # QUE VOCÊ PEGOU NO CLOUDCOMPARE
    # ==========================================================

    xmin = -2.97541118
    xmax = 3.82262826

    ymin = -5.42769956
    ymax = 1.03320932

    zmin = -11.80160141
    zmax = -4.22889137

    # ==========================================================
    # FILTRAR GAUSSIANAS
    # ==========================================================

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
    # NORMALIZAR POSIÇÕES
    # ==========================================================

    cropped_xyz = (
        cropped_xyz - cropped_xyz.mean(dim=0)
    )

    # ==========================================================
    # CRIAR FEATURES
    # ==========================================================

    features = torch.cat([
        cropped_xyz,
        cropped_rgb,
        cropped_opacity
    ], dim=1)

    print("\nFeatures shape:")

    print(features.shape)

    # ==========================================================
    # CONSTRUIR GRAFO KNN
    # ==========================================================

    K = 16

    knn = NearestNeighbors(
        n_neighbors=K
    )

    knn.fit(cropped_xyz.numpy())

    distances, neighbors = knn.kneighbors(
        cropped_xyz.numpy()
    )

    # ==========================================================
    # CRIAR EDGE INDEX
    # ==========================================================

    edge_list = []

    for i in range(cropped_xyz.shape[0]):

        for j in neighbors[i]:

            edge_list.append([i, j])

    edge_index = torch.tensor(
        edge_list,
        dtype=torch.long
    ).t().contiguous()

    print("\nEdge index shape:")

    print(edge_index.shape)

    # ==========================================================
    # CRIAR GRAFO PYTORCH GEOMETRIC
    # ==========================================================

    graph = Data(
        x=features,
        edge_index=edge_index
    )

    print("\nGraph criado:")

    print(graph)

    # ==========================================================
    # VISUALIZAR CROP
    # ==========================================================

    cropped_pcd = o3d.geometry.PointCloud()

    cropped_pcd.points = o3d.utility.Vector3dVector(
        cropped_xyz.numpy()
    )

    cropped_colors = cropped_rgb.numpy()

    cropped_colors = (
        cropped_colors - cropped_colors.min()
    ) / (
        cropped_colors.max() - cropped_colors.min()
    )

    cropped_pcd.colors = o3d.utility.Vector3dVector(
        cropped_colors
    )

    print("\nAbrindo crop...")

    o3d.visualization.draw_geometries(
        [cropped_pcd]
    )

    # ==========================================================
    # PRONTO
    #
    # graph.x
    # graph.edge_index
    #
    # podem agora ser enviados para a GNN
    # ==========================================================


if __name__ == "__main__":

    with torch.no_grad():

        main()