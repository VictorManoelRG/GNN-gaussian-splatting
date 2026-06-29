import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

from argparse import ArgumentParser

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = ArgumentParser(description="Gaussian Clustering")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--method", default="dbscan", 
                        choices=["dbscan", "kmeans", "agglomerative"],
                        help="Método de clustering")
    parser.add_argument("--eps", default=0.5, type=float,
                        help="DBSCAN: raio de vizinhança")
    parser.add_argument("--min_samples", default=20, type=int,
                        help="DBSCAN: mínimo de amostras por cluster")
    parser.add_argument("--k", default=10, type=int,
                        help="K-Means/Agglomerative: número de clusters")
    
    args = get_combined_args(parser)

    safe_state(False)

    # ==========================================================
    # GAUSSIANAS
    # ==========================================================

    print("Carregando Gaussianas...")
    
    gaussians = GaussianModel(model.extract(args).sh_degree)

    scene = Scene(
        model.extract(args),
        gaussians,
        load_iteration=args.iteration,
        shuffle=False
    )

    # ==========================================================
    # EXTRAÇÃO DAS FEATURES
    # ==========================================================

    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().squeeze(1).numpy()
    opacity = gaussians._opacity.detach().cpu().numpy()
    
    # Embedding semântico do Gaussian Grouping
    objects_dc = gaussians._objects_dc.detach().cpu().numpy()
    objects_dc = objects_dc.squeeze(1)

    print(f"\nTotal de Gaussianas: {xyz.shape[0]}")
    print(f"Features: XYZ={xyz.shape[1]}, RGB={rgb.shape[1]}, "
          f"Opacity={opacity.shape[1]}, Objects_DC={objects_dc.shape[1]}")

    # ==========================================================
    # CROP (opcional - ajuste conforme sua cena)
    # ==========================================================
    
    # Exemplo de crop para uma mesa específica
    usar_crop = True
    
    if usar_crop:
        xmin = -4.75142860
        xmax = 1.62931848

        ymin = -4.88524866
        ymax = 4.03019190

        zmin = 1.87001801
        zmax = 13.04788589

        mask = (
            (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
            (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
            (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
        )

        xyz = xyz[mask]
        rgb = rgb[mask]
        opacity = opacity[mask]
        objects_dc = objects_dc[mask]
        
        print(f"\nApós crop: {xyz.shape[0]} Gaussianas")

    # ==========================================================
    # NORMALIZAÇÃO DAS FEATURES
    # ==========================================================
    
    print("\nNormalizando features...")
    
    xyz_norm = StandardScaler().fit_transform(xyz)
    rgb_norm = StandardScaler().fit_transform(rgb)
    obj_norm = StandardScaler().fit_transform(objects_dc)

    # ==========================================================
    # CONCATENAÇÃO COM PESOS (ajuste conforme sua necessidade)
    # ==========================================================
    
    # Pesos para cada feature (ajuste experimentalmente)
    # Quanto maior o peso, mais importância aquela feature tem
    pesos = {
        'xyz': 1.0,      # geometria é importante
        'rgb': 0.1,      # cor ajuda mas não é essencial
        'semantic': 0.6  # semântica é muito importante
    }
    
    features = np.concatenate([
        xyz_norm * pesos['xyz'],
        rgb_norm * pesos['rgb'],
        obj_norm * pesos['semantic']
    ], axis=1)
    
    # Normalização final (opcional)
    features = StandardScaler().fit_transform(features)
    
    print(f"Feature vector final: {features.shape[1]} dimensões")

    # ==========================================================
    # CLUSTERING
    # ==========================================================
    
    print(f"\nAplicando {args.method.upper()} clustering...")
    
    if args.method == "dbscan":
        clustering = DBSCAN(
            eps=args.eps, 
            min_samples=args.min_samples,
            metric='euclidean'
        )
        labels = clustering.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"DBSCAN params: eps={args.eps}, min_samples={args.min_samples}")
        print(f"Clusters encontrados: {n_clusters}")
        print(f"Pontos ruído: {n_noise} ({100*n_noise/len(labels):.1f}%)")
        
    elif args.method == "kmeans":
        from sklearn.cluster import KMeans
        clustering = KMeans(
            n_clusters=args.k,
            random_state=42,
            n_init=10
        )
        labels = clustering.fit_predict(features)
        
        print(f"K-Means com k={args.k}")
        print(f"Clusters: {len(np.unique(labels))}")
        
    elif args.method == "agglomerative":
        clustering = AgglomerativeClustering(
            n_clusters=args.k,
            linkage='ward'
        )
        labels = clustering.fit_predict(features)
        
        print(f"Agglomerative clustering com k={args.k}")
        print(f"Clusters: {len(np.unique(labels))}")

    # ==========================================================
    # PÓS-PROCESSAMENTO (opcional)
    # ==========================================================
    
    # Remover clusters muito pequenos (considerar como ruído)
    min_cluster_size = 10
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        if label != -1 and count < min_cluster_size:
            labels[labels == label] = -1
            print(f"Removido cluster pequeno {label} (tamanho {count})")

    # ==========================================================
    # COLORAÇÃO PARA VISUALIZAÇÃO
    # ==========================================================
    
    print("\nGerando visualização...")
    
    # Mapeia cada label para uma cor diferente
    unique_labels = np.unique(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    
    if n_clusters > 1000:
        # Usa colormap tab20 para até 20 clusters
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, n_clusters))
        
        # Cria array de cores para cada ponto
        point_colors = np.zeros((len(labels), 4))  # RGBA
        color_idx = 0
        
        for label in unique_labels:
            if label == -1:
                # Ruído = preto
                point_colors[labels == label] = [0, 0, 0, 1]
            else:
                point_colors[labels == label] = colors[color_idx % len(colors)]
                color_idx += 1
    else:
        point_colors = np.zeros((len(labels), 4))
        point_colors[:, :3] = [0.5, 0.5, 0.5]  # cinza se sem clusters

    # ==========================================================
    # VISUALIZAÇÃO 3D
    # ==========================================================
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(point_colors[:, :3])
    
    # Opcional: adicionar bounding boxes para cada cluster
    if n_clusters > 0 and n_clusters < 30:  # Evita overdraw se muitos clusters
        print("Adicionando bounding boxes...")
        for label in unique_labels:
            if label == -1:
                continue
                
            cluster_points = xyz[labels == label]
            if len(cluster_points) < 10:
                continue
                
            # Calcula bounding box
            min_bound = cluster_points.min(axis=0)
            max_bound = cluster_points.max(axis=0)
            
            # Cria caixa 3D
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            bbox.color = point_colors[labels == label][0, :3]  # cor do cluster
            pcd = o3d.geometry.PointCloud()  # TODO: melhorar isso
            # Nota: Open3D não suporta adicionar bounding boxes facilmente assim
            # Uma implementação melhor é necessária para múltiplas boxes
    
    # Ajusta visualização
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Clustering - {args.method.upper()}"
    )
    
    # ==========================================================
    # ESTATÍSTICAS FINAIS
    # ==========================================================
    
    print("\n" + "="*50)
    print("RESULTADOS FINAIS")
    print("="*50)
    print(f"Método: {args.method.upper()}")
    print(f"Total de Gaussianas: {len(xyz)}")
    print(f"Número de clusters: {n_clusters}")
    print(f"Ruído (pontos não clusterizados): {list(labels).count(-1)}")
    
    # Tamanho dos clusters
    print("\nTamanho dos clusters:")
    for label in unique_labels:
        if label == -1:
            continue
        size = np.sum(labels == label)
        print(f"  Cluster {label}: {size} gaussianas")


# ==========================================================
# FUNÇÃO AUXILIAR PARA TESTAR DIFERENTES CONFIGURAÇÕES
# ==========================================================

def testar_parametros(xyz_norm, rgb_norm, obj_norm):
    """
    Testa diferentes combinações de parâmetros para encontrar a melhor
    """
    print("\n" + "="*50)
    print("TESTE DE PARÂMETROS")
    print("="*50)
    
    configs = [
        {"name": "Geometria apenas", "pesos": {"xyz": 1.0, "rgb": 0.0, "semantic": 0.0}, "eps": 0.5},
        {"name": "Apenas semântica", "pesos": {"xyz": 0.0, "rgb": 0.0, "semantic": 1.0}, "eps": 0.5},
        {"name": "Geometria + Semântica", "pesos": {"xyz": 0.5, "rgb": 0.0, "semantic": 0.8}, "eps": 0.5},
        {"name": "Tudo balanceado", "pesos": {"xyz": 1.0, "rgb": 0.3, "semantic": 0.8}, "eps": 0.5},
        {"name": "Ênfase na semântica", "pesos": {"xyz": 0.3, "rgb": 0.1, "semantic": 1.2}, "eps": 0.6},
    ]
    
    for config in configs:
        # Concatena com pesos
        features = np.concatenate([
            xyz_norm * config["pesos"]["xyz"],
            rgb_norm * config["pesos"]["rgb"],
            obj_norm * config["pesos"]["semantic"]
        ], axis=1)
        
        features = StandardScaler().fit_transform(features)
        
        # DBSCAN
        clustering = DBSCAN(eps=config["eps"], min_samples=20)
        labels = clustering.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"\n{config['name']}:")
        print(f"  Clusters: {n_clusters}")
        print(f"  Ruído: {n_noise} ({100*n_noise/len(labels):.1f}%)")
        print(f"  Pesos: {config['pesos']}")


# ==========================================================
# ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    main()
    
    # Opcional: rodar teste de parâmetros
    # testar_parametros(xyz_norm, rgb_norm, obj_norm)