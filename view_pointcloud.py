import argparse
import json
import os
import sys
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def visualize_file(file_path, point_size=2.0, show_axis=True):
    if not OPEN3D_AVAILABLE:
        print("❌ Erro: O pacote 'open3d' não está instalado neste ambiente.")
        print("Instale via: pip install open3d")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"❌ Arquivo não encontrado: {file_path}")
        sys.exit(1)

    # Tenta carregar os metadados associados (JSON)
    meta_path = file_path.replace(".ply", "_metadata.json")
    if os.path.exists(meta_path):
        print("\n" + "=" * 60)
        print("📋 METADADOS DO EXPERIMENTO")
        print("=" * 60)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                for k, v in metadata.items():
                    print(f"  • {k:20s}: {v}")
        except Exception as e:
            print(f"⚠️ Erro ao ler metadados: {e}")
        print("=" * 60 + "\n")

    # Carregamento do arquivo .ply ou fallback .npz
    if file_path.endswith(".ply"):
        print(f"🌐 Carregando nuvem de pontos PLY: {file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
    elif file_path.endswith(".npz"):
        print(f"🌐 Carregando nuvem de pontos NPZ: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        xyz = data["points"]
        labels = data["labels"]

        unique_labels = np.unique(labels)
        colors = np.zeros((len(labels), 3), dtype=np.float32)
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab20")

        for label in unique_labels:
            mask = labels == label
            if label == -1:
                colors[mask] = [0.12, 0.12, 0.12]
            else:
                colors[mask] = cmap(int(label) % 20)[:3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        print("❌ Formato não suportado. Use .ply ou .npz")
        sys.exit(1)

    print(f"📊 Total de pontos na nuvem: {len(pcd.points):,}")
    print("🖱️ Controle do visualizador Open3D:")
    print("   - Botão Esquerdo + Arrastar: Rotacionar")
    print("   - Botão Direito + Arrastar: Mover (Pan)")
    print("   - Scroll da Roda: Zoom")
    print("   - Tecla 'N': Alternar normais das superfícies")
    print("   - Tecla 'Q' ou ESC: Fechar janela\n")

    # Configuração da janela do Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualizador de Clusters 3D", width=1280, height=720)
    vis.add_geometry(pcd)

    # Ajuste do tamanho dos pontos
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0.05, 0.05, 0.05])  # Fundo escuro

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis.add_geometry(axis)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador de Nuvens de Pontos 3D (PLY / NPZ)")
    parser.add_argument("input", type=str, help="Caminho para o arquivo .ply ou .npz")
    parser.add_argument("--point_size", type=float, default=2.0, help="Tamanho dos pontos na tela (padrão: 2.0)")
    parser.add_argument("--no_axis", action="store_true", help="Oculta o sistema de eixos XYZ")

    args = parser.parse_args()
    visualize_file(args.input, point_size=args.point_size, show_axis=not args.no_axis)