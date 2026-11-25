"""Gera as versoes em ingles dos dois diagramas do artigo.

Os diagramas originais (figuras/pipeline_metodo.png e figuras/arquiteturas_gnn.png)
estao em portugues; este script desenha o mesmo layout com o texto em ingles e
salva figuras/pipeline_metodo_en.png e figuras/arquiteturas_gnn_en.png (com os
respectivos .pdf).

    python gerar_diagramas_en.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})

CINZA = dict(fc="#ededed", ec="#4d4d4d")
AZUL = dict(fc="#d9e6f7", ec="#2f6fc0")
LARANJA = dict(fc="#fae5cd", ec="#e08a2e")
VERDE = dict(fc="#dfeedd", ec="#3d7a3d")
ROXO = dict(fc="#e6dff3", ec="#7a5bbf")
BRANCO = dict(fc="#ffffff", ec="#111111")


def caixa(ax, x0, y0, x1, y1, titulo, linhas=(), cores=CINZA, tam_titulo=12, tam_texto=10,
          arredonda=0.4, linhas_negrito=False, espaco=1.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={arredonda}",
                                linewidth=1.6, **cores))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    altura = 1.15 * espaco * tam_texto / 10
    total = (1 if titulo else 0) * 1.45 + len(linhas)
    y = cy + (total - 1) * altura / 2
    if titulo:
        ax.text(cx, y, titulo, ha="center", va="center", fontsize=tam_titulo, fontweight="bold")
        y -= altura * 1.45
    for linha in linhas:
        ax.text(cx, y, linha, ha="center", va="center", fontsize=tam_texto,
                fontweight="bold" if linhas_negrito else "normal")
        y -= altura


def seta(ax, x0, y0, x1, y1, largura=1.7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=largura, color="#111111", shrinkA=0, shrinkB=0))


def pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 7.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    # --- Etapa 1 ---
    ax.add_patch(FancyBboxPatch((1.5, 29.5), 97, 25.5, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="#f7f7f7", ec="#b8b8b8", linewidth=1.4))
    ax.text(3.5, 52.6, "Stage 1: scene reconstruction and initial labels",
            fontsize=13.5, fontweight="bold", va="center")

    caixa(ax, 3.5, 33.0, 13.0, 47.5, "Scene", ["images"], CINZA, 13, 13, linhas_negrito=True)
    caixa(ax, 17.0, 40.5, 37.0, 49.0, "COLMAP (SfM)",
          ["camera poses and", "sparse point cloud"], AZUL, 13, 10.5)
    caixa(ax, 41.0, 40.5, 61.0, 49.0, "3D splats",
          ["training with", "3D Gaussian Splatting"], AZUL, 13, 10.5)
    caixa(ax, 17.0, 31.5, 37.0, 39.5, "SAM",
          ["2D segmentation", "masks"], LARANJA, 13, 10.5)
    caixa(ax, 41.0, 31.5, 61.0, 39.5, "DEVA",
          ["consistent IDs", "across images"], LARANJA, 13, 10.5)
    caixa(ax, 65.0, 31.5, 96.5, 49.0, r"2D $\rightarrow$ 3D projection",
          ["voting across views", "defines the initial label", "of each splat"], LARANJA, 13.5, 10.5)

    seta(ax, 13.0, 44.7, 16.8, 44.7)
    seta(ax, 37.0, 44.7, 40.8, 44.7)
    seta(ax, 61.0, 44.7, 64.8, 44.7)
    seta(ax, 13.0, 35.5, 16.8, 35.5)
    seta(ax, 37.0, 35.5, 40.8, 35.5)
    seta(ax, 61.0, 35.5, 64.8, 35.5)

    # --- ligacao entre as etapas ---
    ax.plot([80.7, 80.7], [31.5, 28.0], color="#111111", linewidth=1.7)
    ax.plot([80.7, 1.0], [28.0, 28.0], color="#111111", linewidth=1.7)
    ax.plot([1.0, 1.0], [28.0, 11.5], color="#111111", linewidth=1.7)
    seta(ax, 1.0, 11.5, 4.3, 11.5)

    # --- Etapa 2 ---
    ax.add_patch(FancyBboxPatch((1.5, 1.5), 97, 22.5, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="#f7f7f7", ec="#b8b8b8", linewidth=1.4))
    ax.text(3.5, 21.6, "Stage 2: graph learning and segmentation",
            fontsize=13.5, fontweight="bold", va="center")

    blocos = [
        ("Filtering", ["opacity, scale", "and outliers"], VERDE),
        ("Graph", ["KNN, distance", r"and color ($\Delta$E)"], VERDE),
        ("GAT or GCN", ["contrastive loss", r"$\rightarrow$ embeddings"], VERDE),
        ("HDBSCAN", ["clustering of", "the embeddings"], ROXO),
        ("Propagation", ["KNN of the noise", "splats"], ROXO),
        ("Segmentation", ["3D instances and", "2D masks"], BRANCO),
    ]
    x = 4.5
    largura, espaco = 14.8, 1.3
    for i, (titulo, linhas, cores) in enumerate(blocos):
        caixa(ax, x, 5.0, x + largura, 18.0, titulo, linhas, cores, 12.5, 10.5)
        if i < len(blocos) - 1:
            seta(ax, x + largura, 11.5, x + largura + espaco, 11.5)
        x += largura + espaco

    fig.tight_layout(pad=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURAS, f"pipeline_metodo_en.{ext}"), dpi=160, facecolor="white")
    plt.close(fig)
    print("gerado: pipeline_metodo_en.png / .pdf")


ESPACO_ARQ = 1.55


def arquiteturas():
    fig, ax = plt.subplots(figsize=(9.4, 11.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 118); ax.axis("off")

    ax.text(26, 114, "(a) GAT with 4 heads", ha="center", fontsize=17, fontweight="bold")
    ax.text(26, 108.5, "learned attention coefficients", ha="center", fontsize=11.5, style="italic",
            color="#333333")
    ax.text(74, 114, "(b) GCN with 2 layers", ha="center", fontsize=17, fontweight="bold")
    ax.text(74, 108.5, r"aggregation weighted by $w_{ij}$", ha="center", fontsize=11.5, style="italic",
            color="#333333")

    # coluna (a)
    caixa(ax, 6, 92, 46, 102, "Input", [r"$N\times 6$  (xyz + L*a*b*)"], CINZA, 13.5, 11.5, espaco=ESPACO_ARQ)
    caixa(ax, 6, 70, 46, 84, "GATConv  (4 heads)",
          [r"$6 \rightarrow 32 \times 4 = 128$", "BatchNorm, ELU, Dropout"], LARANJA, 13.5, 11.5, espaco=ESPACO_ARQ)
    caixa(ax, 6, 48, 46, 62, "GATConv  (2 heads)",
          [r"$128 \rightarrow 64 \times 2 = 128$", "BatchNorm, ELU, Dropout"], LARANJA, 13.5, 11.5, espaco=ESPACO_ARQ)
    caixa(ax, 6, 28, 46, 39, "Linear", [r"$128 \rightarrow 32$"], VERDE, 13.5, 11.5, espaco=ESPACO_ARQ)
    caixa(ax, 6, 8, 46, 19, "L2 normalization",
          [r"embedding $\mathbf{z}_i \in \mathbb{R}^{32}$"], CINZA, 13.5, 11.5, espaco=ESPACO_ARQ)
    for y0, y1 in [(92, 84), (70, 62), (48, 39), (28, 19)]:
        seta(ax, 26, y0, 26, y1)

    # coluna (b)
    caixa(ax, 54, 92, 94, 102, "Input", [r"$N\times 6$  (xyz + L*a*b*)"], CINZA, 13.5, 11.5, espaco=ESPACO_ARQ)
    caixa(ax, 54, 70, 94, 84, "GCNConv",
          [r"$6 \rightarrow 64$", "LayerNorm, ELU, Dropout"], AZUL, 13.5, 11.5, espaco=ESPACO_ARQ)
    caixa(ax, 54, 48, 94, 62, "GCNConv", [r"$64 \rightarrow 128$", "LayerNorm"], AZUL, 13.5, 11.5, espaco=ESPACO_ARQ)
    caixa(ax, 54, 28, 94, 39, "Linear", [r"$128 \rightarrow 32$"], VERDE, 13.5, 11.5, espaco=ESPACO_ARQ)
    caixa(ax, 54, 8, 94, 19, "L2 normalization",
          [r"embedding $\mathbf{z}_i \in \mathbb{R}^{32}$"], CINZA, 13.5, 11.5, espaco=ESPACO_ARQ)
    seta(ax, 74, 92, 74, 84)
    seta(ax, 74, 70, 74, 62)

    # conexao residual
    ax.plot([74, 74], [70, 66.5], color="#111111", linewidth=1.7)
    ax.add_patch(Circle((74, 67.7), 0.7, fc="#111111", ec="#111111"))
    ax.plot([74, 49.5], [67.7, 67.7], color="#111111", linewidth=1.7)
    ax.plot([49.5, 49.5], [67.7, 44.0], color="#111111", linewidth=1.7)
    ax.plot([49.5, 72.0], [44.0, 44.0], color="#111111", linewidth=1.7)
    ax.text(48.0, 56.0, r"Linear $64 \rightarrow 128$", rotation=90, ha="center", va="center",
            fontsize=11.5)
    ax.add_patch(Circle((74, 44.0), 1.9, fc="white", ec="#111111", linewidth=1.6))
    ax.text(74, 44.0, "+", ha="center", va="center", fontsize=14)
    seta(ax, 74, 48, 74, 46.0)
    ax.text(77.5, 44.0, "ELU, Dropout", ha="left", va="center", fontsize=11.5)
    seta(ax, 74, 42.1, 74, 39.0)
    seta(ax, 74, 28, 74, 19)

    fig.tight_layout(pad=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURAS, f"arquiteturas_gnn_en.{ext}"), dpi=160, facecolor="white")
    plt.close(fig)
    print("gerado: arquiteturas_gnn_en.png / .pdf")


if __name__ == "__main__":
    pipeline()
    arquiteturas()
