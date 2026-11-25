"""
Figuras qualitativas multivista do Experimento 4.

Para cada cena, escolhe N câmeras de treino bem espalhadas pela trajetória e
monta uma grade com a imagem original, a segmentação do nosso método
(images_coverage da repetição preservada do Experimento 4) e a do Gaussian
Grouping (objects_pred do render.py), uma linha por vista.

As vistas candidatas são as câmeras cujo eixo óptico aponta para o centro de
foco da cena (até 20 graus) e que não estão entre as 30% mais borradas; entre
elas, a amostragem do ponto mais distante espalha as vistas ao redor da cena. Os frames
anotados no ground truth ficam de fora, porque já aparecem nas figuras de
comparação da tese.

Uso: python gerar_figuras_multivistas.py [--n_vistas 5]
"""
import argparse
import glob
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = "/home/victor/Documentos/gaussian_grouping/gaussian-grouping"
GT = "/home/victor/Downloads/lear_ovs/lerf_ovs/label"
CENAS = ["teatime", "ramen", "figurines", "waldo_kitchen"]


def centro_de_foco(pos, frente):
    """Ponto mais próximo, por mínimos quadrados, de todos os eixos ópticos."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p, d in zip(pos, frente):
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ p
    return np.linalg.solve(A, b)


def nitidez(caminho):
    cinza = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(cinza, cv2.CV_64F).var()


def escolher_vistas(cena, cams, excluir, n, max_angulo=20.0, frac_borradas=0.3):
    pos = np.array([c["position"] for c in cams], dtype=float)
    frente = np.array([np.array(c["rotation"])[:, 2] for c in cams], dtype=float)
    frente /= np.linalg.norm(frente, axis=1, keepdims=True)
    centro = centro_de_foco(pos, frente)

    # ângulo entre o eixo óptico e a direção câmera -> centro de foco
    para_centro = centro - pos
    para_centro /= np.linalg.norm(para_centro, axis=1, keepdims=True)
    angulo = np.degrees(np.arccos(np.clip((frente * para_centro).sum(1), -1, 1)))
    nit = np.array([nitidez(f"{RAIZ}/data/{cena}/images/{c['img_name']}.jpg") for c in cams])

    ok = (angulo <= max_angulo) & (nit >= np.quantile(nit, frac_borradas))
    ok &= np.array([c["img_name"] not in excluir for c in cams])
    idx = np.flatnonzero(ok)

    # espalha as vistas pela posição ao redor do centro de foco
    feat = pos[idx] - centro
    feat /= np.linalg.norm(feat, axis=1).max() + 1e-8
    escolhidas = [int(np.argmax(nit[idx]))]  # começa pela vista mais nítida
    dist = np.linalg.norm(feat - feat[escolhidas[0]], axis=1)
    while len(escolhidas) < n:
        i = int(np.argmax(dist))
        escolhidas.append(i)
        dist = np.minimum(dist, np.linalg.norm(feat - feat[i], axis=1))
    print(f"   {cena}: {len(idx)}/{len(cams)} câmeras candidatas (olhando para o centro, não borradas)")
    return sorted((cams[idx[i]] for i in escolhidas), key=lambda c: c["img_name"])


def gerar(cena, n):
    cams = json.load(open(f"{RAIZ}/output/{cena}/cameras.json"))
    anotados = {os.path.basename(p)[:-5] for p in glob.glob(f"{GT}/{cena}/*.json")}
    vistas = escolher_vistas(cena, cams, anotados, n)

    colunas = ["Imagem original", "Nosso método", "Gaussian Grouping"]
    h, w = cv2.imread(f"{RAIZ}/data/{cena}/images/{vistas[0]['img_name']}.jpg").shape[:2]
    fig, axes = plt.subplots(n, 3, figsize=(9, 9 / 3 * h / w * n + 0.4))
    for lin, c in enumerate(vistas):
        f = c["img_name"]
        paths = [
            f"{RAIZ}/data/{cena}/images/{f}.jpg",
            f"{RAIZ}/experimentos_finais/experimento4/{cena}/gat/images_coverage/{f}.png",
            f"{RAIZ}/output/{cena}/train/ours_30000/objects_pred/{c['id']:05d}.png",
        ]
        for col, p in enumerate(paths):
            ax = axes[lin, col]
            im = cv2.imread(p)
            assert im is not None, f"não encontrado: {p}"
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("0.75"); sp.set_linewidth(0.8)
            if lin == 0:
                ax.set_title(colunas[col], fontsize=11, pad=6)
        axes[lin, 0].set_ylabel(f.replace("frame_", "frame "), fontsize=8)

    fig.subplots_adjust(left=0.04, right=0.995, top=1 - 0.35 / fig.get_figheight(),
                        bottom=0.005, wspace=0.04, hspace=0.04)
    saida = f"{RAIZ}/figuras/multivistas_{cena}.png"
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"{cena:<14} {[v['img_name'] for v in vistas]} -> {saida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_vistas", type=int, default=5)
    args = ap.parse_args()
    for cena in CENAS:
        gerar(cena, args.n_vistas)
