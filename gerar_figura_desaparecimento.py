"""
Figura de inconsistência multivista: um objeto que o Gaussian Grouping segmenta
em algumas vistas e deixa como fundo em outras.

Usa as mesmas sondas 3D de consistencia_multivista.py (splats opacas projetadas
com z-buffer, observações de borda descartadas). As splats são agrupadas pelo
rótulo mais frequente que o Gaussian Grouping lhes dá; para cada grupo e cada
vista, mede-se a fração das splats visíveis que o GG prediz como fundo e a
fração que o nosso método mantém no seu rótulo mais frequente. O exemplo
escolhido é o grupo com mais vistas nos dois extremos (GG segmenta >= 80% das
splats / GG prediz fundo em >= 70%) e com o nosso método estável (>= 80%) em
todas elas. A figura mostra duas vistas de cada extremo, com a região do objeto
marcada pelo retângulo que envolve as splats projetadas.

Uso: python gerar_figura_desaparecimento.py [--cenas teatime ...] [--cena_figura teatime]
"""
import argparse
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from consistencia_multivista import RAIZ, carregar_splats, empacotar, projetar, uniforme_3x3

MIN_SPLATS_VISTA = 40


def moda_por_splat(g, rot, n_splats):
    chave = g * (1 << 24) + rot
    k, c = np.unique(chave, return_counts=True)
    gk, rk = k // (1 << 24), k % (1 << 24)
    ordem = np.lexsort((c, gk))
    gk, rk = gk[ordem], rk[ordem]
    ultimo = np.r_[np.flatnonzero(np.diff(gk)), gk.size - 1]
    moda = np.full(n_splats, -1, dtype=np.int64)
    moda[gk[ultimo]] = rk[ultimo]
    return moda


def observar(cena):
    cams = json.load(open(f"{RAIZ}/output/{cena}/cameras.json"))
    xyz, _ = carregar_splats(cena, 0.5)
    vistas = []
    for cam in cams:
        nome = cam["img_name"]
        nosso = cv2.imread(f"{RAIZ}/experimentos_finais/experimento4/{cena}/gat/images_coverage/{nome}.png")
        gg = cv2.imread(f"{RAIZ}/output/{cena}/train/ours_30000/objects_pred/{cam['id']:05d}.png")
        H, W = nosso.shape[:2]
        idx, u, v = projetar(xyz, cam, W, H, 0.05, 0.015)
        ok = uniforme_3x3(nosso)[v, u] & uniforme_3x3(gg)[v, u]
        idx, u, v = idx[ok], u[ok], v[ok]
        vistas.append(dict(cam=cam, idx=idx, u=u, v=v,
                           nosso=empacotar(nosso)[v, u], gg=empacotar(gg)[v, u]))
    return cams, xyz, vistas


def escolher_exemplo(cena, xyz, vistas):
    g = np.concatenate([x["idx"] for x in vistas])
    moda_gg = moda_por_splat(g, np.concatenate([x["gg"] for x in vistas]), len(xyz))
    moda_nosso = moda_por_splat(g, np.concatenate([x["nosso"] for x in vistas]), len(xyz))

    melhor = None
    for obj in np.unique(moda_gg[moda_gg > 0]):
        membros = moda_gg == obj
        linhas = []
        for iv, x in enumerate(vistas):
            m = membros[x["idx"]]
            if m.sum() < MIN_SPLATS_VISTA:
                continue
            frac_obj = (x["gg"][m] == obj).mean()
            frac_fundo = (x["gg"][m] == 0).mean()
            estavel = (x["nosso"][m] == moda_nosso[x["idx"][m]]).mean()
            fundo_nosso = (x["nosso"][m] == 0).mean()
            linhas.append((iv, m.sum(), frac_obj, frac_fundo, estavel, fundo_nosso))
        presentes = [l for l in linhas if l[2] >= 0.8 and l[4] >= 0.8 and l[5] <= 0.1]
        ausentes = [l for l in linhas if l[3] >= 0.7 and l[4] >= 0.8 and l[5] <= 0.1]
        if len(presentes) >= 2 and len(ausentes) >= 2:
            score = min(len(presentes), len(ausentes))
            if melhor is None or score > melhor[0]:
                melhor = (score, obj, presentes, ausentes, int(membros.sum()))
    if melhor is None:
        print(f"{cena:<14} nenhum exemplo encontrado")
        return None
    score, obj, presentes, ausentes, n = melhor
    print(f"{cena:<14} objeto GG {obj}: {n} splats | {len(presentes)} vistas segmentado, "
          f"{len(ausentes)} vistas como fundo")
    return melhor


def espalhadas(linhas, k):
    """k vistas com mais splats visíveis, evitando frames vizinhos."""
    linhas = sorted(linhas, key=lambda l: -l[1])
    escolha = []
    for l in linhas:
        if all(abs(l[0] - e[0]) >= 8 for e in escolha):
            escolha.append(l)
        if len(escolha) == k:
            break
    return escolha


def desenhar(cena, xyz, vistas, exemplo):
    _, obj, presentes, ausentes, _ = exemplo
    sel = [(l, True) for l in espalhadas(presentes, 2)] + [(l, False) for l in espalhadas(ausentes, 2)]
    sel.sort(key=lambda s: s[0][0])

    colunas = ["Imagem original", "Nosso método", "Gaussian Grouping"]
    h, w = cv2.imread(f"{RAIZ}/data/{cena}/images/{vistas[0]['cam']['img_name']}.jpg").shape[:2]
    n = len(sel)
    fig, axes = plt.subplots(n, 3, figsize=(9, 9 / 3 * h / w * n + 0.4))
    for lin, (l, presente) in enumerate(sel):
        x = vistas[l[0]]
        cam = x["cam"]
        membros = np.flatnonzero(
            np.isin(x["idx"], np.flatnonzero(moda_gg_global == obj)))
        uu, vv = x["u"][membros], x["v"][membros]
        x0, x1 = np.percentile(uu, [2, 98]); y0, y1 = np.percentile(vv, [2, 98])
        paths = [
            f"{RAIZ}/data/{cena}/images/{cam['img_name']}.jpg",
            f"{RAIZ}/experimentos_finais/experimento4/{cena}/gat/images_coverage/{cam['img_name']}.png",
            f"{RAIZ}/output/{cena}/train/ours_30000/objects_pred/{cam['id']:05d}.png",
        ]
        for col, p in enumerate(paths):
            ax = axes[lin, col]
            im = cv2.imread(p)
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.add_patch(plt.Rectangle((x0 - 3, y0 - 3), x1 - x0 + 6, y1 - y0 + 6,
                                       fill=False, edgecolor="yellow", linewidth=1.6))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("0.75"); sp.set_linewidth(0.8)
            if lin == 0:
                ax.set_title(colunas[col], fontsize=11, pad=6)
        estado = "GG segmenta" if presente else "GG: fundo"
        axes[lin, 0].set_ylabel(f"{cam['img_name'].replace('frame_', 'frame ')}\n({estado})", fontsize=8)

    fig.subplots_adjust(left=0.06, right=0.995, top=1 - 0.35 / fig.get_figheight(),
                        bottom=0.005, wspace=0.04, hspace=0.04)
    saida = f"{RAIZ}/figuras/consistencia_multivista_{cena}.png"
    fig.savefig(saida, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"   frames: {[vistas[l[0]]['cam']['img_name'] for l, _ in sel]} -> {saida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cenas", nargs="+", default=["teatime", "ramen", "figurines", "waldo_kitchen"])
    args = ap.parse_args()
    for cena in args.cenas:
        cams, xyz, vistas = observar(cena)
        g = np.concatenate([x["idx"] for x in vistas])
        moda_gg_global = moda_por_splat(g, np.concatenate([x["gg"] for x in vistas]), len(xyz))
        exemplo = escolher_exemplo(cena, xyz, vistas)
        if exemplo is not None:
            desenhar(cena, xyz, vistas, exemplo)
