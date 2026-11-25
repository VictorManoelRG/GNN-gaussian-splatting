"""
Consistência multivista das segmentações 2D, sem ground truth.

Pergunta: uma mesma região 3D recebe a mesma instância em todas as vistas em
que aparece? Para responder, as splats do modelo treinado são usadas como
"sondas" 3D:

  1. Cada splat opaca (opacidade >= --min_opacity) tem o centro projetado em
     todas as câmeras de treino, com a mesma projeção e o mesmo z-buffer por
     pixel da votação de rótulos iniciais (train_gnn_final_version.py): só
     conta a splat que está na superfície visível daquele pixel.
  2. Em cada vista, lê-se a instância predita no pixel onde a splat caiu, na
     máscara do nosso método (images_coverage) e na do Gaussian Grouping
     (objects_pred). Pixel preto é fundo (sem instância) nos dois.
  3. Observações em pixels de borda (vizinhança 3x3 não uniforme em qualquer
     uma das duas máscaras) são descartadas, para que um erro de um pixel na
     fronteira não conte como inconsistência. As MESMAS observações
     (splat, vista) são usadas nos dois métodos.
  4. Para cada splat com pelo menos --min_obs observações:
       consistência = nº de vistas com o rótulo mais frequente / nº de vistas
     e, quando o rótulo mais frequente é um objeto, as vistas discordantes são
     separadas em "desaparecimento" (a vista prediz fundo) e "troca" (a vista
     prediz outro objeto).

Uso: python consistencia_multivista.py [--cenas teatime ramen ...]
"""
import argparse
import json
import os

import cv2
import numpy as np
from plyfile import PlyData

RAIZ = "/home/victor/Documentos/gaussian_grouping/gaussian-grouping"
STRIDE = 1 << 24  # rótulo = cor RGB empacotada em 24 bits


def carregar_splats(cena, min_opacity):
    v = PlyData.read(f"{RAIZ}/output/{cena}/point_cloud/iteration_30000/point_cloud.ply")["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    opac = 1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1) * 0.28209479 + 0.5
    ok = opac >= min_opacity
    return xyz[ok], np.clip(dc[ok], 0, 1)


def empacotar(img_bgr):
    return ((img_bgr[..., 2].astype(np.int64) << 16)
            | (img_bgr[..., 1].astype(np.int64) << 8)
            | img_bgr[..., 0].astype(np.int64))


def uniforme_3x3(img_bgr):
    k = np.ones((3, 3), np.uint8)
    return np.all(cv2.erode(img_bgr, k) == cv2.dilate(img_bgr, k), axis=2)


def projetar(xyz, cam, W, H, depth_tol, depth_rel_tol):
    """Índices das splats visíveis e seus pixels (u, v), com z-buffer por pixel."""
    R_c2w = np.array(cam["rotation"]); C = np.array(cam["position"])
    p = (xyz - C) @ R_c2w  # = R_c2w^T (X - C), coordenadas de câmera
    z = p[:, 2]
    fx = cam["fx"] * W / cam["width"]; fy = cam["fy"] * H / cam["height"]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(fx * p[:, 0] / z + W / 2).astype(np.int64)
        v = np.floor(fy * p[:, 1] / z + H / 2).astype(np.int64)
    idx = np.flatnonzero((z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H))
    if idx.size == 0:
        return idx, idx, idx
    pix = v[idx] * W + u[idx]
    ordem = np.lexsort((z[idx], pix))
    idx, pix = idx[ordem], pix[ordem]
    uniq, primeiro = np.unique(pix, return_index=True)
    z_frente = z[idx][primeiro][np.searchsorted(uniq, pix)]
    frente = z[idx] <= z_frente + np.maximum(z_frente * depth_rel_tol, depth_tol)
    idx = idx[frente]
    return idx, u[idx], v[idx]


def resumir(chaves, contagens, min_obs):
    g = chaves // STRIDE
    rot = chaves % STRIDE
    n = np.bincount(g, weights=contagens)
    # rótulo mais frequente por splat: ordena por (splat, contagem)
    ordem = np.lexsort((contagens, g))
    g_s, c_s, r_s = g[ordem], contagens[ordem], rot[ordem]
    ultimo = np.r_[np.flatnonzero(np.diff(g_s)), g_s.size - 1]
    moda_g, moda_c, moda_r = g_s[ultimo], c_s[ultimo], r_s[ultimo]
    fundo = np.bincount(g[rot == 0], weights=contagens[rot == 0], minlength=n.size)

    nt = n[moda_g]
    ok = nt >= min_obs
    moda_g, moda_c, moda_r, nt = moda_g[ok], moda_c[ok], moda_r[ok], nt[ok]
    cons = moda_c / nt
    obj = moda_r != 0
    desap = fundo[moda_g[obj]] / nt[obj]
    troca = (nt[obj] - moda_c[obj] - fundo[moda_g[obj]]) / nt[obj]
    return {
        "splats_avaliadas": int(ok.sum()),
        "obs_por_splat": float(nt.mean()),
        "consistencia_media": float(cons.mean()),
        "splats_100pct_consistentes": float((cons == 1).mean()),
        "splats_com_objeto_como_moda": float(obj.mean()),
        "consistencia_media_splats_de_objeto": float(cons[obj].mean()),
        "desaparecimento_splats_de_objeto": float(desap.mean()),
        "troca_splats_de_objeto": float(troca.mean()),
    }


def avaliar(cena, args):
    cams = json.load(open(f"{RAIZ}/output/{cena}/cameras.json"))
    xyz, cor = carregar_splats(cena, args.min_opacity)
    chaves = {"nosso": [], "gg": []}
    corr = []

    for i, cam in enumerate(cams):
        nome = cam["img_name"]
        nosso = cv2.imread(f"{RAIZ}/experimentos_finais/experimento4/{cena}/gat/images_coverage/{nome}.png")
        gg = cv2.imread(f"{RAIZ}/output/{cena}/train/ours_30000/objects_pred/{cam['id']:05d}.png")
        assert nosso is not None and gg is not None and nosso.shape == gg.shape, nome
        H, W = nosso.shape[:2]

        idx, u, v = projetar(xyz, cam, W, H, args.depth_tol, args.depth_rel_tol)
        valido = uniforme_3x3(nosso)[v, u] & uniforme_3x3(gg)[v, u]
        idx, u, v = idx[valido], u[valido], v[valido]
        chaves["nosso"].append(idx * STRIDE + empacotar(nosso)[v, u])
        chaves["gg"].append(idx * STRIDE + empacotar(gg)[v, u])

        # checagem da projeção: cor base das splats x foto no mesmo pixel
        if i % 10 == 0:
            foto = cv2.imread(f"{RAIZ}/data/{cena}/images/{nome}.jpg")
            if foto is not None and foto.shape[:2] == (H, W):
                corr.append(np.corrcoef(cor[idx].mean(1), foto[v, u].mean(1) / 255)[0, 1])

    res = {"cena": cena, "cameras": len(cams), "splats_opacas": int(len(xyz)),
           "corr_projecao_foto": float(np.nanmean(corr))}
    for metodo, lst in chaves.items():
        k, c = np.unique(np.concatenate(lst), return_counts=True)
        res[metodo] = resumir(k, c.astype(np.float64), args.min_obs)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cenas", nargs="+", default=["teatime", "ramen", "figurines", "waldo_kitchen"])
    ap.add_argument("--min_opacity", type=float, default=0.5)
    ap.add_argument("--min_obs", type=int, default=5)
    ap.add_argument("--depth_tol", type=float, default=0.05)
    ap.add_argument("--depth_rel_tol", type=float, default=0.015)
    ap.add_argument("--saida", default=f"{RAIZ}/experimentos_finais/experimento4/consistencia_multivista")
    args = ap.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    todos = []
    for cena in args.cenas:
        r = avaliar(cena, args)
        todos.append(r)
        json.dump(r, open(f"{args.saida}/{cena}.json", "w"), indent=2)
        print(json.dumps(r, indent=2))
    json.dump({"parametros": vars(args), "resultados": todos},
              open(f"{args.saida}/todas.json", "w"), indent=2)
