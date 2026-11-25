#!/usr/bin/env python3
"""
color_segment_fixer.py

Problema: imagens de segmentacao "pintadas" onde cada objeto deveria ter UMA
cor, mas na pratica os pixels de um mesmo objeto variam levemente de cor
(ruido, blend com o fundo, anti-aliasing, textura de pincel etc).

Solucao: em vez de comparar cores exatas, os pixels sao agrupados por
SIMILARIDADE de cor (clustering) no espaco HSV, e cada cluster resultante
recebe uma cor fixa e maximamente distinta das demais. Se voce passar uma
PASTA com varios frames, o clustering e ajustado usando pixels de todos os
frames ao mesmo tempo -> o mesmo objeto (ex.: "o urso") fica com a MESMA cor
final em todos os frames, nao so dentro de uma imagem.

Uso basico:
    python3 color_segment_fixer.py --input frame_00001.png --output out/
    python3 color_segment_fixer.py --input pasta_de_frames/ --output out/

Parametros uteis:
    --method meanshift|kmeans   (meanshift = auto-detecta o numero de cores)
    --k N                       (numero de clusters, so para --method kmeans)
    --quantile 0.05..0.3        (granularidade do meanshift; menor = mais clusters)
    --min-fraction 0.003        (funde clusters menores que 0.3% dos pixels no
                                 vizinho mais proximo, para remover ruido)
    --keep-background-white     (mantem o cluster de fundo/baixa saturacao em branco)
"""

import argparse
import glob
import json
import os
import sys
import colorsys

import numpy as np
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans


# --------------------------------------------------------------------------
# 1. Extracao de features de cor robusta a variacao de tom/brilho
# --------------------------------------------------------------------------
def image_to_hsv_features(path):
    """Le uma imagem e retorna (rgb_array, features Nx3, shape (h, w))."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    hsv = np.array(img.convert("HSV")).astype(np.float32)

    H = hsv[:, :, 0] / 255.0 * 2 * np.pi   # matiz em radianos (circular)
    S = hsv[:, :, 1] / 255.0               # saturacao 0..1
    V = hsv[:, :, 2] / 255.0               # valor/brilho 0..1

    # Embedding que respeita a circularidade do matiz: pixels de baixa
    # saturacao (brancos/cinzas/fundo) caem perto do centro (0,0,V)
    # independente do matiz, e nao criam clusters falsos por "hue ruidoso".
    X = np.stack([S * np.cos(H), S * np.sin(H), V], axis=-1).reshape(-1, 3)
    return np.array(img), X, (h, w)


# --------------------------------------------------------------------------
# 2. Ajuste do clustering (uma vez, usando pixels de TODOS os frames)
# --------------------------------------------------------------------------
def fit_clusters(all_features, method="meanshift", k=8, quantile=0.15,
                  sample_size=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = all_features.shape[0]
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        sample = all_features[idx]
    else:
        sample = all_features

    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        model.fit(sample)
        centers = model.cluster_centers_
    else:  # meanshift (numero de clusters automatico)
        bw = estimate_bandwidth(sample, quantile=quantile,
                                 n_samples=min(3000, len(sample)),
                                 random_state=seed)
        if bw <= 0:
            bw = 0.1
        model = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True)
        model.fit(sample)
        centers = model.cluster_centers_

    return centers


def assign_labels(features, centers):
    """Atribui cada pixel ao centro de cluster mais proximo (em blocos, p/ nao estourar RAM)."""
    labels = np.empty(features.shape[0], dtype=np.int32)
    block = 200_000
    for i in range(0, features.shape[0], block):
        chunk = features[i:i + block]
        d = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels[i:i + block] = d.argmin(1)
    return labels


def merge_small_clusters(centers, counts, min_fraction):
    """Funde clusters muito pequenos (provavel ruido) no cluster mais proximo maior."""
    total = counts.sum()
    keep_mask = counts >= max(1, min_fraction * total)
    if keep_mask.all() or keep_mask.sum() == 0:
        return np.arange(len(centers)), centers

    keep_idx = np.where(keep_mask)[0]
    drop_idx = np.where(~keep_mask)[0]

    remap = np.arange(len(centers))
    for d in drop_idx:
        dists = ((centers[keep_idx] - centers[d]) ** 2).sum(-1)
        remap[d] = keep_idx[dists.argmin()]
    for k_ in keep_idx:
        remap[k_] = k_

    # renumerar 0..M-1
    uniq = sorted(set(remap.tolist()))
    final_map = {old: new for new, old in enumerate(uniq)}
    remap_final = np.array([final_map[r] for r in remap])
    new_centers = centers[uniq]
    return remap_final, new_centers


# --------------------------------------------------------------------------
# 3. Paleta de cores fixas e maximamente distintas
# --------------------------------------------------------------------------
def build_palette(centers, keep_background_white=True):
    n = len(centers)
    xy_norm = np.linalg.norm(centers[:, :2], axis=1)  # ~saturacao media do cluster
    bg_idx = int(np.argmin(xy_norm)) if (keep_background_white and xy_norm.min() < 0.12) else -1

    palette = np.zeros((n, 3), dtype=np.uint8)
    others = [i for i in range(n) if i != bg_idx]
    for c, idx in enumerate(others):
        hue = c / max(1, len(others))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        palette[idx] = [int(r * 255), int(g * 255), int(b * 255)]
    if bg_idx >= 0:
        palette[bg_idx] = [255, 255, 255]
    return palette, bg_idx


# --------------------------------------------------------------------------
# 4. Pipeline principal
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                     help="Caminho de uma imagem ou de uma pasta com varias imagens")
    ap.add_argument("--output", required=True, help="Pasta de saida")
    ap.add_argument("--method", choices=["meanshift", "kmeans"], default="meanshift")
    ap.add_argument("--k", type=int, default=8, help="numero de clusters (so --method kmeans)")
    ap.add_argument("--quantile", type=float, default=0.15,
                     help="granularidade do meanshift (menor = mais clusters)")
    ap.add_argument("--min-fraction", type=float, default=0.003,
                     help="funde clusters menores que essa fracao de pixels (ruido)")
    ap.add_argument("--sample-size", type=int, default=20000,
                     help="pixels amostrados (por eficiencia) para ajustar o clustering")
    ap.add_argument("--keep-background-white", action="store_true", default=True)
    args = ap.parse_args()

    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.png")) +
                        glob.glob(os.path.join(args.input, "*.jpg")) +
                        glob.glob(os.path.join(args.input, "*.jpeg")))
    else:
        paths = [args.input]

    if not paths:
        sys.exit(f"Nenhuma imagem encontrada em {args.input}")

    os.makedirs(args.output, exist_ok=True)

    print(f"[1/4] Lendo {len(paths)} imagem(ns) e extraindo features HSV...")
    rgb_list, feat_list, shape_list = [], [], []
    for p in paths:
        rgb, feats, shape = image_to_hsv_features(p)
        rgb_list.append(rgb)
        feat_list.append(feats)
        shape_list.append(shape)

    all_features = np.concatenate(feat_list, axis=0)

    print(f"[2/4] Ajustando clustering ({args.method}) em pixels de todos os frames...")
    centers = fit_clusters(all_features, method=args.method, k=args.k,
                            quantile=args.quantile, sample_size=args.sample_size)
    print(f"      -> {len(centers)} clusters brutos encontrados")

    print("[3/4] Atribuindo cada pixel ao cluster mais proximo e removendo ruido...")
    all_labels = assign_labels(all_features, centers)
    counts = np.bincount(all_labels, minlength=len(centers))
    remap, centers = merge_small_clusters(centers, counts, args.min_fraction)
    all_labels = remap[all_labels]
    n_final = len(centers)
    print(f"      -> {n_final} clusters finais (depois de fundir ruido)")

    palette, bg_idx = build_palette(centers, args.keep_background_white)

    print("[4/4] Salvando imagens quantizadas e legenda...")
    offset = 0
    final_counts = np.zeros(n_final, dtype=np.int64)
    for p, rgb, (h, w) in zip(paths, rgb_list, shape_list):
        n_pix = h * w
        labels = all_labels[offset:offset + n_pix]
        offset += n_pix
        final_counts += np.bincount(labels, minlength=n_final)

        out_img = palette[labels].reshape(h, w, 3).astype(np.uint8)
        out_name = os.path.join(args.output, os.path.basename(p))
        Image.fromarray(out_img).save(out_name)
        print(f"      {os.path.basename(p)} -> {out_name}")

    # Legenda: swatch de cada cor final + contagem de pixels no dataset todo
    legend_path = os.path.join(args.output, "legend.json")
    legend = [
        {
            "cluster_id": int(i),
            "color_rgb": palette[i].tolist(),
            "color_hex": "#%02x%02x%02x" % tuple(palette[i]),
            "pixel_count": int(final_counts[i]),
            "is_background": bool(i == bg_idx),
        }
        for i in range(n_final)
    ]
    with open(legend_path, "w") as f:
        json.dump(legend, f, indent=2, ensure_ascii=False)
    print(f"      legenda salva em {legend_path}")

    # imagem de swatches (paleta visual)
    sw = 60
    swatch_img = np.zeros((sw, sw * n_final, 3), dtype=np.uint8)
    for i in range(n_final):
        swatch_img[:, i * sw:(i + 1) * sw] = palette[i]
    Image.fromarray(swatch_img).save(os.path.join(args.output, "palette_preview.png"))

    print("\nConcluido.")


if __name__ == "__main__":
    main()