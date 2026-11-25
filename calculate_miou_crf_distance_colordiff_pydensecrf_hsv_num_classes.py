import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral


# ---------------------------------------------------------------------------
# 1. FUNÇÕES AUXILIARES DE ESPAÇO DE COR
# ---------------------------------------------------------------------------

def _rgb_to_hsv_norm(img_rgb_uint8):
    """Converte RGB uint8 -> HSV normalizado em [0,1]."""
    hsv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] /= 179.0  # OpenCV usa H em [0,179]
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0
    return hsv


def _hsv_to_circular_features(hsv_norm, hue_weight=4.0, sat_weight=1.5, val_weight=0.5):
    """
    Transforma HSV em espaço euclidiano circular: (cos(2*pi*H), sin(2*pi*H), S, V).
    Aplica pesos para priorizar a identidade do Hue em relação a saturação e brilho.
    """
    H, S, V = hsv_norm[..., 0], hsv_norm[..., 1], hsv_norm[..., 2]
    angle = 2 * np.pi * H
    feats = np.stack([
        np.cos(angle) * hue_weight,
        np.sin(angle) * hue_weight,
        S * sat_weight,
        V * val_weight,
    ], axis=-1)
    return feats


# ---------------------------------------------------------------------------
# 2. ESTIMATIVA DINÂMICA DE K E GERAÇÃO DAS UNÁRIAS
# ---------------------------------------------------------------------------

def estimate_optimal_k(feats_flat, k_min=3, k_max=15, max_samples=5000):
    """
    Estima dinamicamente o K ideal para o frame atual avaliando
    a separabilidade dos clusters via Calinski-Harabasz Score.
    """
    # Filtra fundo totalmente escuro/preto para não distorcer a contagem
    nonzero_mask = np.linalg.norm(feats_flat, axis=1) > 0.1
    valid_feats = feats_flat[nonzero_mask]
    
    if len(valid_feats) == 0:
        return k_min

    # Subamostragem aleatória para execução rápida (poucos ms)
    if len(valid_feats) > max_samples:
        idx = np.random.choice(len(valid_feats), max_samples, replace=False)
        sample = valid_feats[idx]
    else:
        sample = valid_feats

    best_k = k_min
    best_score = -1.0

    for k in range(k_min, min(k_max + 1, len(sample))):
        km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(sample)
        score = calinski_harabasz_score(sample, km.labels_)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def known_colors_to_unaries(rendered_mask_rgb, known_colors_rgb,
                              hue_weight=4.0, sat_weight=1.5, val_weight=0.5,
                              temperature=0.05):
    """Gera unárias caso você já saiba de antemão a paleta exata de cores."""
    H, W, _ = rendered_mask_rgb.shape
    K = len(known_colors_rgb)

    hsv_pix = _hsv_to_circular_features(
        _rgb_to_hsv_norm(rendered_mask_rgb), hue_weight, sat_weight, val_weight
    )

    known_arr = np.array(known_colors_rgb, dtype=np.uint8).reshape(1, K, 3)
    hsv_known = _hsv_to_circular_features(
        _rgb_to_hsv_norm(known_arr), hue_weight, sat_weight, val_weight
    ).reshape(K, 4)

    pix_flat = hsv_pix.reshape(-1, 4)
    dists = np.linalg.norm(pix_flat[:, None, :] - hsv_known[None, :, :], axis=2)

    logits = -dists / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

    return probs.reshape(H, W, K), np.array(known_colors_rgb, dtype=np.uint8)


def color_mask_to_unaries_hsv(rendered_mask_rgb, n_clusters=None,
                                hue_weight=4.0, sat_weight=1.5, val_weight=0.5,
                                temperature=0.05):
    """
    Clusterização no espaço HSV circular. Se n_clusters for None,
    calcula automaticamente o número de classes do frame.
    """
    H, W, _ = rendered_mask_rgb.shape
    hsv_pix = _hsv_to_circular_features(
        _rgb_to_hsv_norm(rendered_mask_rgb), hue_weight, sat_weight, val_weight
    )
    feats = hsv_pix.reshape(-1, 4).astype(np.float32)

    if n_clusters is None:
        n_clusters = estimate_optimal_k(feats, k_min=3, k_max=15)
        print(f"   [Auto-K] K estimado dinamicamente: {n_clusters} classes.")

    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42).fit(feats)
    centers_feat = kmeans.cluster_centers_

    dists = np.linalg.norm(feats[:, None, :] - centers_feat[None, :, :], axis=2)
    dists = dists.reshape(H, W, n_clusters)

    logits = -dists / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

    # Identificação da cor média de cada cluster para debug
    labels = kmeans.labels_.reshape(H, W)
    rep_colors = np.zeros((n_clusters, 3), dtype=np.uint8)
    flat_rgb = rendered_mask_rgb.reshape(-1, 3)
    for c in range(n_clusters):
        m = kmeans.labels_ == c
        if m.any():
            rep_colors[c] = flat_rgb[m].mean(axis=0).astype(np.uint8)

    return probs, rep_colors


# ---------------------------------------------------------------------------
# 3. DENSE CRF (REFINAMENTO DE BORDAS COM GUIA HSV)
# ---------------------------------------------------------------------------

def apply_pydensecrf(image_rgb, unary_probs, n_iter=20,
                       sxy_spatial=1, compat_spatial=2,
                       sxy_bilateral=10, srgb_bilateral=15, compat_bilateral=10,
                       bilateral_guide="hsv"):
    H, W, K = unary_probs.shape

    d = dcrf.DenseCRF2D(W, H, K)

    probs_transposed = np.ascontiguousarray(np.transpose(unary_probs, (2, 0, 1)))
    unary = unary_from_softmax(probs_transposed)
    d.setUnaryEnergy(unary)

    pairwise_spatial = create_pairwise_gaussian(
        sdims=(sxy_spatial, sxy_spatial),
        shape=(H, W)
    )
    d.addPairwiseEnergy(
        pairwise_spatial,
        compat=compat_spatial,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    if bilateral_guide == "hsv":
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        angle = 2 * np.pi * (hsv[..., 0] / 179.0)
        guide = np.stack([
            (np.cos(angle) * 0.5 + 0.5) * 255.0,
            (np.sin(angle) * 0.5 + 0.5) * 255.0,
            hsv[..., 1],
            hsv[..., 2],
        ], axis=-1).astype(np.uint8)
        schan = (srgb_bilateral,) * 4
    else:
        guide = image_rgb
        schan = (srgb_bilateral, srgb_bilateral, srgb_bilateral)

    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(sxy_bilateral, sxy_bilateral),
        schan=schan,
        img=guide,
        chdim=2
    )
    d.addPairwiseEnergy(
        pairwise_bilateral,
        compat=compat_bilateral,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    Q = d.inference(n_iter)
    map_sol = np.argmax(Q, axis=0).reshape((H, W))
    return map_sol


# ---------------------------------------------------------------------------
# 4. SEPARAÇÃO E COLORIZAÇÃO DAS INSTÂNCIAS
# ---------------------------------------------------------------------------

def split_instances_by_dilation(labels_map, bg_label=0, kernel_size=1, min_area=20):
    H, W = labels_map.shape
    instance_map = np.zeros((H, W), dtype=np.int32)
    current_instance_id = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    unique_classes = np.unique(labels_map)

    for c in unique_classes:
        if c == bg_label:
            continue

        class_mask = (labels_map == c).astype(np.uint8)
        dilated_mask = cv2.dilate(class_mask, kernel, iterations=1)
        num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(dilated_mask, connectivity=8)

        for comp_id in range(1, num_labels):
            if stats[comp_id, cv2.CC_STAT_AREA] < min_area:
                continue

            comp_mask_dilated = (comp_labels == comp_id)
            original_pixels = comp_mask_dilated & (class_mask == 1)
            instance_map[original_pixels] = current_instance_id
            current_instance_id += 1

    return instance_map, current_instance_id - 1


def generate_random_palette(num_colors):
    np.random.seed(42)
    palette = np.random.randint(50, 255, size=(num_colors + 1, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0] # Fundo em preto
    return palette


# ---------------------------------------------------------------------------
# 5. EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    MASK_PATH = "/home/victor/Documentos/gaussian_grouping/gaussian-grouping/exp_res/gat_2heads/gat/images/frame_00006.png"
    RGB_GUIDE_PATH = None
    KNOWN_COLORS_RGB = None  # Mantenha None para usar a busca dinâmica por Auto-K

    if not os.path.exists(MASK_PATH):
        print(f"Erro: Arquivo não encontrado em: {MASK_PATH}")
        return

    mask_bgr = cv2.imread(MASK_PATH)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    if RGB_GUIDE_PATH and os.path.exists(RGB_GUIDE_PATH):
        guide_rgb = cv2.cvtColor(cv2.imread(RGB_GUIDE_PATH), cv2.COLOR_BGR2RGB)
    else:
        guide_rgb = mask_rgb.copy()

    print("1/3 Estimando número de classes e gerando unárias...")
    if KNOWN_COLORS_RGB is not None:
        unary_probs, class_colors = known_colors_to_unaries(
            mask_rgb, KNOWN_COLORS_RGB,
            hue_weight=4.0, sat_weight=1.5, val_weight=0.5, temperature=0.05
        )
    else:
        unary_probs, class_colors = color_mask_to_unaries_hsv(
            mask_rgb, 
            n_clusters=None,  # Detecção automática de K
            hue_weight=4.0, 
            sat_weight=1.5, 
            val_weight=0.5, 
            temperature=0.05
        )

    print("2/3 Refinando com PyDenseCRF (Guia HSV)...")
    refined_labels = apply_pydensecrf(
        guide_rgb,
         unary_probs,
        # n_iter=20,
        # sxy_spatial=1,
        # compat_spatial=2,
        # sxy_bilateral=10,
        # srgb_bilateral=15,
        # compat_bilateral=10,
        bilateral_guide="hsv",
    )

    bg_class = np.argmin(np.linalg.norm(class_colors.astype(np.float32), axis=1))

    print("3/3 Isolando componentes conectadas/instâncias...")
    instance_map, total_instances = split_instances_by_dilation(
        refined_labels,
        bg_label=bg_class,
        kernel_size=1,
        min_area=20
    )

    palette = generate_random_palette(total_instances)
    instance_colored = palette[instance_map]

    output_dir = os.path.dirname(MASK_PATH)
    output_path = os.path.join(output_dir, "frame_00002_pydensecrf_autoK.png")
    cv2.imwrite(output_path, cv2.cvtColor(instance_colored, cv2.COLOR_RGB2BGR))

    print(f"\nConcluído com sucesso! {total_instances} instâncias segmentadas com cor uniforme.")
    print(f"Resultado salvo em: {output_path}")


if __name__ == "__main__":
    main()