import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral


# ---------------------------------------------------------------------------
# 1. FUNÇÕES AUXILIARES DE ESPAÇO DE COR E POSIÇÃO
# ---------------------------------------------------------------------------

def _rgb_to_hsv_norm(img_rgb_uint8):
    hsv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] /= 179.0
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0
    return hsv


def _hsv_to_spatial_features(hsv_norm, spatial_weight=0.2, hue_weight=4.0, sat_weight=1.5, val_weight=0.5):
    """
    Extrai atributos de cor (HSV circular) e inclui coordenadas espaciais (X, Y)
    normalizadas para evitar que regiões distantes compartilhem a mesma classe.
    """
    H, W, _ = hsv_norm.shape
    angle = 2 * np.pi * hsv_norm[..., 0]

    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    grid_y /= H
    grid_x /= W

    feats = np.stack([
        np.cos(angle) * hue_weight,
        np.sin(angle) * hue_weight,
        hsv_norm[..., 1] * sat_weight,
        hsv_norm[..., 2] * val_weight,
        grid_x * spatial_weight,
        grid_y * spatial_weight
    ], axis=-1)
    return feats


def _get_ignore_mask(hsv_norm, dark_value_thresh=0.12, white_value_thresh=0.92, white_sat_thresh=0.12):
    """
    Identifica pixels que NÃO pertencem a nenhum objeto real:
    - pixels escuros (fundo preto)
    - pixels claros e pouco saturados (fundo/espaço branco, ex: (255,255,255))
    Esses pixels são excluídos do clustering e tratados como classe de fundo dedicada.
    """
    dark_pixels = hsv_norm[..., 2] < dark_value_thresh
    white_pixels = (hsv_norm[..., 2] > white_value_thresh) & (hsv_norm[..., 1] < white_sat_thresh)
    return dark_pixels | white_pixels


# ---------------------------------------------------------------------------
# 2. ESTIMATIVA DINÂMICA DE K E GERAÇÃO DAS UNÁRIAS
# ---------------------------------------------------------------------------

def estimate_optimal_k(feats_flat, k_min=3, k_max=10, max_samples=5000):
    if len(feats_flat) == 0:
        return k_min

    if len(feats_flat) > max_samples:
        idx = np.random.choice(len(feats_flat), max_samples, replace=False)
        sample = feats_flat[idx]
    else:
        sample = feats_flat

    best_k = k_min
    best_score = -1.0

    for k in range(k_min, min(k_max + 1, len(sample))):
        km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(sample)
        score = calinski_harabasz_score(sample, km.labels_)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def color_mask_to_unaries_hsv(rendered_mask_rgb, n_clusters=None, spatial_weight=0.2,
                               temperature=0.05, dark_value_thresh=0.12,
                               white_value_thresh=0.92, white_sat_thresh=0.12,
                               k_max=10):
    """
    Gera probabilidades unárias por classe. Fundo (preto E branco) vira uma classe
    reservada extra (índice n_clusters), montada ANTES do KMeans rodar, então nunca
    se mistura com as cores reais dos objetos.

    Retorna:
        probs_full   -> (H, W, n_clusters+1)
        rep_colors   -> (n_clusters+1, 3) cor representativa de cada classe (última = preto/fundo)
        bg_class_idx -> índice da classe de fundo dentro de rep_colors/probs_full
    """
    H, W, _ = rendered_mask_rgb.shape
    hsv_norm = _rgb_to_hsv_norm(rendered_mask_rgb)

    ignore_pixels = _get_ignore_mask(hsv_norm, dark_value_thresh, white_value_thresh, white_sat_thresh)
    ignore_flat = ignore_pixels.reshape(-1)

    spatial_feats = _hsv_to_spatial_features(hsv_norm, spatial_weight=spatial_weight)
    feats = spatial_feats.reshape(-1, 6).astype(np.float32)
    valid_feats = feats[~ignore_flat]

    if len(valid_feats) == 0:
        # imagem inteira é fundo — caso degenerado
        probs_full = np.zeros((H, W, 1), dtype=np.float32)
        probs_full[..., 0] = 1.0
        return probs_full, np.array([[0, 0, 0]], dtype=np.uint8), 0

    if n_clusters is None:
        n_clusters = estimate_optimal_k(valid_feats, k_min=3, k_max=k_max)

    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42).fit(valid_feats)
    centers_feat = kmeans.cluster_centers_

    # distância de TODOS os pixels (inclusive fundo) aos centros, só para montar as probs
    dists = np.linalg.norm(feats[:, None, :] - centers_feat[None, :, :], axis=2)
    dists = dists.reshape(H, W, n_clusters)

    logits = -dists / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

    # cor representativa de cada cluster, calculada SÓ com pixels válidos
    rep_colors = np.zeros((n_clusters, 3), dtype=np.uint8)
    flat_rgb = rendered_mask_rgb.reshape(-1, 3)
    valid_rgb = flat_rgb[~ignore_flat]
    for c in range(n_clusters):
        m = kmeans.labels_ == c
        if m.any():
            rep_colors[c] = valid_rgb[m].mean(axis=0).astype(np.uint8)

    # classe de fundo dedicada (preto/branco), sempre pintada de preto no resultado final
    bg_class_idx = n_clusters
    probs_full = np.zeros((H, W, n_clusters + 1), dtype=np.float32)
    probs_full[..., :n_clusters] = probs
    probs_full[ignore_pixels, :] = 0.0
    probs_full[ignore_pixels, bg_class_idx] = 1.0

    rep_colors_full = np.vstack([rep_colors, np.array([[0, 0, 0]], dtype=np.uint8)])

    return probs_full, rep_colors_full, bg_class_idx


# ---------------------------------------------------------------------------
# 3. DENSE CRF (REFINAMENTO DE BORDAS COM GUIA HSV)
# ---------------------------------------------------------------------------

def apply_pydensecrf(image_rgb, unary_probs, n_iter=10,
                      sxy_spatial=3, compat_spatial=3,
                      sxy_bilateral=40, srgb_bilateral=20, compat_bilateral=8):
    H, W, K = unary_probs.shape
    d = dcrf.DenseCRF2D(W, H, K)

    probs_transposed = np.ascontiguousarray(np.transpose(unary_probs, (2, 0, 1)))
    unary = unary_from_softmax(probs_transposed)
    d.setUnaryEnergy(unary)

    pairwise_spatial = create_pairwise_gaussian(
        sdims=(sxy_spatial, sxy_spatial), shape=(H, W)
    )
    d.addPairwiseEnergy(
        pairwise_spatial, compat=compat_spatial,
        kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    angle = 2 * np.pi * (hsv[..., 0] / 179.0)
    guide = np.stack([
        (np.cos(angle) * 0.5 + 0.5) * 255.0,
        (np.sin(angle) * 0.5 + 0.5) * 255.0,
        hsv[..., 1],
        hsv[..., 2],
    ], axis=-1).astype(np.uint8)

    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(sxy_bilateral, sxy_bilateral), schan=(srgb_bilateral,) * 4,
        img=guide, chdim=2
    )
    d.addPairwiseEnergy(
        pairwise_bilateral, compat=compat_bilateral,
        kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    Q = d.inference(n_iter)
    return np.argmax(Q, axis=0).reshape((H, W))


# ---------------------------------------------------------------------------
# 4. PROCESSAMENTO DE INSTÂNCIAS DESCONECTADAS
# ---------------------------------------------------------------------------

def convert_to_instance_mask(labels_map, bg_label, min_area=150):
    """
    Gera IDs de instância únicos para objetos fisicamente separados por componentes conectados.
    """
    H, W = labels_map.shape
    instance_map = np.zeros((H, W), dtype=np.int32)
    current_id = 1
    unique_classes = np.unique(labels_map)

    for c in unique_classes:
        if c == bg_label:
            continue
        mask = (labels_map == c).astype(np.uint8)
        num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for comp_id in range(1, num_labels):
            if stats[comp_id, cv2.CC_STAT_AREA] < min_area:
                continue
            instance_map[comp_labels == comp_id] = current_id
            current_id += 1

    return instance_map, current_id - 1


def generate_distinct_palette(num_colors):
    np.random.seed(42)
    palette = np.random.randint(40, 255, size=(num_colors + 1, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Fundo em preto
    return palette


# ---------------------------------------------------------------------------
# 5. PROCESSAMENTO EM LOTE
# ---------------------------------------------------------------------------

def process_directory(input_dir, output_folder_name="pydensecrf_results", mode="instance",
                       spatial_weight=0.2, n_clusters=None, min_area=150,
                       dark_value_thresh=0.12, white_value_thresh=0.92, white_sat_thresh=0.12):
    """
    Parâmetros:
    - mode: "instance" (separa objetos distantes com cores únicas) ou "semantic" (mantém cores por classe).
    - spatial_weight: Peso da coordenada (X,Y) no KMeans. Valores altos fragmentam demais — use ~0.2.
    - n_clusters: fixe manualmente se souber quantos objetos existem na cena (recomendado).
    - min_area: componentes conectados menores que isso (em pixels) são descartados como ruído.
    - dark_value_thresh / white_value_thresh / white_sat_thresh: controlam o que é considerado
      "fundo" (preto e branco) e portanto excluído da segmentação.
    """
    if not os.path.exists(input_dir):
        print(f"Erro: O diretório de entrada não existe: {input_dir}")
        return

    output_dir = os.path.join(input_dir, output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(input_dir, f))
    ])

    if not image_files:
        print(f"Nenhuma imagem encontrada em: {input_dir}")
        return

    print(f"Processando {len(image_files)} imagens (Modo: {mode.upper()})...")

    for idx, filename in enumerate(image_files, start=1):
        mask_path = os.path.join(input_dir, filename)
        mask_bgr = cv2.imread(mask_path)
        if mask_bgr is None:
            continue

        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        hsv_norm = _rgb_to_hsv_norm(mask_rgb)
        ignore_pixels = _get_ignore_mask(hsv_norm, dark_value_thresh, white_value_thresh, white_sat_thresh)

        # 1. Unárias com informação de cor + espaço (fundo já isolado numa classe própria)
        unary_probs, class_colors, bg_class_idx = color_mask_to_unaries_hsv(
            mask_rgb, n_clusters=n_clusters, spatial_weight=spatial_weight,
            dark_value_thresh=dark_value_thresh, white_value_thresh=white_value_thresh,
            white_sat_thresh=white_sat_thresh
        )

        # 2. PyDenseCRF Refinamento
        refined_labels = apply_pydensecrf(mask_rgb, unary_probs)

        # 3. Trava de segurança: garante que pixels originalmente pretos/brancos
        #    continuam classificados como fundo mesmo se o CRF tentar espalhar cor neles
        refined_labels[ignore_pixels] = bg_class_idx

        # 4. Mapeamento de Cores
        if mode == "instance":
            instance_map, total_instances = convert_to_instance_mask(
                refined_labels, bg_label=bg_class_idx, min_area=min_area
            )
            palette = generate_distinct_palette(total_instances)
            colored_result = palette[instance_map]
        else:
            colored_result = class_colors[refined_labels]

        # 5. Salvar
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cv2.cvtColor(colored_result, cv2.COLOR_RGB2BGR))
        print(f"[{idx}/{len(image_files)}] Salvo: {filename}")

    print("\nConcluído com sucesso!")


if __name__ == "__main__":
    INPUT_DIR = "/home/victor/Documentos/gaussian_grouping/gaussian-grouping/exp_res/gat_8heads/gat/images"

    # mode="instance" -> Cada objeto espacialmente isolado ganha uma cor única.
    # spatial_weight=0.2 -> Peso da posição (X,Y). Valores altos (ex: 1) fragmentam demais.
    # n_clusters -> defina manualmente (ex: 6) se souber quantos objetos existem na cena.
    process_directory(
        input_dir=INPUT_DIR,
        output_folder_name="pydensecrf_results",
        mode="inctance",
        spatial_weight=0.2,
        n_clusters=None,
        min_area=150,
    )