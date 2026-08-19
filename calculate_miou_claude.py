"""
Avalia uma segmentação de instância predita (imagem RGB onde cada instância
tem uma cor única -- saída do segment_instances.py) contra um Ground Truth
no formato COCO-like (objects[i].segmentation em polígono ou RLE).

Métricas reportadas (padrão usado em papers de segmentação de instância /
panóptica, ex: Mask R-CNN, Panoptic-FPN, Mask2Former):

  - IoU por par casado (matching húngaro, maximizando IoU)
  - mIoU: média do IoU dos pares casados
  - Precision / Recall / F1 @ IoU >= t, para t em {0.5, 0.75} e a
    varredura COCO t = 0.50, 0.55, ..., 0.95
  - Panoptic Quality (PQ = SQ x RQ), de Kirillov et al. 2019:
        TP = pares com IoU >= 0.5 (matching 1-para-1)
        FP = instâncias preditas sem casamento
        FN = instâncias do GT sem casamento
        SQ = média do IoU dos TP  (Segmentation Quality)
        RQ = TP / (TP + 0.5*FP + 0.5*FN)  (Recognition Quality, = F1)
        PQ = SQ * RQ

Por que PQ e não COCO AP "de verdade": AP integra uma curva
precisão-recall sobre o *confidence score* de cada detecção. As instâncias
aqui vêm de um agrupamento de cor (sem score de confiança por instância),
então cada máscara é tratada como uma "detecção" com confiança fixa —
nesse regime, PQ / mIoU / P-R-F1 por IoU é o que os próprios papers de
panoptic segmentation usam para comparar métodos sem score.

Matching é *class-agnostic* (não usa a categoria do GT), já que a
segmentação de cor não produz rótulo semântico -- ela só separa "isso é um
objeto, isso é outro".

Uso:
    python3 evaluate_segmentation.py \
        --pred resultado_final.png \
        --gt frame_00041.json \
        --gt-image frame_00041.png \
        --out relatorio.json
"""

import argparse
import json
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

try:
    from pycocotools import mask as coco_mask
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


# --------------------------------------------------------------------------
# GT: parsing e rasterização
# --------------------------------------------------------------------------

def _looks_like_point_list(seg):
    """True se seg for uma lista de pontos [x,y] (formato deste dataset),
    em vez do formato COCO padrão (lista de polígonos, cada um uma lista
    plana [x1,y1,x2,y2,...])."""
    if not isinstance(seg, list) or len(seg) == 0:
        return False
    first = seg[0]
    return (
        isinstance(first, list)
        and len(first) == 2
        and all(isinstance(v, (int, float)) for v in first)
    )


def segmentation_to_polygons(seg):
    """Normaliza qualquer uma das variações de formato de polígono pra uma
    lista de arrays Nx2 (um array por parte/polígono do objeto)."""
    if _looks_like_point_list(seg):
        # ex: [[786.0, 2.0], [780.0, 5.0], ...] -> um único polígono
        return [np.array(seg, dtype=np.float32)]

    # formato COCO padrão: lista de polígonos, cada um flat [x1,y1,x2,y2,...]
    if len(seg) > 0 and isinstance(seg[0], (int, float)):
        seg = [seg]
    return [np.array(poly, dtype=np.float32).reshape(-1, 2) for poly in seg]


def polygon_to_mask(polygons, height, width):
    """polygons: lista de arrays Nx2 (já normalizados)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def rle_to_mask(rle, height, width):
    if not HAS_PYCOCOTOOLS:
        raise RuntimeError(
            "GT está em RLE mas pycocotools não está instalado "
            "(pip install pycocotools --break-system-packages)."
        )
    if isinstance(rle.get("counts"), list):
        rle = coco_mask.frPyObjects(rle, height, width)
    return coco_mask.decode(rle).astype(bool)


def load_gt_instances(gt_json_path):
    """
    Retorna: height, width, lista de (category, mask_bool) -- uma entrada
    por objeto do GT (cada objeto pode ter múltiplos polígonos, ex: partes
    desconectadas do mesmo objeto, que são unidas na mesma máscara).
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    info = gt["info"]
    height, width = info["height"], info["width"]

    instances = []
    for obj in gt["objects"]:
        seg = obj["segmentation"]
        if isinstance(seg, dict):  # RLE
            mask = rle_to_mask(seg, height, width)
        else:  # polígono(s), em qualquer uma das variações de formato
            polygons = segmentation_to_polygons(seg)
            mask = polygon_to_mask(polygons, height, width)
        instances.append({
            "category": obj.get("category", "unknown"),
            "mask": mask,
            "area_json": obj.get("area"),
            "iscrowd": obj.get("iscrowd", 0),
        })
    return height, width, instances


# --------------------------------------------------------------------------
# Predição: extrai uma máscara booleana por cor única na imagem
# --------------------------------------------------------------------------

def load_pred_instances(pred_png_path, bg_color=(0, 0, 0), min_area=1):
    bgr = cv2.imread(pred_png_path)
    if bgr is None:
        raise FileNotFoundError(pred_png_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    flat = rgb.reshape(-1, 3)
    colors, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    label_map = inverse.reshape(h, w)

    instances = []
    for idx, color in enumerate(colors):
        if tuple(color) == tuple(bg_color):
            continue
        mask = label_map == idx
        area = int(mask.sum())
        if area < min_area:
            continue
        instances.append({"color": tuple(int(c) for c in color), "mask": mask, "area": area})
    return h, w, instances


# --------------------------------------------------------------------------
# Matching e métricas
# --------------------------------------------------------------------------

def compute_iou_matrix(gt_masks, pred_masks):
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    iou = np.zeros((n_gt, n_pred), dtype=np.float64)
    # pré-calcula áreas
    gt_areas = [m.sum() for m in gt_masks]
    pred_areas = [m.sum() for m in pred_masks]
    for i in range(n_gt):
        gi = gt_masks[i]
        for j in range(n_pred):
            pj = pred_masks[j]
            inter = np.logical_and(gi, pj).sum()
            if inter == 0:
                continue
            union = gt_areas[i] + pred_areas[j] - inter
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou


def hungarian_match(iou_matrix):
    """Matching 1-para-1 maximizando soma de IoU. Retorna lista de (i, j, iou)."""
    if iou_matrix.size == 0:
        return []
    cost = -iou_matrix  # linear_sum_assignment minimiza
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(i), int(j), float(iou_matrix[i, j])) for i, j in zip(row_ind, col_ind)]


def precision_recall_f1_at_iou(matches, n_gt, n_pred, threshold):
    tp = sum(1 for _, _, iou in matches if iou >= threshold)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def panoptic_quality(matches, n_gt, n_pred, threshold=0.5):
    tp_ious = [iou for _, _, iou in matches if iou >= threshold]
    tp = len(tp_ious)
    fp = n_pred - tp
    fn = n_gt - tp
    sq = float(np.mean(tp_ious)) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq
    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


def evaluate(gt_json_path, pred_png_path, gt_image_path=None, bg_color=(0, 0, 0),
             pred_min_area=1, out_path=None):
    h_gt, w_gt, gt_instances = load_gt_instances(gt_json_path)
    h_pred, w_pred, pred_instances = load_pred_instances(pred_png_path, bg_color, pred_min_area)

    if (h_pred, w_pred) != (h_gt, w_gt):
        print(f"[aviso] tamanho da predição ({w_pred}x{h_pred}) difere do GT "
              f"({w_gt}x{h_gt}); redimensionando predição pro tamanho do GT "
              f"(nearest-neighbor, pra não inventar cor nova nas bordas).")
        resized = []
        for inst in pred_instances:
            m = inst["mask"].astype(np.uint8)
            m = cv2.resize(m, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST).astype(bool)
            resized.append({**inst, "mask": m})
        pred_instances = resized

    gt_masks = [inst["mask"] for inst in gt_instances]
    pred_masks = [inst["mask"] for inst in pred_instances]

    iou_matrix = compute_iou_matrix(gt_masks, pred_masks)
    matches = hungarian_match(iou_matrix)  # inclui pares com iou baixo/0, filtramos depois por threshold

    n_gt, n_pred = len(gt_masks), len(pred_masks)

    report = {
        "n_gt_instances": n_gt,
        "n_pred_instances": n_pred,
        "matches_iou": sorted(
            [{"gt_index": i, "gt_category": gt_instances[i]["category"],
              "pred_color": pred_instances[j]["color"], "iou": round(iou, 4)}
             for i, j, iou in matches],
            key=lambda x: -x["iou"],
        ),
    }

    # mIoU sobre os pares casados (inclui os com IoU baixo, é a "qualidade média do matching")
    all_ious = [iou for _, _, iou in matches]
    report["mean_iou_all_matches"] = float(np.mean(all_ious)) if all_ious else 0.0

    # P/R/F1 em vários thresholds
    thresholds = [0.5, 0.75] + [round(0.5 + 0.05 * k, 2) for k in range(10)]  # 0.50..0.95
    thresholds = sorted(set(thresholds))
    prf = [precision_recall_f1_at_iou(matches, n_gt, n_pred, t) for t in thresholds]
    report["precision_recall_f1_by_threshold"] = prf

    # AP-like: média do F1 na varredura 0.50:0.95 (equivalente ao "AP" da COCO
    # mas sem integrar confidence, já que não há score por instância)
    coco_sweep = [x for x in prf if 0.5 <= x["threshold"] <= 0.95 and abs(x["threshold"] * 100 % 5) < 1e-6]
    report["mean_f1_iou_50_95"] = float(np.mean([x["f1"] for x in coco_sweep])) if coco_sweep else 0.0

    # Panoptic Quality
    report["panoptic_quality_iou50"] = panoptic_quality(matches, n_gt, n_pred, threshold=0.5)
    report["panoptic_quality_iou75"] = panoptic_quality(matches, n_gt, n_pred, threshold=0.75)

    print(f"GT: {n_gt} instâncias | Predito: {n_pred} instâncias (cores únicas != fundo)")
    print(f"mIoU (todos os pares casados pelo Hungarian): {report['mean_iou_all_matches']:.4f}")
    for x in prf:
        if x["threshold"] in (0.5, 0.75):
            print(f"  @IoU>={x['threshold']}: precision={x['precision']:.3f} "
                  f"recall={x['recall']:.3f} f1={x['f1']:.3f} "
                  f"(TP={x['tp']} FP={x['fp']} FN={x['fn']})")
    pq50 = report["panoptic_quality_iou50"]
    print(f"Panoptic Quality @IoU>=0.5: PQ={pq50['pq']:.4f}  SQ={pq50['sq']:.4f}  RQ={pq50['rq']:.4f}")
    print(f"F1 médio na varredura COCO IoU 0.50:0.95: {report['mean_f1_iou_50_95']:.4f}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Relatório completo salvo em {out_path}")

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Imagem de segmentação predita (cor única por instância).")
    ap.add_argument("--gt", required=True, help="JSON do ground truth (formato COCO-like: info + objects[].segmentation).")
    ap.add_argument("--gt-image", default=None, help="(opcional) imagem original, só usada se precisar conferir tamanho/rasterizar visualmente.")
    ap.add_argument("--bg-color", nargs=3, type=int, default=[0, 0, 0], help="Cor RGB tratada como fundo na predição.")
    ap.add_argument("--pred-min-area", type=int, default=1, help="Ignora cores da predição com menos pixels que isso.")
    ap.add_argument("--out", default=None, help="Onde salvar o relatório JSON completo.")
    args = ap.parse_args()

    evaluate(
        gt_json_path=args.gt,
        pred_png_path=args.pred,
        gt_image_path=args.gt_image,
        bg_color=tuple(args.bg_color),
        pred_min_area=args.pred_min_area,
        out_path=args.out,
    )