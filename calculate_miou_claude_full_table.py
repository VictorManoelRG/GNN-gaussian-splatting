#!/usr/bin/env python3
"""
Batch evaluator para segmentação de instâncias.
Compara múltiplas imagens entre Ground Truth e Predições.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

try:
    from pycocotools import mask as coco_mask
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


# ============================================================================
# FUNÇÕES DO AVALIADOR ORIGINAL
# ============================================================================

def _looks_like_point_list(seg):
    if not isinstance(seg, list) or len(seg) == 0:
        return False
    first = seg[0]
    return (
        isinstance(first, list)
        and len(first) == 2
        and all(isinstance(v, (int, float)) for v in first)
    )


def segmentation_to_polygons(seg):
    if _looks_like_point_list(seg):
        return [np.array(seg, dtype=np.float32)]

    if len(seg) > 0 and isinstance(seg[0], (int, float)):
        seg = [seg]
    return [np.array(poly, dtype=np.float32).reshape(-1, 2) for poly in seg]


def polygon_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def rle_to_mask(rle, height, width):
    if not HAS_PYCOCOTOOLS:
        raise RuntimeError(
            "GT está em RLE mas pycocotools não está instalado "
            "(pip install pycocotools)."
        )
    if isinstance(rle.get("counts"), list):
        rle = coco_mask.frPyObjects(rle, height, width)
    return coco_mask.decode(rle).astype(bool)


def load_gt_instances(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    info = gt["info"]
    height, width = info["height"], info["width"]

    instances = []
    for obj in gt["objects"]:
        seg = obj["segmentation"]
        if isinstance(seg, dict):  # RLE
            mask = rle_to_mask(seg, height, width)
        else:  # polígono(s)
            polygons = segmentation_to_polygons(seg)
            mask = polygon_to_mask(polygons, height, width)
        instances.append({
            "category": obj.get("category", "unknown"),
            "mask": mask,
            "area_json": obj.get("area"),
            "iscrowd": obj.get("iscrowd", 0),
        })
    return height, width, instances


def load_pred_instances(pred_png_path, bg_color=None, bg_tolerance=15, min_area=30):
    """
    `bg_color=None` autodetecta o fundo como a cor de maior contagem de pixels
    na imagem (robusto a fundo branco, preto, ou qualquer outro). Qualquer
    pixel dentro de `bg_tolerance` (distância euclidiana em 0-255) dessa cor
    também é tratado como fundo, não só o match exato — isso descarta a poeira
    de pixels quase-fundo que sobra do blending alfa nas bordas dos objetos.

    `min_area` (default 30, era 1): sem um mínimo razoável, cada cor distinta
    de 1-2 pixels (ruído de borda/antialiasing) virava uma "instância
    predita" própria, inflando n_pred em ordens de grandeza e afundando
    precision/F1/PQ mesmo quando os matches reais têm IoU alto.
    """
    bgr = cv2.imread(pred_png_path)
    if bgr is None:
        raise FileNotFoundError(f"Arquivo não encontrado: {pred_png_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    flat = rgb.reshape(-1, 3)
    colors, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    label_map = inverse.reshape(h, w)

    if bg_color is None:
        bg_color = colors[np.argmax(counts)]
    bg_color = np.asarray(bg_color, dtype=np.float64)
    dist_to_bg = np.linalg.norm(colors.astype(np.float64) - bg_color, axis=1)

    instances = []
    for idx, color in enumerate(colors):
        if dist_to_bg[idx] <= bg_tolerance:
            continue
        mask = label_map == idx
        area = int(mask.sum())
        if area < min_area:
            continue
        instances.append({"color": tuple(int(c) for c in color), "mask": mask, "area": area})
    return h, w, instances


def compute_iou_matrix(gt_masks, pred_masks):
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    iou = np.zeros((n_gt, n_pred), dtype=np.float64)
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
    if iou_matrix.size == 0:
        return []
    cost = -iou_matrix
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


def evaluate_single(gt_json_path, pred_png_path, gt_image_path=None, bg_color=None,
                    bg_tolerance=15, pred_min_area=30, out_path=None, verbose=True):
    """Versão modificada que retorna o report sem imprimir tabela"""

    h_gt, w_gt, gt_instances = load_gt_instances(gt_json_path)
    h_pred, w_pred, pred_instances = load_pred_instances(
        pred_png_path, bg_color=bg_color, bg_tolerance=bg_tolerance, min_area=pred_min_area
    )

    if (h_pred, w_pred) != (h_gt, w_gt):
        if verbose:
            print(f"[aviso] tamanho da predição ({w_pred}x{h_pred}) difere do GT "
                  f"({w_gt}x{h_gt}); redimensionando...")
        resized = []
        for inst in pred_instances:
            m = inst["mask"].astype(np.uint8)
            m = cv2.resize(m, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST).astype(bool)
            resized.append({**inst, "mask": m})
        pred_instances = resized

    gt_masks = [inst["mask"] for inst in gt_instances]
    pred_masks = [inst["mask"] for inst in pred_instances]

    iou_matrix = compute_iou_matrix(gt_masks, pred_masks)
    matches = hungarian_match(iou_matrix)

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

    # O matching húngaro só cobre min(n_gt, n_pred) pares: quando há menos
    # predições que instâncias no GT, sobram instâncias do GT sem par. Contamos
    # essas como IoU=0 na média (em vez de simplesmente ignorá-las), senão o
    # mIoU fica inflado exatamente nos casos de sub-segmentação — a mesma lógica
    # que já é usada em fn = n_gt - tp no cálculo de precision/recall/F1.
    all_ious = [iou for _, _, iou in matches] + [0.0] * (n_gt - len(matches))
    report["mean_iou_all_matches"] = float(np.mean(all_ious)) if all_ious else 0.0

    thresholds = [0.5, 0.75] + [round(0.5 + 0.05 * k, 2) for k in range(10)]
    thresholds = sorted(set(thresholds))
    prf = [precision_recall_f1_at_iou(matches, n_gt, n_pred, t) for t in thresholds]
    report["precision_recall_f1_by_threshold"] = prf

    coco_sweep = [x for x in prf if 0.5 <= x["threshold"] <= 0.95 and abs(x["threshold"] * 100 % 5) < 1e-6]
    report["mean_f1_iou_50_95"] = float(np.mean([x["f1"] for x in coco_sweep])) if coco_sweep else 0.0

    report["panoptic_quality_iou50"] = panoptic_quality(matches, n_gt, n_pred, threshold=0.5)
    report["panoptic_quality_iou75"] = panoptic_quality(matches, n_gt, n_pred, threshold=0.75)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"📁 Relatório JSON salvo em: {out_path}")

    return report


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

def process_all_pairs(gt_dir, pred_dir, output_json=None, verbose=True,
                       pred_min_area=30, bg_tolerance=15):
    """
    Processa todos os pares de imagens entre o diretório ground_truth e pred.
    """
    
    gt_dir_path = Path(gt_dir)
    pred_dir_path = Path(pred_dir)
    
    if not gt_dir_path.exists():
        print(f"❌ Diretório GT não encontrado: {gt_dir}")
        return None
    if not pred_dir_path.exists():
        print(f"❌ Diretório pred não encontrado: {pred_dir}")
        return None
    
    # Encontrar todas as imagens no ground truth
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    gt_images = []
    
    for ext in image_extensions:
        gt_images.extend(gt_dir_path.glob(f'*{ext}'))
        gt_images.extend(gt_dir_path.glob(f'*{ext.upper()}'))
    
    if not gt_images:
        print(f"⚠️ Nenhuma imagem encontrada em {gt_dir}")
        return None
    
    print(f"📁 Encontradas {len(gt_images)} imagens no ground truth")
    print("=" * 80)
    
    # Estrutura para armazenar todas as métricas
    all_metrics = {
        'per_image': [],
        'aggregated': {}
    }
    
    # Métricas para agregação
    metrics_aggregator = {
        'n_gt': [],
        'n_pred': [],
        'mean_iou': [],
        'precision_50': [],
        'recall_50': [],
        'f1_50': [],
        'precision_75': [],
        'recall_75': [],
        'f1_75': [],
        'pq_50': [],
        'sq_50': [],
        'rq_50': [],
        'pq_75': [],
        'sq_75': [],
        'rq_75': [],
        'mean_f1_50_95': [],
        'tp_50': [],
        'fp_50': [],
        'fn_50': [],
        'tp_75': [],
        'fp_75': [],
        'fn_75': [],
    }
    
    # Processar cada par
    processed_count = 0
    for img_path in gt_images:
        base_name = img_path.stem
        gt_json_path = img_path.with_suffix('.json')
        
        if not gt_json_path.exists():
            if verbose:
                print(f"⚠️ JSON não encontrado para {img_path.name}, pulando...")
            continue
        
        # Tentar encontrar a imagem predita
        pred_img = None
        for ext in image_extensions:
            candidate = pred_dir_path / f"{base_name}{ext}"
            if candidate.exists():
                pred_img = candidate
                break
            candidate = pred_dir_path / f"{base_name}{ext.upper()}"
            if candidate.exists():
                pred_img = candidate
                break
        
        if pred_img is None:
            if verbose:
                print(f"⚠️ Imagem predita não encontrada para {base_name}, pulando...")
            continue
        
        try:
            if verbose:
                print(f"🔄 Processando: {img_path.name} ↔ {pred_img.name}")
            
            report = evaluate_single(
                gt_json_path=str(gt_json_path),
                pred_png_path=str(pred_img),
                gt_image_path=None,
                bg_color=None,
                bg_tolerance=bg_tolerance,
                pred_min_area=pred_min_area,
                out_path=None,
                verbose=False
            )
            
            # Armazenar métricas desta imagem
            image_metrics = {
                'image_name': base_name,
                'gt_file': str(img_path),
                'pred_file': str(pred_img),
                'n_gt_instances': report['n_gt_instances'],
                'n_pred_instances': report['n_pred_instances'],
                'mean_iou_all_matches': report['mean_iou_all_matches'],
                'mean_f1_iou_50_95': report['mean_f1_iou_50_95'],
                'precision_recall_f1_by_threshold': report['precision_recall_f1_by_threshold'],
                'panoptic_quality_iou50': report['panoptic_quality_iou50'],
                'panoptic_quality_iou75': report['panoptic_quality_iou75'],
                'matches_iou': report['matches_iou']
            }
            
            all_metrics['per_image'].append(image_metrics)
            
            # Agregar métricas
            metrics_aggregator['n_gt'].append(report['n_gt_instances'])
            metrics_aggregator['n_pred'].append(report['n_pred_instances'])
            metrics_aggregator['mean_iou'].append(report['mean_iou_all_matches'])
            metrics_aggregator['mean_f1_50_95'].append(report['mean_f1_iou_50_95'])
            
            # Extrair métricas @ 0.5
            p50 = next(x for x in report['precision_recall_f1_by_threshold'] if x['threshold'] == 0.5)
            metrics_aggregator['precision_50'].append(p50['precision'])
            metrics_aggregator['recall_50'].append(p50['recall'])
            metrics_aggregator['f1_50'].append(p50['f1'])
            metrics_aggregator['tp_50'].append(p50['tp'])
            metrics_aggregator['fp_50'].append(p50['fp'])
            metrics_aggregator['fn_50'].append(p50['fn'])
            
            # Extrair métricas @ 0.75
            p75 = next(x for x in report['precision_recall_f1_by_threshold'] if x['threshold'] == 0.75)
            metrics_aggregator['precision_75'].append(p75['precision'])
            metrics_aggregator['recall_75'].append(p75['recall'])
            metrics_aggregator['f1_75'].append(p75['f1'])
            metrics_aggregator['tp_75'].append(p75['tp'])
            metrics_aggregator['fp_75'].append(p75['fp'])
            metrics_aggregator['fn_75'].append(p75['fn'])
            
            # Extrair PQ @ 0.5
            pq50 = report['panoptic_quality_iou50']
            metrics_aggregator['pq_50'].append(pq50['pq'])
            metrics_aggregator['sq_50'].append(pq50['sq'])
            metrics_aggregator['rq_50'].append(pq50['rq'])
            
            # Extrair PQ @ 0.75
            pq75 = report['panoptic_quality_iou75']
            metrics_aggregator['pq_75'].append(pq75['pq'])
            metrics_aggregator['sq_75'].append(pq75['sq'])
            metrics_aggregator['rq_75'].append(pq75['rq'])
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Erro ao processar {base_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calcular médias
    if metrics_aggregator['n_gt']:
        all_metrics['aggregated'] = {
            'total_images': len(metrics_aggregator['n_gt']),
            'total_gt_instances': sum(metrics_aggregator['n_gt']),
            'total_pred_instances': sum(metrics_aggregator['n_pred']),
            'mean_gt_instances_per_image': np.mean(metrics_aggregator['n_gt']),
            'mean_pred_instances_per_image': np.mean(metrics_aggregator['n_pred']),
            'mean_iou_all_matches': np.mean(metrics_aggregator['mean_iou']),
            'mean_f1_iou_50_95': np.mean(metrics_aggregator['mean_f1_50_95']),
            
            # Métricas @ 0.5
            'mean_precision_50': np.mean(metrics_aggregator['precision_50']),
            'mean_recall_50': np.mean(metrics_aggregator['recall_50']),
            'mean_f1_50': np.mean(metrics_aggregator['f1_50']),
            'total_tp_50': sum(metrics_aggregator['tp_50']),
            'total_fp_50': sum(metrics_aggregator['fp_50']),
            'total_fn_50': sum(metrics_aggregator['fn_50']),
            
            # Métricas @ 0.75
            'mean_precision_75': np.mean(metrics_aggregator['precision_75']),
            'mean_recall_75': np.mean(metrics_aggregator['recall_75']),
            'mean_f1_75': np.mean(metrics_aggregator['f1_75']),
            'total_tp_75': sum(metrics_aggregator['tp_75']),
            'total_fp_75': sum(metrics_aggregator['fp_75']),
            'total_fn_75': sum(metrics_aggregator['fn_75']),
            
            # Panoptic Quality @ 0.5
            'mean_pq_50': np.mean(metrics_aggregator['pq_50']),
            'mean_sq_50': np.mean(metrics_aggregator['sq_50']),
            'mean_rq_50': np.mean(metrics_aggregator['rq_50']),
            
            # Panoptic Quality @ 0.75
            'mean_pq_75': np.mean(metrics_aggregator['pq_75']),
            'mean_sq_75': np.mean(metrics_aggregator['sq_75']),
            'mean_rq_75': np.mean(metrics_aggregator['rq_75']),
        }
        
        # Salvar JSON se solicitado
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            print(f"\n📁 Relatório completo salvo em: {output_json}")
        
        # Imprimir resumo
        print_summary(all_metrics['aggregated'])
        print(f"\n✅ Processadas {processed_count} imagens com sucesso!")
        
    else:
        print("❌ Nenhuma imagem foi processada com sucesso!")
        print(f"   Verifique se os arquivos JSON existem no diretório GT")
        print(f"   e se as imagens preditas estão no diretório especificado.")
    
    return all_metrics


def print_summary(agg):
    """Imprime um resumo bonito das métricas agregadas"""
    print("\n" + "=" * 80)
    print("📊 MÉDIAS TOTAIS SOBRE TODAS AS IMAGENS")
    print("=" * 80)
    print(f"• Total de imagens processadas        : {agg['total_images']}")
    print(f"• Total de instâncias GT              : {agg['total_gt_instances']}")
    print(f"• Total de instâncias preditas        : {agg['total_pred_instances']}")
    print(f"• Média de instâncias GT por imagem   : {agg['mean_gt_instances_per_image']:.2f}")
    print(f"• Média de instâncias preditas/imagem : {agg['mean_pred_instances_per_image']:.2f}")
    print(f"• mIoU médio (todos os pares)         : {agg['mean_iou_all_matches']:.4f}")
    print("-" * 80)
    
    print("📌 MÉDIAS @ IoU >= 0.50:")
    print(f"  - Precision média : {agg['mean_precision_50']:.4f}")
    print(f"  - Recall médio    : {agg['mean_recall_50']:.4f}")
    print(f"  - F1-Score médio  : {agg['mean_f1_50']:.4f}")
    print(f"  - TP total        : {agg['total_tp_50']}")
    print(f"  - FP total        : {agg['total_fp_50']}")
    print(f"  - FN total        : {agg['total_fn_50']}")
    print("-" * 80)
    
    print("📌 MÉDIAS @ IoU >= 0.75:")
    print(f"  - Precision média : {agg['mean_precision_75']:.4f}")
    print(f"  - Recall médio    : {agg['mean_recall_75']:.4f}")
    print(f"  - F1-Score médio  : {agg['mean_f1_75']:.4f}")
    print(f"  - TP total        : {agg['total_tp_75']}")
    print(f"  - FP total        : {agg['total_fp_75']}")
    print(f"  - FN total        : {agg['total_fn_75']}")
    print("-" * 80)
    
    print("📌 PANOPTIC QUALITY MÉDIA @ IoU >= 0.50:")
    print(f"  - PQ (Panoptic Quality)    : {agg['mean_pq_50']:.4f}")
    print(f"  - SQ (Segmentation Quality): {agg['mean_sq_50']:.4f}")
    print(f"  - RQ (Recognition Quality) : {agg['mean_rq_50']:.4f}")
    print("-" * 80)
    
    print("📌 PANOPTIC QUALITY MÉDIA @ IoU >= 0.75:")
    print(f"  - PQ (Panoptic Quality)    : {agg['mean_pq_75']:.4f}")
    print(f"  - SQ (Segmentation Quality): {agg['mean_sq_75']:.4f}")
    print(f"  - RQ (Recognition Quality) : {agg['mean_rq_75']:.4f}")
    print("-" * 80)
    
    print("📌 VARREUDURA COCO:")
    print(f"  - F1 Médio (IoU 0.50:0.95)  : {agg['mean_f1_iou_50_95']:.4f}")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Avalia múltiplas imagens comparando GT e predições'
    )
    parser.add_argument('--gt', required=True, 
                        help='Diretório com imagens e JSONs do Ground Truth')
    parser.add_argument('--pred', required=True, 
                        help='Diretório com imagens preditas')
    parser.add_argument('--out', default=None,
                        help='Arquivo JSON para salvar o relatório completo')
    parser.add_argument('--quiet', action='store_true',
                        help='Modo silencioso (menos output)')
    parser.add_argument('--min-area', type=int, default=30,
                        help='Área mínima (px) para uma cor da predição contar como instância '
                             '(default: 30). Evita que ruído de antialiasing de 1-2px vire '
                             'instâncias falsas.')
    parser.add_argument('--bg-tolerance', type=int, default=15,
                        help='Distância de cor (0-255) para tratar um pixel como fundo, além '
                             'do match exato (default: 15). O fundo é autodetectado como a cor '
                             'mais frequente da predição.')

    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 BATCH EVALUATOR - Segmentação de Instâncias")
    print("=" * 80)
    print(f"📂 Ground Truth : {args.gt}")
    print(f"📂 Predições    : {args.pred}")
    if args.out:
        print(f"📄 Saída JSON   : {args.out}")
    print("=" * 80 + "\n")
    
    process_all_pairs(
        gt_dir=args.gt,
        pred_dir=args.pred,
        output_json=args.out,
        verbose=not args.quiet,
        pred_min_area=args.min_area,
        bg_tolerance=args.bg_tolerance,
    )