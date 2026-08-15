import json
import cv2
import numpy as np
from pathlib import Path

import json
import cv2
import numpy as np
from pathlib import Path

class RestrictedMetrics2D:
    """
    Avaliação de segmentação 2D baseada na COR MODAL (tom mais frequente)
    restrita à área do Ground Truth (GT).
    """
    def __init__(self, json_path, scale_factor=0.25):
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        self.scale_factor = scale_factor
        self.gt_objects, self.shape = self._load_gt_objects()
        
    def _load_gt_objects(self):
        orig_h = self.json_data["info"]["height"]
        orig_w = self.json_data["info"]["width"]
        
        target_h = int(round(orig_h * self.scale_factor))
        target_w = int(round(orig_w * self.scale_factor))
        
        gt_objects = []
        for obj in self.json_data["objects"]:
            category = obj.get("category", "object")
            pts = np.array(obj["segmentation"], dtype=np.float32)
            pts_scaled = (pts * self.scale_factor).astype(np.int32).reshape((-1, 1, 2))
            
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_scaled], color=1)
            
            gt_objects.append({
                "category": category,
                "mask": mask == 1,
                "area": np.sum(mask == 1)
            })
        return gt_objects, (target_h, target_w)
    
    def evaluate(self, pred_img_path, color_tolerance=30.0, visualize=True):
        pred_bgr = cv2.imread(pred_img_path)
        if pred_bgr is None:
            raise FileNotFoundError(f"Imagem não encontrada: {pred_img_path}")
        
        H, W = self.shape
        if pred_bgr.shape[:2] != (H, W):
            pred_bgr = cv2.resize(pred_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
        
        results = []
        
        # Imagem para visualização: vamos pintar as áreas que ENTRARAM na métrica
        visual = pred_bgr.copy()
        combined_pred_mask = np.zeros((H, W), dtype=bool)
        
        print(f"\n🎯 AVALIAÇÃO ANCORADA NA MODA DOMINANTE (Tolerância = {color_tolerance}):")
        print("=" * 70)
        
        for obj in self.gt_objects:
            cat_name = obj["category"]
            gt_mask = obj["mask"]
            
            if np.sum(gt_mask) == 0:
                continue
            
            pred_pixels_in_gt = pred_bgr[gt_mask]
            
            if len(pred_pixels_in_gt) == 0:
                results.append({
                    "category": cat_name, "precision": 0.0, "recall": 0.0,
                    "f1": 0.0, "iou": 0.0, "pixel_acc": 0.0, "area": obj["area"], "covered": False
                })
                continue

            # --- FILTRO DE PIXELS PRETOS / MUITO ESCUROS ---
            # Define um limiar de brilho (soma dos canais BGR ou intensidade do pixel)
            # Pixels onde B+G+R <= 30 serão considerados fundo/preto e descartados da moda
            black_threshold = 30  
            pixel_brightness = np.sum(pred_pixels_in_gt, axis=1)
            valid_color_pixels = pred_pixels_in_gt[pixel_brightness > black_threshold]

            # Se todos os pixels do objeto forem pretos, usamos a lista original para evitar erro de array vazio
            if len(valid_color_pixels) > 0:
                target_pixels_for_mode = valid_color_pixels
            else:
                target_pixels_for_mode = pred_pixels_in_gt

            # Cor da Moda Dominante (calculada APENAS com os pixels não-pretos)
            quantized_in_gt = (target_pixels_for_mode // 8) * 8
            unique_colors, counts = np.unique(quantized_in_gt, axis=0, return_counts=True)
            dominant_quantized_color = unique_colors[np.argmax(counts)]
            
            in_dominant_bin = np.all(quantized_in_gt == dominant_quantized_color, axis=1)
            mode_color_bgr = np.mean(target_pixels_for_mode[in_dominant_bin], axis=0)
            # ------------------------------------------------
            
            # Distância Euclidiana BGR
            color_diff = np.linalg.norm(pred_bgr.astype(np.float32) - mode_color_bgr, axis=2)
            
            # Apenas pixels aceitos pela cor E dentro do GT
            pred_cluster_mask = (color_diff <= color_tolerance) & gt_mask
            
            # Guarda para visualização
            combined_pred_mask = combined_pred_mask | pred_cluster_mask
            
            # Métricas
            tp = np.logical_and(pred_cluster_mask, gt_mask).sum()
            fp = np.logical_and(pred_cluster_mask, ~gt_mask).sum()
            fn = np.logical_and(~pred_cluster_mask, gt_mask).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            union = tp + fp + fn
            iou = tp / union if union > 0 else 0.0
            pixel_acc = tp / np.sum(gt_mask) if np.sum(gt_mask) > 0 else 0.0
            
            results.append({
                "category": cat_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "pixel_acc": pixel_acc,
                "area": obj["area"],
                "covered": tp > 0
            })

        # --- DESENHA AÁREA CONSIDERADA ---
        if visualize:
            # 1. Pinta a área aceita na métrica de VERDE LIME brilhante (0, 255, 0)
            overlay = visual.copy()
            overlay[combined_pred_mask] = (255, 0, 0)
            
            # 2. Aplica uma transparência suave (alpha blend) para ainda conseguir ver a imagem embaixo
            cv2.addWeighted(overlay, 0.5, visual, 0.5, 0, visual)
            
            # 3. Desenha também os contornos exatos dessas áreas aceitas
            contours, _ = cv2.findContours(combined_pred_mask.astype(np.uint8), 
                                          cv2.RETR_TREE, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(visual, contours, -1, (0, 255, 0), 1)
            
            cv2.imwrite("debug_evaluated_areas.png", visual)
            print("\n📸 Visualização das áreas avaliadas salva em 'debug_evaluated_areas.png'")
        
        # Print dos resultados agregados
        valid_results = [r for r in results if r["area"] > 0]
        if valid_results:
            mean_precision = np.mean([r["precision"] for r in valid_results])
            mean_recall = np.mean([r["recall"] for r in valid_results])
            mean_f1 = np.mean([r["f1"] for r in valid_results])
            mean_iou = np.mean([r["iou"] for r in valid_results])
            mean_pixel_acc = np.mean([r["pixel_acc"] for r in valid_results])
            covered_objects = sum(1 for r in valid_results if r["covered"])
            total_objects = len(valid_results)
        else:
            mean_precision = mean_recall = mean_f1 = mean_iou = mean_pixel_acc = 0.0
            covered_objects = total_objects = 0
            
        print("=" * 70)
        print(f"📊 RESULTADOS AGREGADOS:")
        print(f"  • Precision Média (no GT):    {mean_precision*100:.2f}%")
        print(f"  • Recall Médio:               {mean_recall*100:.2f}%")
        print(f"  • F1-score Médio:             {mean_f1:.4f}")
        print(f"  • mIoU Restrito:              {mean_iou:.4f}")
        print(f"  • Pixel Accuracy Médio:       {mean_pixel_acc*100:.2f}%")
        print(f"  • Objetos Detectados:         {covered_objects}/{total_objects} ({100*covered_objects/total_objects:.1f}%)")
        
        print("\n📋 DETALHAMENTO POR OBJETO:")
        print("-" * 70)
        print(f"{'Categoria':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'IoU':>8} {'Área':>8}")
        print("-" * 70)
        for r in valid_results:
            print(f"{r['category']:<20} {r['precision']*100:>7.1f}% {r['recall']*100:>7.1f}% "
                  f"{r['f1']:>7.3f} {r['iou']:>7.3f} {r['area']:>8.0f}")

        return results

if __name__ == "__main__":
    json_path = "/home/victor/Downloads/lear_ovs/lerf_ovs/label/teatime/frame_00002.json"
    #pred_path = "/home/victor/Documentos/gaussian_grouping/gaussian-grouping/output_seg/gat/frame_00002.png"
    pred_path = "/home/victor/Documentos/gaussian_grouping/gaussian-grouping/output/teatime/train/ours_30000/objects_pred/00001.png"

    
    evaluator = RestrictedMetrics2D(json_path, scale_factor=0.25)
    evaluator.evaluate(pred_path, color_tolerance=50.0)