import json
import cv2
import numpy as np
from pathlib import Path
from scipy.stats import mode

def json_to_individual_masks(json_data, scale_factor=0.25):
    """
    Retorna uma lista de dicionários com a máscara binária individual 
    e a categoria de cada objeto anotado no JSON.
    """
    orig_h = json_data["info"]["height"]
    orig_w = json_data["info"]["width"]
    
    target_h = int(round(orig_h * scale_factor))
    target_w = int(round(orig_w * scale_factor))
    
    gt_objects = []
    
    for obj in json_data["objects"]:
        category = obj.get("category", "object")
        pts = np.array(obj["segmentation"], dtype=np.float32)
        pts_scaled = (pts * scale_factor).astype(np.int32).reshape((-1, 1, 2))
        
        # Máscara individual para ESTE objeto específico
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_scaled], color=1)
        
        gt_objects.append({
            "category": category,
            "mask": mask == 1
        })
        
    return gt_objects, (target_h, target_w)

def evaluate_restricted_masks(pred_img_path, json_path, scale_factor=0.25):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    gt_objects, (target_h, target_w) = json_to_individual_masks(json_data, scale_factor=scale_factor)
    
    pred_bgr = cv2.imread(pred_img_path)
    if pred_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {pred_img_path}")
        
    if pred_bgr.shape[:2] != (target_h, target_w):
        pred_bgr = cv2.resize(pred_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
    # Quantização leve de cor para agrupar variações sutis de compressão
    pred_quant = (pred_bgr // 16) * 16
    
    # Atribui um ID inteiro para cada cor única da sua imagem predita
    flat_img = pred_quant.reshape(-1, 3)
    _, cluster_mask_flat = np.unique(flat_img, axis=0, return_inverse=True)
    pred_clusters = cluster_mask_flat.reshape(target_h, target_w)
    
    # Criar imagem visual de teste
    visual_output = pred_bgr.copy()
    
    ious = []
    precisions = []
    
    print("\n🎯 AVALIAÇÃO RESTRITA ÀS MÁSCARAS DO GT:")
    print("=" * 60)
    
    for obj in gt_objects:
        cat_name = obj["category"]
        gt_mask = obj["mask"]  # Apenas os pixels deste objeto (ex: bear nose)
        
        # Pixels da sua imagem que caem DENTRO da máscara desse objeto
        pred_pixels_in_gt = pred_clusters[gt_mask]
        
        if len(pred_pixels_in_gt) == 0:
            continue
            
        # Qual cor/cluster o seu modelo predominantemente colocou dentro desse polígono?
        dominant_cluster = mode(pred_pixels_in_gt, keepdims=True).mode[0]
        
        # Máscara do cluster dominante predito pelo seu modelo
        pred_cluster_mask = (pred_clusters == dominant_cluster)
        
        # 1. Mask Precision (Purity): Quanto do polígono anotado foi preenchido por essa cor dominante
        correct_pixels = np.logical_and(pred_cluster_mask, gt_mask).sum()
        total_gt_pixels = gt_mask.sum()
        precision = correct_pixels / total_gt_pixels if total_gt_pixels > 0 else 0.0
        
        # 2. IoU Restrito: Interseção / União considerando a cor mapeada
        union_pixels = np.logical_or(pred_cluster_mask, gt_mask).sum()
        iou = correct_pixels / union_pixels if union_pixels > 0 else 0.0
        
        ious.append(iou)
        precisions.append(precision)
        
        print(f"  📌 Objeto: {cat_name:<20} | Cobertura (Precision): {precision*100:5.1f}% | IoU: {iou:.4f}")
        
        # Desenha a borda do GT sobre o render para ver se bateu
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visual_output, contours, -1, (0, 255, 0), 1)

    cv2.imwrite("debug_restricted_masks.png", visual_output)
    
    print("=" * 60)
    print(f"📊 Cobertura Média dos Objetos (Precision): {np.mean(precisions)*100:.2f}%")
    print(f"📊 mIoU Restrito aos Objetos:               {np.mean(ious):.4f}")
    print("📸 Visualização salva em 'debug_restricted_masks.png' (Borda verde = GT)\n")

# Execute para testar
if __name__ == "__main__":
    evaluate_restricted_masks("/home/victor/Documentos/gaussian_grouping/gaussian-grouping/output_seg/tests/waldo_kitchen_tests/gat_waldo_kitchen_post_propagation_KNN/frame_00066.png",
                               "/home/victor/Downloads/lear_ovs/lerf_ovs/label/waldo_kitchen/frame_00066.json", scale_factor=0.25)