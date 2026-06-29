import torch
import os
import numpy as np
import cv2
from ultralytics import YOLO
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
import json
from datetime import datetime

class ObjectClassifierMapper:
    def __init__(self, iou_threshold=0.3):
        """
        Mapeia objetos segmentados (do Gaussian Grouping) para classes YOLO
        
        Args:
            iou_threshold: Threshold de IoU para considerar sobreposição válida
        """
        self.iou_threshold = iou_threshold
        self.object_to_yolo_mapping = {}
        self.object_class_counts = defaultdict(lambda: defaultdict(int))
        self.detection_history = []
        
    def compute_iou(self, mask, bbox):
        """Calcula IoU entre máscara segmentada e bounding box"""
        x1, y1, x2, y2 = bbox
        h, w = mask.shape
        
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 1
        
        intersection = np.logical_and(mask, bbox_mask).sum()
        union = np.logical_or(mask, bbox_mask).sum()
        
        if union == 0:
            return 0.0
        return float(intersection / union)  # Converter para float nativo
    
    def process_detection(self, object_id, class_name, confidence, mask, bbox, view_id):
        """Processa uma detecção e atualiza o mapeamento"""
        iou = self.compute_iou(mask, bbox)
        
        if iou >= self.iou_threshold:
            self.object_class_counts[object_id][class_name] += 1
            self.detection_history.append({
                'view_id': int(view_id),  # Converter para int nativo
                'object_id': int(object_id),  # Converter para int nativo
                'class': str(class_name),
                'confidence': float(confidence),
                'iou': float(iou),
                'bbox': [int(x) for x in bbox]  # Converter cada elemento
            })
            return True
        return False
    
    def finalize_mapping(self, min_samples=3, min_confidence=0.5):
        """
        Finaliza o mapeamento baseado nas contagens
        """
        final_mapping = {}
        
        for object_id, class_counts in self.object_class_counts.items():
            if len(class_counts) == 0:
                continue
                
            most_common_class = max(class_counts.items(), key=lambda x: x[1])
            class_name, count = most_common_class
            
            if count >= min_samples:
                total_samples = sum(class_counts.values())
                confidence = count / total_samples
                
                if confidence >= min_confidence:
                    # Converter todos os valores para tipos nativos Python
                    final_mapping[str(object_id)] = {  # Usar string como chave para JSON
                        'class': str(class_name),
                        'confidence': float(confidence),
                        'samples': int(count),
                        'total_samples': int(total_samples),
                        'all_classes': {str(k): int(v) for k, v in class_counts.items()}  # Converter tudo
                    }
                    
                    print(f"✅ Object {object_id} -> {class_name} (conf: {confidence:.2f}, samples: {count}/{total_samples})")
                else:
                    print(f"⚠️ Object {object_id} -> {class_name} (conf: {confidence:.2f} < {min_confidence}, ignorado)")
            else:
                print(f"❌ Object {object_id} -> amostras insuficientes: {count}/{min_samples}")
        
        return final_mapping

def save_visualization(image, pred_map, final_mapping, view_id, output_dir, class_colors=None):
    """Salva visualização individual com classes atribuídas"""
    
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    if class_colors is None:
        class_colors = {}
        for obj_id, info in final_mapping.items():
            class_name = info['class']
            if class_name not in class_colors:
                np.random.seed(hash(class_name) % 2**32)
                class_colors[class_name] = np.random.randint(0, 255, 3).tolist()
    
    # Aplicar cores aos objetos mapeados (object_id agora é string, converter para int)
    for obj_id_str, info in final_mapping.items():
        obj_id = int(obj_id_str)  # Converter de volta para int
        class_name = info['class']
        color = class_colors.get(class_name, [128, 128, 128])
        color_mask[pred_map == obj_id] = color
    
    all_mapped_ids = {int(k) for k in final_mapping.keys()}
    unmapped_mask = np.isin(pred_map, list(all_mapped_ids), invert=True) & (pred_map != 0)
    color_mask[unmapped_mask] = [128, 128, 128]
    
    overlay = cv2.addWeighted(img_bgr, 0.6, color_mask, 0.4, 0)
    
    # Adicionar legenda
    y_offset = 30
    for class_name, color in class_colors.items():
        cv2.rectangle(overlay, (10, y_offset - 20), (30, y_offset), color, -1)
        cv2.putText(overlay, class_name, (40, y_offset - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    cv2.putText(overlay, f"View: {view_id}", (10, y_offset + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Objects mapped: {len(final_mapping)}", (10, y_offset + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    output_path = os.path.join(output_dir, f"view_{view_id:04d}_classification.jpg")
    cv2.imwrite(output_path, overlay)
    
    return output_path


def save_annotation_helper(image, pred_map, output_dir, view_id):
    """Gera visualização com IDs dos objetos para facilitar anotação manual"""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    unique_ids = np.unique(pred_map)
    unique_ids = unique_ids[unique_ids != 0] # Remove background
    
    overlay = img_bgr.copy()
    
    for obj_id in unique_ids:
        # Criar máscara para o objeto atual
        mask = (pred_map == obj_id).astype(np.uint8)
        
        # Encontrar contornos para achar o centro
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Pegar o maior contorno (evita ruído)
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            
            if M["m00"] != 0:
                # Centro de massa da máscara
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Desenha um círculo no fundo do texto para legibilidade
                cv2.circle(overlay, (cX, cY), 15, (0, 0, 0), -1)
                # Escreve o ID do objeto (ex: "12")
                cv2.putText(overlay, str(obj_id), (cX - 10, cY + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    output_path = os.path.join(output_dir, f"annotation_view_{view_id:04d}.jpg")
    cv2.imwrite(output_path, overlay)


def save_detailed_report(final_mapping, detection_history, output_dir):
    """Salva relatório detalhado em JSON e TXT"""
    
    # Converter detection_history para tipos nativos
    clean_history = []
    for det in detection_history:
        clean_history.append({
            'view_id': int(det['view_id']),
            'object_id': int(det['object_id']),
            'class': str(det['class']),
            'confidence': float(det['confidence']),
            'iou': float(det['iou']),
            'bbox': [int(x) for x in det['bbox']]
        })
    
    report_json = {
        'timestamp': datetime.now().isoformat(),
        'mapping': final_mapping,  # Já está com tipos nativos
        'total_objects_mapped': len(final_mapping),
        'detection_history': clean_history,
        'statistics': {
            'total_detections': len(clean_history),
            'unique_objects': len(set(d['object_id'] for d in clean_history)),
            'unique_classes': len(set(d['class'] for d in clean_history))
        }
    }
    
    json_path = os.path.join(output_dir, 'classification_report.json')
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2)
    
    # Salvar em TXT legível
    txt_path = os.path.join(output_dir, 'classification_report.txt')
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total objects mapped: {len(final_mapping)}\n")
        f.write(f"Total detections: {len(clean_history)}\n\n")
        
        f.write("MAPPING DETAILS:\n")
        f.write("-"*40 + "\n")
        for obj_id, info in final_mapping.items():
            f.write(f"\nObject ID: {obj_id}\n")
            f.write(f"  -> Class: {info['class']}\n")
            f.write(f"  -> Confidence: {info['confidence']:.2f}\n")
            f.write(f"  -> Samples: {info['samples']}/{info['total_samples']}\n")
            f.write(f"  -> All detections:\n")
            for cls, count in info['all_classes'].items():
                f.write(f"       - {cls}: {count} times\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("CLASS SUMMARY:\n")
        class_summary = defaultdict(int)
        for info in final_mapping.values():
            class_summary[info['class']] += 1
        for cls, count in class_summary.items():
            f.write(f"  {cls}: {count} objects\n")
    
    return json_path, txt_path

def create_summary_video(output_dir, fps=2):
    """Cria um vídeo com todas as visualizações"""
    images = [f for f in os.listdir(output_dir) if f.startswith('view_') and f.endswith('_classification.jpg')]
    images.sort()
    
    if not images:
        return None
    
    first_img = cv2.imread(os.path.join(output_dir, images[0]))
    if first_img is None:
        return None
    
    height, width, _ = first_img.shape
    
    video_path = os.path.join(output_dir, 'classification_summary.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for img_name in images:
        img_path = os.path.join(output_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            video_writer.write(img)
    
    video_writer.release()
    return video_path

def save_class_colors_legend(class_colors, output_dir):
    """Salva uma legenda com as cores das classes"""
    if not class_colors:
        return None
    
    # Calcular altura necessária
    item_height = 30
    padding = 20
    height = len(class_colors) * item_height + padding * 2
    width = 300
    
    legend = np.ones((height, width, 3), dtype=np.uint8) * 255  # Fundo branco
    
    y_offset = padding
    for i, (class_name, color) in enumerate(class_colors.items()):
        # Desenhar retângulo colorido
        cv2.rectangle(legend, (20, y_offset), (50, y_offset + 20), tuple(color), -1)
        # Escrever nome da classe
        cv2.putText(legend, class_name, (70, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += item_height
    
    legend_path = os.path.join(output_dir, 'class_colors_legend.jpg')
    cv2.imwrite(legend_path, legend)
    return legend_path

def main():
    parser = ArgumentParser(description="Render + YOLO detection + segmentation overlay")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_views", default=50, type=int, help="Número de views para processar")
    parser.add_argument("--min_samples", default=3, type=int, help="Mínimo de amostras para mapeamento")
    parser.add_argument("--iou_threshold", default=0.3, type=float, help="Threshold de IoU")
    parser.add_argument("--conf_threshold", default=0.5, type=float, help="Threshold de confiança YOLO")
    parser.add_argument("--output_dir", default="output_yolo", type=str, help="Diretório de saída")
    args = get_combined_args(parser)
    safe_state(False)
    
    # Criar diretório de saída
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Criando diretório de saída: {output_dir}")
    
    with torch.no_grad():
        # Carregar Gaussian Scene
        print("\n🔧 Carregando Gaussian Scene...")
        gaussians = GaussianModel(model.extract(args).sh_degree)
        scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
        dataset = model.extract(args)
        
        # Carregar classificador
        print("🔧 Carregando classificador...")
        classifier = torch.nn.Conv2d(
            gaussians.num_objects,
            dataset.num_classes,
            kernel_size=1
        ).cuda()
        classifier_path = os.path.join(
            dataset.model_path,
            "point_cloud",
            f"iteration_{scene.loaded_iter}",
            "classifier.pth"
        )
        classifier.load_state_dict(torch.load(classifier_path))
        classifier.eval()
        
        # YOLO Model
        print("🔧 Carregando YOLO...")
        yolo_model = YOLO("yolov8s.pt")
        
        # Background
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Mapeador de objetos
        mapper = ObjectClassifierMapper(iou_threshold=args.iou_threshold)
        
        # Processar múltiplas views
        cameras = scene.getTrainCameras()
        num_views = min(args.num_views, len(cameras))
        
        print(f"\n🔍 Processando {num_views} views...")
        print(f"   YOLO confidence threshold: {args.conf_threshold}")
        print(f"   IoU threshold: {args.iou_threshold}")
        
        for view_idx in tqdm(range(num_views), desc="Processando views"):
            view = cameras[view_idx]
            
            results = render(view, gaussians, pipeline.extract(args), background)
            img_rgb = (view.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            rendering_obj = results["render_object"]
            logits = classifier(rendering_obj)
            pred_map = torch.argmax(logits, dim=0).cpu().numpy()
            save_annotation_helper(img_rgb, pred_map, output_dir, view_id=view_idx)
            
            yolo_results = yolo_model(img_rgb, conf=args.conf_threshold)
            detections = yolo_results[0]
            
            unique_objects = np.unique(pred_map)
            unique_objects = unique_objects[unique_objects != 0]
            
            for obj_id in unique_objects:
                obj_mask = (pred_map == obj_id).astype(np.uint8)
                
                if detections.boxes is not None:
                    for box in detections.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = yolo_model.names[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = (x1, y1, x2, y2)
                        
                        mapper.process_detection(obj_id, class_name, conf, obj_mask, bbox, view_idx)
        
        # Finalizar mapeamento
        print("\n" + "="*50)
        print("📊 GERANDO MAPEAMENTO FINAL")
        print("="*50)
        
        final_mapping = mapper.finalize_mapping(
            min_samples=args.min_samples,
            min_confidence=0.5
        )
        
        # Salvar relatório
        print("\n💾 Salvando relatórios...")
        json_path, txt_path = save_detailed_report(final_mapping, mapper.detection_history, output_dir)
        print(f"   ✓ JSON report: {json_path}")
        print(f"   ✓ TXT report: {txt_path}")
        
        # Salvar mapeamento simplificado
        mapping_path = os.path.join(output_dir, "object_class_mapping.txt")
        with open(mapping_path, 'w') as f:
            f.write("# object_id -> class_name (confidence, samples)\n")
            for obj_id, info in final_mapping.items():
                line = f"{obj_id} -> {info['class']} (conf: {info['confidence']:.2f}, samples: {info['samples']}/{info['total_samples']})\n"
                f.write(line)
        
        # Gerar visualizações para TODAS as views
        print("\n🎨 Gerando visualizações para todas as views...")
        
        # Gerar cores consistentes
        class_colors = {}
        for obj_id, info in final_mapping.items():
            class_name = info['class']
            if class_name not in class_colors:
                np.random.seed(hash(class_name) % 2**32)
                class_colors[class_name] = np.random.randint(0, 255, 3).tolist()
        
        # Salvar legenda de cores
        save_class_colors_legend(class_colors, output_dir)
        
        # Processar todas as views para visualização
        all_views = cameras[:num_views]
        for view_idx in tqdm(range(len(all_views)), desc="Gerando visualizações"):
            view = all_views[view_idx]
            
            results = render(view, gaussians, pipeline.extract(args), background)
            img_rgb = (results["render"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            rendering_obj = results["render_object"]
            logits = classifier(rendering_obj)
            pred_map = torch.argmax(logits, dim=0).cpu().numpy()
            
            save_visualization(img_rgb, pred_map, final_mapping, view_idx, output_dir, class_colors)
            
        print(f"\n✅ Visualizações salvas em: {output_dir}")
        
        # Criar vídeo resumo
        print("\n🎬 Criando vídeo resumo...")
        video_path = create_summary_video(output_dir, fps=2)
        if video_path:
            print(f"   ✓ Vídeo salvo: {video_path}")
        
        # Estatísticas finais
        print("\n" + "="*50)
        print("📊 ESTATÍSTICAS FINAIS")
        print("="*50)
        print(f"Total objetos mapeados: {len(final_mapping)}")
        print(f"Total detecções processadas: {len(mapper.detection_history)}")
        print(f"Classes únicas: {len(set(info['class'] for info in final_mapping.values()))}")
        print(f"\nDistribuição por classe:")
        class_dist = defaultdict(int)
        for info in final_mapping.values():
            class_dist[info['class']] += 1
        for cls, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cls}: {count} objetos")
        
        print(f"\n💾 Todos os resultados salvos em: {output_dir}")

if __name__ == "__main__":
    main()