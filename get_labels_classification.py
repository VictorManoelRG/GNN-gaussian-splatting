import torch
import os
import numpy as np
from ultralytics import YOLO
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
import json

class ObjectClassifierMapper:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        # Armazena contagem de classes detectadas pelo YOLO para cada ID do Gaussian
        self.object_class_counts = defaultdict(lambda: defaultdict(int))
        # Conjunto de todos os IDs que apareceram em cena (mesmo sem YOLO)
        self.seen_object_ids = set()
        
    def compute_iou(self, mask, bbox):
        x1, y1, x2, y2 = bbox
        h, w = mask.shape
        
        # Crop e segurança de índices
        x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1: return 0.0
        
        bbox_mask = np.zeros((h, w), dtype=np.bool_)
        bbox_mask[y1:y2, x1:x2] = True
        
        intersection = np.logical_and(mask, bbox_mask).sum()
        union = np.logical_or(mask, bbox_mask).sum()
        
        return float(intersection / union) if union > 0 else 0.0

    def add_to_seen(self, obj_ids):
        """Registra que esses IDs existem na cena"""
        for oid in obj_ids:
            if oid != 0: # Pula background
                self.seen_object_ids.add(int(oid))

    def process_detection(self, object_id, class_name, mask, bbox):
        iou = self.compute_iou(mask, bbox)
        if iou >= self.iou_threshold:
            self.object_class_counts[int(object_id)][class_name] += 1
            return True
        return False

    def generate_realtime_dict(self, min_samples=2):
        """
        Gera o dicionário final: { "ID": "CLASSE" }
        Se o YOLO não detectou com confiança, deixa como "unlabeled"
        """
        mapping = {}
        # Ordenar IDs para facilitar edição manual posterior
        for obj_id in sorted(list(self.seen_object_ids)):
            str_id = str(obj_id)
            class_counts = self.object_class_counts.get(obj_id)
            
            if class_counts:
                # Pega a classe que mais apareceu nas views
                most_common_class, count = max(class_counts.items(), key=lambda x: x[1])
                if count >= min_samples:
                    mapping[str_id] = most_common_class
                else:
                    mapping[str_id] = "unlabeled"
            else:
                mapping[str_id] = "unlabeled"
        
        return mapping

def main():
    parser = ArgumentParser(description="YOLO to Gaussian ID Mapper")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--iou_threshold", default=0.3, type=float)
    parser.add_argument("--conf_threshold", default=0.4, type=float)
    parser.add_argument("--output_json", default="object_class_mapping.json", type=str)
    args = get_combined_args(parser)
    safe_state(False)

    # Inicializar Modelos
    gaussians = GaussianModel(model.extract(args).sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    
    # Carregar Classificador de Objetos (Gaussian Grouping)
    classifier = torch.nn.Conv2d(gaussians.num_objects, model.extract(args).num_classes, kernel_size=1).cuda()
    classifier_path = os.path.join(model.extract(args).model_path, "point_cloud", f"iteration_{scene.loaded_iter}", "classifier.pth")
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.eval()

    yolo_model = YOLO("yolov8s.pt")
    mapper = ObjectClassifierMapper(iou_threshold=args.iou_threshold)
    
    cameras = scene.getTrainCameras()
    
    bg_color = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for view_idx in tqdm(range(len(cameras))):
        view = cameras[view_idx]
        render_pkg = render(view, gaussians, pipeline.extract(args), background)
        
        # Imagem para o YOLO
        img_np = (view.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Mapa de Segmentação do Gaussian Grouping
        logits = classifier(render_pkg["render_object"])
        pred_map = torch.argmax(logits, dim=0).cpu().numpy()
        
        # Registrar todos os IDs que aparecem nesta view
        unique_ids = np.unique(pred_map)
        mapper.add_to_seen(unique_ids)
        
        # Detecção YOLO
        yolo_results = yolo_model(img_np, conf=args.conf_threshold, verbose=False)
        detections = yolo_results[0]

        if detections.boxes is not None:
            for box in detections.boxes:
                cls_name = yolo_model.names[int(box.cls[0])]
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                
                # Para cada ID do Gaussian presente na tela, checar se bate com o BBOX do YOLO
                for obj_id in unique_ids:
                    if obj_id == 0: continue
                    obj_mask = (pred_map == obj_id)
                    mapper.process_detection(obj_id, cls_name, obj_mask, bbox)

    # Gerar Dicionário Final
    final_dict = mapper.generate_realtime_dict()

    # Salvar para uso em tempo de execução
    with open(args.output_json, 'w') as f:
        json.dump(final_dict, f, indent=4)

    print(f"\n✅ Mapeamento concluído!")
    print(f"📂 Arquivo salvo em: {args.output_json}")
    print(f"Total de objetos processados: {len(final_dict)}")
    
    unlabeled_count = list(final_dict.values()).count("unlabeled")
    if unlabeled_count > 0:
        print(f"⚠️ {unlabeled_count} objetos não foram classificados automaticamente e estão como 'unlabeled'.")

if __name__ == "__main__":
    with torch.no_grad():
        main()