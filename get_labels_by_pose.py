#!/usr/bin/python
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import torch
import json
import os
from std_msgs.msg import Float32MultiArray

# Imports do Gaussian Grouping / Splatting
from scene.cameras import Camera
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser

class GaussianRealTimeInference(Node):
    def __init__(self, dataset_params, pipe_params, iteration):
        super().__init__('gaussian_realtime_inference')

        # 1. Carregar o Modelo e Classificador
        self.gaussians = GaussianModel(dataset_params.sh_degree)
        # Passamos None para o Scene pois não vamos carregar câmeras de treino, apenas o modelo
        self.gaussians.load_ply(os.path.join(dataset_params.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        self.dataset_params = dataset_params
        self.pipe_params = pipe_params
        self.bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        # 2. Carregar Classificador (Gaussian Grouping)
        self.classifier = torch.nn.Conv2d(self.gaussians.num_objects, dataset_params.num_classes, kernel_size=1).cuda()
        classifier_path = os.path.join(dataset_params.model_path, "point_cloud", f"iteration_{iteration}", "classifier.pth")
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.eval()
        print("Model path: " + dataset_params.model_path)
        # 3. Carregar seu Dicionário de Labels (O que você criou no passo anterior)
        mapping_path = os.path.join(dataset_params.model_path,"object_class_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                self.id_to_class = json.load(f)
            self.get_logger().info(f"✅ Mapeamento carregado: {len(self.id_to_class)} objetos")
        else:
            self.id_to_class = {}
            self.get_logger().warn(f"⚠️ Mapeamento não encontrado em: {mapping_path}")

        # Configurações de Câmera (Baseado no seu fx, fy, cx, cy)
        self.fx = 397.64
        self.fy = 397.64
        self.cx = 960
        self.cy = 640
        self.width = 1920
        self.height = 1080

    def aruco_to_gs_camera(self, p, view_id=0):
        """Converte pose do Aruco para objeto Camera do GS"""
        tvec = np.array([p[0], p[1], p[2]]).reshape(3, 1)
        rvec = np.array([p[3], p[4], p[5]])
        R, _ = cv2.Rodrigues(rvec)

        R_inv = R.T
        t_inv = -R_inv @ tvec

        # Matriz de Transformação
        pose = np.eye(4)
        pose[:3, :3] = R_inv
        pose[:3, 3] = t_inv.flatten()
        
        # Ajuste de coordenadas (Inversão de eixos comum entre ROS/Aruco e GS)
        pose[1, :] *= -1
        pose[2, :] *= -1
        
        # Extrair R e T para a Camera do GS
        R_gs = pose[:3, :3].T
        T_gs = pose[:3, 3]

        return Camera(colmap_id=view_id, R=R_gs, T=T_gs, 
                      fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, 
                      width=self.width, height=self.height,
                      gt_alpha_mask=None, image_name=f"render_{view_id}", trans=[0,0,0], scale=1.0)

    def pose_callback(self, msg):
        # 1. Converter pose recebida
        gs_cam = self.aruco_to_gs_camera(msg.data)
        
        with torch.no_grad():
            # 2. Renderizar Gaussiana e Objetos
            render_pkg = render(gs_cam, self.gaussians, self.pipe_params, self.bg)
            logits = self.classifier(render_pkg["render_object"])
            
            # 3. Pegar IDs presentes no campo de visão (argmax dos logits)
            pred_map = torch.argmax(logits, dim=0).cpu().numpy()
            unique_ids = np.unique(pred_map)
            
            # 4. Cruzar com o dicionário e imprimir
            visible_objects = []
            for obj_id in unique_ids:
                if obj_id == 0: continue # Pula background
                
                class_name = self.id_to_class.get(str(obj_id), "Desconhecido")
                visible_objects.append(f"{class_name} (ID: {obj_id})")
            
            # Saída limpa no terminal
            if visible_objects:
                print(f"👀 No campo de visão: {', '.join(visible_objects)}")
            else:
                print("🌑 Nenhum objeto mapeado visível.")

def main():
    # Setup de argumentos do Gaussian Splatting
    parser = ArgumentParser(description="Real-time GS Object Inference")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--agent_name", type=str, required=True)
    
    # Gambiarra para lidar com argumentos do ROS e do Parser juntos
    args = get_combined_args(parser)
    
    rclpy.init()
    
    node = GaussianRealTimeInference(
        dataset_params=lp.extract(args),
        pipe_params=pp.extract(args),
        iteration=args.iteration
    )

    # Subscribe na pose do agente
    topic_pose = f"/{args.agent_name}/pose"
    node.create_subscription(
        Float32MultiArray, # Certifique-se do import correto
        topic_pose,
        node.pose_callback,
        10
    )

    node.get_logger().info(f"🚀 Iniciado! Ouvindo pose em {topic_pose}")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()