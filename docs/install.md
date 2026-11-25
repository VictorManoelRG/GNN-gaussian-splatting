### **Standard Installation**


Clone the repository locally
```
git clone https://github.com/lkeab/gaussian-grouping.git
cd gaussian-grouping
```

Our default, provided install method is based on Conda package and environment management:
```bash
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

(Optional) If you want to prepare masks on your own dataset, you will also need to prepare [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) environment.

```bash
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh     # Download the pretrained models

git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

cd ../..
```

(Optional) If you want to inpaint on your own dataset, you will also need to prepare [LaMa](https://github.com/advimman/lama) environment.

```bash
cd lama
pip install -r requirements.txt
cd ..
```

### **Ambiente da tese**

Os experimentos da tese não usaram o PyTorch 1.12.1 acima, e sim o ambiente abaixo (Python 3.8.20, PyTorch 2.0.1, CUDA 11.8). As versões exatas dos pacotes estão em `requirements_tese.txt`, e o `conda env export` completo do ambiente original está em `docs/ambiente_gaussian_grouping.yml`, só como registro (ele cita pacotes locais que não existem no PyPI).

```bash
git lfs install                     # os modelos 3DGS treinados (point_cloud.ply) estão no Git LFS
git clone git@github.com:VictorManoelRG/GNN-gaussian-splatting.git gaussian-grouping
cd gaussian-grouping

conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_tese.txt -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

O DEVA (seção opcional acima) só é necessário para gerar máscaras de cenas novas. Para as 4 cenas da tese (teatime, ramen, figurines e waldo_kitchen), o repositório já traz tudo o que o algoritmo lê:

- `data/<cena>/{images,sparse,object_mask}`: entrada do Scene;
- `output/<cena>/`: modelo 3DGS treinado (iteração 30000), sem precisar retreinar;
- `Tracking-Anything-with-DEVA/example/output_gaussian_dataset/<cena>/`: máscaras e `pred.json` do DEVA;
- `data/lerf_ovs/label/<cena>/`: ground truth do LERF-OVS, usado pelas métricas.

Rodando da raiz do repositório, a configuração Padrão na cena teatime e o mIoU ficam assim (os outros experimentos estão em `run_experiments5*.sh` e `refazer_waldo_kitchen_187cams_2.sh`):

```bash
DEVA=Tracking-Anything-with-DEVA/example/output_gaussian_dataset/teatime
python train_gnn_final_version.py -m output/teatime --iteration 30000 \
  --masks_path $DEVA/Annotations --deva_json $DEVA/pred.json \
  --model_type_choice gat --gat_heads 8 --no_visualize_3d \
  --target_gaussians 250000 --use_bbox --no_render_full_pointcloud \
  --edge_distance_mode local_scaling \
  --min_cluster_size 500 --hdbscan_min_samples 3 --hdbscan_cluster_selection_method eom \
  --output exp_res --experiment_name teatime_padrao

python calculate_miou_claude_full_table.py --gt data/lerf_ovs/label/teatime \
  --pred exp_res/teatime_padrao/gat/images_coverage --out exp_res/teatime_padrao_miou.json
```