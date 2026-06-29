#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name> <scale>"
    exit 1
fi

dataset_name="$1"
scale="$2"
dataset_folder="data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi

# 1. DEVA anything mask
cd Tracking-Anything-with-DEVA/

if [ "$scale" = "1" ]; then
    img_path="../data/${dataset_name}/images"
else
    img_path="../data/${dataset_name}/images_${scale}"
fi

echo "Using images from: $img_path"

############################################
# CONFIG OTIMIZADA PRA RTX 4060 (8GB)
############################################

COMMON_ARGS="
  --chunk_size 1
  --amp
  --temporal_setting semionline
  --size 320
  --suppress_small_objects
  --SAM_PRED_IOU_THRESHOLD 0.7
  --sam_variant mobile
  --SAM_NUM_POINTS_PER_SIDE 32
  --SAM_NUM_POINTS_PER_BATCH 32
  --disable_long_term
"

############################################
# 1️⃣ COLORED MASK (visualização)
############################################

python demo/demo_automatic.py \
  --img_path "$img_path" \
  --output "./example/output_gaussian_dataset/${dataset_name}" \
  $COMMON_ARGS

# rename se existir
if [ -d "./example/output_gaussian_dataset/${dataset_name}/Annotations" ]; then
    mv ./example/output_gaussian_dataset/${dataset_name}/Annotations \
       ./example/output_gaussian_dataset/${dataset_name}/Annotations_color
fi

############################################
# 2️⃣ GRAY MASK (treino)
############################################

python demo/demo_automatic.py \
  --img_path "$img_path" \
  --output "./example/output_gaussian_dataset/${dataset_name}" \
  --use_short_id \
  $COMMON_ARGS

############################################
# 3️⃣ COPIAR PARA DATASET
############################################

mkdir -p ../data/${dataset_name}/object_mask

if [ -d "./example/output_gaussian_dataset/${dataset_name}/Annotations" ]; then
    cp -r ./example/output_gaussian_dataset/${dataset_name}/Annotations \
          ../data/${dataset_name}/object_mask
else
    echo "⚠️ Warning: Annotations not generated"
fi

cd ..

echo "✅ Finished pseudo label generation"