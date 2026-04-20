#!/bin/bash

DATASET="datasets/img_preprocess"
OUTDIR="outputs"
OUTFILE="$OUTDIR/throughput_results.txt"

mkdir -p $OUTDIR

BATCHES=(32)

MODEL_PT_UNET="unet.pth"
MODEL_PT_ATT="attunet.pth"

MODEL_TRT_UNET_FP32="unet_fp32.engine"
MODEL_TRT_UNET_FP16="unet_fp16.engine"
MODEL_TRT_UNET_INT8="unet_int8.engine"

MODEL_TRT_ATT_FP32="attunet_fp32.engine"
MODEL_TRT_ATT_FP16="attunet_fp16.engine"
MODEL_TRT_ATT_INT8="attunet_int8.engine"

echo "===== THROUGHPUT TEST =====" > $OUTFILE
echo "Batch sizes: ${BATCHES[@]}" >> $OUTFILE
echo "Dataset: $DATASET" >> $OUTFILE
echo "Fecha ejecución: $(date)" >> $OUTFILE
echo "-----------------------------------------------" >> $OUTFILE

run_test () {
  NAME=$1
  WEIGHTS=$2
  MODEL_TYPE=$3

  for B in "${BATCHES[@]}"; do

    echo "[$NAME | batch $B]" | tee -a $OUTFILE

    python3 eval.py \
      --weights weights/$WEIGHTS \
      --model $MODEL_TYPE \
      --dataset $DATASET \
      --throughput \
      --batch_size $B \
      2>&1 | tee -a $OUTFILE

    python3 - <<EOF
import torch
torch.cuda.empty_cache()
EOF

    echo "-----------------------------------------------" >> $OUTFILE

  done
}

# =========================
#  U-Net
# =========================

run_test "U-Net PyTorch Base" $MODEL_PT_UNET "unet"

# run_test "U-Net TensorRT FP32" $MODEL_TRT_UNET_FP32 "tensorrt"
run_test "U-Net TensorRT FP16" $MODEL_TRT_UNET_FP16 "tensorrt"
# run_test "U-Net TensorRT INT8" $MODEL_TRT_UNET_INT8 "tensorrt"

# =========================
#  Attention U-Net
# =========================

run_test "Attention U-Net PyTorch Base" $MODEL_PT_ATT "attunet"

# run_test "Attention U-Net TensorRT FP32" $MODEL_TRT_ATT_FP32 "tensorrt"
run_test "Attention U-Net TensorRT FP16" $MODEL_TRT_ATT_FP16 "tensorrt"
# run_test "Attention U-Net TensorRT INT8" $MODEL_TRT_ATT_INT8 "tensorrt"

echo "Pruebas de throughput guardadas en $OUTFILE"