#!/bin/bash

BATCHES=(2 4 8 16 32)
DATASET="datasets/img_preprocess"
OUTFILE="outputs/latency_results"

# Solo 3 modelos a ejecutar
MODEL_TRT_UNET="unet_fp16.engine"
MODEL_PT_BASE="attunet.pth"
MODEL_TRT_ATT="attunet_fp16.engine"

echo "===== Latency & Throughput Test (FP16 & Base) =====" > $OUTFILE
echo "Dataset: $DATASET" >> $OUTFILE
echo "Fecha ejecución: $(date)" >> $OUTFILE
echo "-----------------------------------------------" >> $OUTFILE

# ---- U-Net FP16 (TensorRT) ----
for B in "${BATCHES[@]}"; do
  echo "[TRT-FP16] Engine $MODEL_TRT_UNET — batch $B" | tee -a $OUTFILE
  python3 eval.py --weights weights/$MODEL_TRT_UNET --model tensorrt --dataset $DATASET --latency --batch_size $B 2>&1 | tee -a $OUTFILE
  python3 - <<EOF
import torch; torch.cuda.empty_cache()
EOF
  echo "-----------------------------------------------" >> $OUTFILE
done

# ---- Attention U-Net Base (PyTorch) ----
for B in "${BATCHES[@]}"; do
  echo "[BASE] Modelo $MODEL_PT_BASE — batch $B" | tee -a $OUTFILE
  python3 eval.py --weights weights/$MODEL_PT_BASE --model attunet --dataset $DATASET --latency --batch_size $B 2>&1 | tee -a $OUTFILE
  python3 - <<EOF
import torch; torch.cuda.empty_cache()
EOF
  echo "-----------------------------------------------" >> $OUTFILE
done

# ---- Attention U-Net FP16 (TensorRT) ----
for B in "${BATCHES[@]}"; do
  echo "[TRT-FP16] Engine $MODEL_TRT_ATT — batch $B" | tee -a $OUTFILE
  python3 eval.py --weights weights/$MODEL_TRT_ATT --model tensorrt --dataset $DATASET --latency --batch_size $B 2>&1 | tee -a $OUTFILE
  python3 - <<EOF
import torch; torch.cuda.empty_cache()
EOF
  echo "-----------------------------------------------" >> $OUTFILE
done

echo "Pruebas guardadas en $OUTFILE"
