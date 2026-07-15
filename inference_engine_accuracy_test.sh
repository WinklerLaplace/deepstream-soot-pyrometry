#!/bin/bash

DATASET="datasets/data_experimental"
OUTDIR="outputs"
OUTFILE="$OUTDIR/inference_engine_accuracy_results.txt"

CASES=("A" "B" "C")

mkdir -p "$OUTDIR"

echo "===== ACCURACY / PRECISION TEST =====" > "$OUTFILE"
echo "Dataset: $DATASET" >> "$OUTFILE"
echo "Casos: ${CASES[@]}" >> "$OUTFILE"
echo "Fecha ejecución: $(date)" >> "$OUTFILE"
echo "=================================================" >> "$OUTFILE"

run_test () {

    NAME=$1
    WEIGHTS=$2
    MODEL_TYPE=$3
    CASE=$4

    echo "" | tee -a "$OUTFILE"
    echo "[MODEL: $NAME | CASE: $CASE]" | tee -a "$OUTFILE"

    python3 eval.py \
        --weights "weights/$WEIGHTS" \
        --model "$MODEL_TYPE" \
        --experiment \
        --case "$CASE" \
        2>&1 | tee -a "$OUTFILE"

    echo "-----------------------------------------------" >> "$OUTFILE"
}

# =========================
#  EJECUCIÓN
# =========================

for CASE in "${CASES[@]}"; do

    echo "" | tee -a "$OUTFILE"
    echo "#################### CASE $CASE ####################" | tee -a "$OUTFILE"

    # U-Net
    run_test "U-Net PyTorch Base" "unet.pth" "unet" "$CASE"
    run_test "U-Net TensorRT FP32" "unet_fp32.engine" "tensorrt" "$CASE"
    run_test "U-Net TensorRT FP16" "unet_fp16.engine" "tensorrt" "$CASE"
    run_test "U-Net TensorRT INT8" "unet_int8.engine" "tensorrt" "$CASE"

    # Attention U-Net
    run_test "Attention U-Net PyTorch Base" "attunet.pth" "attunet" "$CASE"
    run_test "Attention U-Net TensorRT FP32" "attunet_fp32.engine" "tensorrt" "$CASE"
    run_test "Attention U-Net TensorRT FP16" "attunet_fp16.engine" "tensorrt" "$CASE"
    run_test "Attention U-Net TensorRT INT8" "attunet_int8.engine" "tensorrt" "$CASE"

done

echo "" >> "$OUTFILE"
echo "Fin: $(date)" >> "$OUTFILE"

echo "Resultados guardados en: $OUTFILE"