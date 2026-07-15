#!/bin/bash

DATASET="datasets/img_preprocess"
OUTDIR="outputs"
OUTFILE="$OUTDIR/inference_engine_latency_results.txt"
REPEATS=10
WARMUP=90

mkdir -p "$OUTDIR"

echo "===== INFERENCE ENGINE LATENCY TEST =====" > "$OUTFILE"
echo "Dataset:      $DATASET"          >> "$OUTFILE"
echo "Repeticiones: $REPEATS"          >> "$OUTFILE"
echo "Warm-up:      $WARMUP frames/rep" >> "$OUTFILE"
echo "Fecha:        $(date)"           >> "$OUTFILE"
echo "=================================================" >> "$OUTFILE"

run_test() {
    NAME=$1
    WEIGHTS=$2
    MODEL_TYPE=$3

    echo "" | tee -a "$OUTFILE"
    echo "[MODEL: $NAME]" | tee -a "$OUTFILE"

    python3 eval.py \
        --weights  "weights/$WEIGHTS" \
        --model    "$MODEL_TYPE" \
        --dataset  "$DATASET" \
        --latency \
        --repeats  "$REPEATS" \
        --warmup   "$WARMUP" \
        2>&1 \
        | awk '
            /^  Inferencia$/        { print; capture=1; next }
            capture && /^  [^ ]/    { capture=0 }
            capture                 { print }
        ' \
        | tee -a "$OUTFILE"

    echo "-----------------------------------------------" >> "$OUTFILE"
}

# =========================
#  U-Net
# =========================
echo "" | tee -a "$OUTFILE"
echo "#################### U-Net ####################" | tee -a "$OUTFILE"

run_test "U-Net PyTorch Base"      "unet.pth"          "unet"
run_test "U-Net TensorRT FP32"     "unet_fp32.engine"  "tensorrt"
run_test "U-Net TensorRT FP16"     "unet_fp16.engine"  "tensorrt"
run_test "U-Net TensorRT INT8"     "unet_int8.engine"  "tensorrt"

# =========================
#  Attention U-Net
# =========================
echo "" | tee -a "$OUTFILE"
echo "#################### Attention U-Net ####################" | tee -a "$OUTFILE"

run_test "Att. U-Net PyTorch Base"  "attunet.pth"          "attunet"
run_test "Att. U-Net TensorRT FP32" "attunet_fp32.engine"  "tensorrt"
run_test "Att. U-Net TensorRT FP16" "attunet_fp16.engine"  "tensorrt"
run_test "Att. U-Net TensorRT INT8" "attunet_int8.engine"  "tensorrt"

echo "" >> "$OUTFILE"
echo "Fin: $(date)" >> "$OUTFILE"
echo "Resultados guardados en: $OUTFILE"