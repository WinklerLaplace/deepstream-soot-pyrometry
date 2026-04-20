#!/bin/bash

DATASET="datasets/img_preprocess"

OUTDIR="outputs"
RESULTS_FILE="$OUTDIR/memory_results.txt"
TEGRALOG_DIR="$OUTDIR/tegrastats_log"
INFERLOG_DIR="$OUTDIR/inference_log"

mkdir -p "$TEGRALOG_DIR" "$INFERLOG_DIR"

BATCH_SIZE=32

MODEL_PT_UNET="unet.pth"
MODEL_PT_ATT="attunet.pth"

MODEL_TRT_UNET_FP32="unet_fp32.engine"
MODEL_TRT_UNET_FP16="unet_fp16.engine"
MODEL_TRT_UNET_INT8="unet_int8.engine"

MODEL_TRT_ATT_FP32="attunet_fp32.engine"
MODEL_TRT_ATT_FP16="attunet_fp16.engine"
MODEL_TRT_ATT_INT8="attunet_int8.engine"

echo "===== MEMORY TEST =====" > "$RESULTS_FILE"
echo "Dataset: $DATASET" >> "$RESULTS_FILE"
echo "Batch size: $BATCH_SIZE" >> "$RESULTS_FILE"
echo "Fecha: $(date)" >> "$RESULTS_FILE"
echo "=================================================" >> "$RESULTS_FILE"

echo "Midiendo memoria idle..."
idle_ram=$(tegrastats --interval 1000 | head -n 1 | awk '{for(i=1;i<=NF;i++) if($i=="RAM") print $(i+1)}' | cut -d'/' -f1)

if [[ -z "$idle_ram" ]]; then
    echo "Error midiendo RAM idle"
    exit 1
fi

echo "Idle RAM: ${idle_ram} MB" | tee -a "$RESULTS_FILE"
echo "-----------------------------------------------" >> "$RESULTS_FILE"

run_test () {

    NAME=$1
    WEIGHTS=$2
    MODEL_TYPE=$3

    LOG_NAME=$(echo "$NAME" | tr ' ' '_' | tr -d '()')

    TEGRALOG="$TEGRALOG_DIR/${LOG_NAME}.txt"
    INFERLOG="$INFERLOG_DIR/${LOG_NAME}.txt"

    echo ""
    echo "[MODEL: $NAME]" | tee -a "$RESULTS_FILE"

    tegrastats --interval 1000 --logfile "$TEGRALOG" &
    PID=$!

    sleep 1

    python3 eval.py \
        --weights "weights/$WEIGHTS" \
        --model "$MODEL_TYPE" \
        --dataset "$DATASET" \
        --simulate \
        --batch_size "$BATCH_SIZE" \
        > "$INFERLOG"

    kill "$PID"
    wait "$PID" 2>/dev/null

    RAM_VALUES=$(grep -oP 'RAM \K[0-9]+' "$TEGRALOG")

    if [[ -z "$RAM_VALUES" ]]; then
        echo "Error: sin datos RAM" | tee -a "$RESULTS_FILE"
        return
    fi

    RAM_MAX=$(echo "$RAM_VALUES" | sort -nr | head -1)
    RAM_AVG=$(echo "$RAM_VALUES" | awk '{sum+=$1} END {printf "%.2f", sum/NR}')

    RAM_MAX_NET=$((RAM_MAX - idle_ram))
    RAM_AVG_NET=$(awk "BEGIN {printf \"%.2f\", $RAM_AVG - $idle_ram}")

    echo "RAM max total : ${RAM_MAX} MB" | tee -a "$RESULTS_FILE"
    echo "RAM avg total : ${RAM_AVG} MB" | tee -a "$RESULTS_FILE"
    echo "RAM max net   : ${RAM_MAX_NET} MB" | tee -a "$RESULTS_FILE"
    echo "RAM avg net   : ${RAM_AVG_NET} MB" | tee -a "$RESULTS_FILE"

    echo "-----------------------------------------------" >> "$RESULTS_FILE"
}

# =========================
#  U-Net
# =========================

run_test "U-Net PyTorch Base" $MODEL_PT_UNET "unet"

# run_test "U-Net TensorRT FP32" $MODEL_TRT_UNET_FP32 "tensorrt"
# run_test "U-Net TensorRT FP16" $MODEL_TRT_UNET_FP16 "tensorrt"
# run_test "U-Net TensorRT INT8" $MODEL_TRT_UNET_INT8 "tensorrt"

# =========================
#  Attention U-Net
# =========================

run_test "Attention U-Net PyTorch Base" $MODEL_PT_ATT "attunet"

# run_test "Attention U-Net TensorRT FP32" $MODEL_TRT_ATT_FP32 "tensorrt"
# run_test "Attention U-Net TensorRT FP16" $MODEL_TRT_ATT_FP16 "tensorrt"
# run_test "Attention U-Net TensorRT INT8" $MODEL_TRT_ATT_INT8 "tensorrt"

echo ""
echo "Fin: $(date)" >> "$RESULTS_FILE"
echo "Resultados guardados en: $RESULTS_FILE"