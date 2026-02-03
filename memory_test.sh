#!/bin/bash

# ==== Modelos a evaluar ====
models_pt=("unet" "attunet")            # Modelos base PyTorch
models_trt=("unet_fp16" "attunet_fp16") # Modelos TensorRT FP16
batch_size=32                             # Solo batch 32

mkdir -p outputs/tegrastats_log

export FIX_DATASET=1  # Flag opcional que ya usabas

echo "=== Medición de memoria RAM (batch 32) ==="

# Función para medir RAM de un solo modelo
measure_ram() {
    local model_name=$1
    local model_file=$2
    local framework=$3
    local log_name="${model_name}_bs${batch_size}_ram"

    echo -e "\nEjecutando $framework $model_name con batch size $batch_size"

    # Inicia tegrastats
    tegrastats --interval 1 --logfile outputs/tegrastats_log/${log_name}.txt & 
    tegrastat_pid=$!

    # Ejecuta inferencia
    python3 eval.py --weights weights/$model_file --model $framework --dataset datasets/img_preprocess --latency --batch_size $batch_size > ${log_name}.txt

    # Mata tegrastats
    kill -9 $tegrastat_pid
    wait $tegrastat_pid 2>/dev/null

    # Extrae RAM
    ram_values=$(grep -oP 'RAM \K[0-9]+' outputs/tegrastats_log/${log_name}.txt)
    if [[ -z "$ram_values" ]]; then
        echo "No se encontraron valores de RAM en el log para $model_name"
        return
    fi

    ram_max=$(echo "$ram_values" | sort -nr | head -1)
    ram_avg=$(echo "$ram_values" | awk '{sum+=$1} END {print sum/NR}')

    echo "----------------------------------------------------------"
    echo "Resultados de memoria RAM para $model_name (batch $batch_size):"
    echo "RAM máxima del sistema : ${ram_max} MB"
    echo "RAM promedio del sistema: ${ram_avg} MB"
    echo "----------------------------------------------------------"
}

# --- Modelos PyTorch ---
for model in "${models_pt[@]}"; do
    measure_ram "$model" "$model.pth" "$model"
done

# --- Modelos TensorRT FP16 ---
for model in "${models_trt[@]}"; do
    measure_ram "$model" "$model.engine" "tensorrt"
done

echo -e "\nMedición de memoria completada."
