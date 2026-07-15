#!/bin/bash

# Calcula throughput de inferencia (fps) a partir de los resultados de latencia.
# Requiere haber ejecutado engine_latency_test.sh previamente.

INFILE="outputs/inference_engine_latency_results.txt"
OUTFILE="outputs/inference_engine_throughput_results.txt"

if [ ! -f "$INFILE" ]; then
    echo "ERROR: No se encontró $INFILE. Ejecuta engine_latency_test.sh primero."
    exit 1
fi

echo "===== INFERENCE ENGINE THROUGHPUT TEST =====" > "$OUTFILE"
echo "Derivado de: $INFILE"                               >> "$OUTFILE"
echo "Fecha: $(date)"                                     >> "$OUTFILE"
echo "===================================================" >> "$OUTFILE"

python3 - <<'EOF' | tee -a "$OUTFILE"
import re
import sys

INFILE = "outputs/inference_engine_latency_results.txt"

current_model = None
inferencia_block = False
median_ms = None

results = []

with open(INFILE) as f:
    for line in f:
        line_s = line.rstrip()

        # Detectar nombre de modelo
        m = re.match(r'^\[MODEL: (.+)\]$', line_s)
        if m:
            current_model = m.group(1)
            inferencia_block = False
            median_ms = None
            continue

        # Detectar inicio del bloque Inferencia
        if line_s == '  Inferencia':
            inferencia_block = True
            continue

        # Dentro del bloque, capturar mediana
        if inferencia_block and median_ms is None:
            m = re.match(r'\s+Mediana:\s+([\d.]+)\s+ms', line_s)
            if m:
                median_ms = float(m.group(1))
                results.append((current_model, median_ms))
                inferencia_block = False
                continue

        # Cualquier línea con dos espacios + no-espacio cierra el bloque
        if inferencia_block and re.match(r'^  \S', line_s):
            inferencia_block = False

print()
print(f"{'Modelo':<40}  {'Mediana inf. (ms)':>18}  {'Throughput (fps)':>18}")
print("-" * 80)
for model, med in results:
    fps = 1000.0 / med
    print(f"{model:<40}  {med:>18.3f}  {fps:>18.2f}")
print()
EOF

echo "Resultados guardados en: $OUTFILE"