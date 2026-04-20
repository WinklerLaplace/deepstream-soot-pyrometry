import argparse
import numpy as np

from utils.processing import (
    process_llamas_frame, 
    resize_temp_fast,
    R_EXP, Z_EXP,
    H_PX, R1, R2, R_X,
    NEW_R, NEW_Z,
    GRID_POINTS,
    decode
)

from scipy.interpolate import interp2d, RegularGridInterpolator

# =========================
# VERSIONES DE REFERENCIA
# =========================

def resize_temp_interp2d(r, z, Tp):
    f = interp2d(r, z, Tp, kind='linear', copy=True, bounds_error=False, fill_value=None)
    new_temp = f(NEW_R, NEW_Z)
    return new_temp

def resize_temp_rgi(r, z, Tp):
    interp = RegularGridInterpolator((z, r), Tp, method="linear", bounds_error=False, fill_value=None)
    new_temp = interp(GRID_POINTS)
    return new_temp.reshape(128, 32)

# =========================
# COMPARACIÓN
# =========================

def compare(name, a, b):
    diff = np.abs(a - b)
    print(f"{name}:")
    print(f"  max  diff: {diff.max():.6e}")
    print(f"  mean diff: {diff.mean():.6e}")
    print(f"  std  diff: {diff.std():.6e}")

# =========================
# EXTRACCIÓN PARCIAL
# =========================

def extract_Tp(frame):
    Py_rot = frame.transpose(1,0,2)[::-1]

    slice_line = Py_rot[H_PX, R1:R2, 1]
    m = np.argmin(slice_line)
    center_x = R1 + m
    border_x = center_x + R_X
    
    Py_rgb = Py_rot[:, center_x:border_x, :].transpose(2,0,1)

    return Py_rgb

# =========================
# TEST PRINCIPAL
# =========================

def run_test(video_path, num_frames=1):
    decoder = decode(video_path)

    for i in range(num_frames):
        try:
            frame = next(decoder)
        except StopIteration:
            break

        print(f"\n================ FRAME {i} ================")

        Py_rgb = extract_Tp(frame)

        for c in range(3):
            Tp = Py_rgb[c]

            out_interp2d = resize_temp_interp2d(R_EXP, Z_EXP, Tp)
            out_rgi      = resize_temp_rgi(R_EXP, Z_EXP, Tp)
            out_fast     = resize_temp_fast(Tp)

            print(f"\n[Canal {c}]")

            compare("interp2d vs RGI", out_interp2d, out_rgi)
            compare("RGI vs FAST", out_rgi, out_fast)
            compare("interp2d vs FAST", out_interp2d, out_fast)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Ruta al video")
    parser.add_argument("--frames", type=int, default=1, help="Cantidad de frames a probar")

    args = parser.parse_args()

    run_test(args.video, args.frames)
    