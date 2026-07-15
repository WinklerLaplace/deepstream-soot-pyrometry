# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2
import time
import os
from PIL import Image

from processing_stats import *

# --------------------------------------------
#  VARIABLES GLOBALES Y PRECOMPUTO
# --------------------------------------------

# Dominio físico
R_EXP = np.linspace(0, NX - 1, NX, dtype=np.float32)
Z_EXP = np.linspace(0, NZ - 1, NZ, dtype=np.float32)

# Dominio de interpolación
NEW_R = np.linspace(0, NX - 1, 32, dtype=np.float32)
NEW_Z = np.linspace(0, NZ - 1, 128, dtype=np.float32)

# Precómputo de la geometría de interpolación
def precompute_interp(z, r, new_z, new_r):
    nz, nr = len(z), len(r)

    iz = np.searchsorted(z, new_z) - 1
    ir = np.searchsorted(r, new_r) - 1

    iz = np.clip(iz, 0, nz - 2)
    ir = np.clip(ir, 0, nr - 2)

    z0, z1 = z[iz], z[iz + 1]
    r0, r1 = r[ir], r[ir + 1]

    dz = z1 - z0
    dr = r1 - r0
    dz[dz == 0] = 1e-6
    dr[dr == 0] = 1e-6

    wz = (new_z - z0) / dz
    wr = (new_r - r0) / dr

    wz = wz[:, None]
    wr = wr[None, :]
    iz = iz[:, None]
    ir = ir[None, :]

    return iz, ir, wz, wr

IZ, IR, WZ, WR = precompute_interp(Z_EXP, R_EXP, NEW_Z, NEW_R)

# Normalización
MEANS = np.array([X1_MEAN, X2_MEAN, X3_MEAN], dtype=np.float32)[:, None, None]
STDS  = np.array([X1_STD, X2_STD, X3_STD], dtype=np.float32)[:, None, None]
INV_XMAX = 1.0 / X_MAX

# --------------------------------------------
#  PREPROCESAMIENTO (CONSTRUCCIÓN/PRECISIÓN)
# --------------------------------------------

def process_llamas_image(image_dir):
    # Leer imagen
    ss = cv2.imread(image_dir, cv2.IMREAD_UNCHANGED)

    # Crop
    Py_crop = ss[CENTER_X:BORDER_X, :, :]

    # Rotación
    Py_rot = Py_crop.transpose(1, 0, 2)[::-1]

    # Reordenamiento CHW
    Py_rgb = Py_rot.transpose(2, 0, 1)

    # Interpolación
    Py_exp_interp = np.empty((3, 128, 32), dtype=np.float32)
    for c in range(3):
        Py_exp_interp[c] = resize_temp_fast(Py_rgb[c])

    # Normalización + estandarización
    Py_exp_interp = (Py_exp_interp * INV_XMAX - MEANS) / STDS

    return Py_exp_interp

# --------------------------------------------
#  PREPROCESAMIENTO (PRODUCCIÓN)
# --------------------------------------------

def process_llamas_frame(ss):
    # Crop
    Py_crop = ss[CENTER_X:BORDER_X, :, :]

    # Rotación
    Py_rot = Py_crop.transpose(1, 0, 2)[::-1]

    # Reordenamiento CHW
    Py_rgb = Py_rot.transpose(2, 0, 1)

    # Interpolación
    Py_exp_interp = np.empty((3, 128, 32), dtype=np.float32)
    for c in range(3):
        Py_exp_interp[c] = resize_temp_fast(Py_rgb[c])

    # Normalización + estandarización
    Py_exp_interp = (Py_exp_interp * INV_XMAX - MEANS) / STDS

    return Py_exp_interp

# --------------------------------------------
#  PREPROCESAMIENTO (PROFILING)
# --------------------------------------------

PROF_COUNT = 0

T_ROT = 0.0
T_CENTER = 0.0
T_CROP = 0.0
T_INTERP = 0.0
T_FLIP = 0.0
T_NORM = 0.0

def process_llamas_frame_prof(ss):
    global PROF_COUNT
    global T_ROT, T_CENTER, T_CROP, T_INTERP, T_FLIP, T_NORM

    t0 = time.perf_counter()

    Py_crop = ss[CENTER_X:BORDER_X, :, :]

    t1 = time.perf_counter()

    '''
    slice_line = Py_rot[H_PX, R1:R2, 1]
    m = np.argmin(slice_line)
    center_x = R1 + m
    border_x = center_x + R_X
    t2 = time.perf_counter()
    '''
    
    Py_rot = Py_crop.transpose(1, 0, 2)[::-1]
    Py_rgb = Py_rot.transpose(2, 0, 1)
    t3 = time.perf_counter()

    Py_exp_interp = np.empty((3, 128, 32), dtype=np.float32)
    for c in range(3):
        Py_exp_interp[c] = resize_temp_fast(Py_rgb[c])
    t4 = time.perf_counter()

    Py_exp_interp = Py_exp_interp[:, ::-1, :]
    t5 = time.perf_counter()

    Py_exp_interp = (Py_exp_interp * INV_XMAX - MEANS) / STDS
    t6 = time.perf_counter()

    T_ROT += t1 - t0
    # T_CENTER += t2 - t1
    # T_CROP += t3 - t2
    T_CROP += t3 - t1
    T_INTERP += t4 - t3
    T_FLIP += t5 - t4
    T_NORM += t6 - t5

    PROF_COUNT += 1

    if PROF_COUNT % 50 == 0:
        print("\n[PREPROCESS PROFILING - Promedio últimos 50 frames]")
        print(f"Rotación:      {T_ROT/50:.6f}s")
        # print(f"Centro:        {T_CENTER/50:.6f}s")
        print(f"Crop:          {T_CROP/50:.6f}s")
        print(f"Interpolación: {T_INTERP/50:.6f}s")
        print(f"Flip:          {T_FLIP/50:.6f}s")
        print(f"Norm+Std:      {T_NORM/50:.6f}s")
        # print(f"Total:         {(T_ROT+T_CENTER+T_CROP+T_INTERP+T_FLIP+T_NORM)/50:.6f}s")
        print(f"Total:         {(T_ROT+T_CROP+T_INTERP+T_FLIP+T_NORM)/50:.6f}s")

        # T_ROT = T_CENTER = T_CROP = 0.0
        T_ROT = T_CROP = 0.0
        T_INTERP = T_FLIP = T_NORM = 0.0

    return Py_exp_interp

# --------------------------------------------
#  INTERPOLACIÓN
# --------------------------------------------

def resize_temp_fast(Tp):
    v00 = Tp[IZ, IR]
    v01 = Tp[IZ, IR + 1]
    v10 = Tp[IZ + 1, IR]
    v11 = Tp[IZ + 1, IR + 1]

    out = (
        (1 - WZ) * (1 - WR) * v00 +
        (1 - WZ) * WR       * v01 +
        WZ       * (1 - WR) * v10 +
        WZ       * WR       * v11
    )

    return out.astype(np.float32)

# --------------------------------------------
#  DECODIFICACIÓN
# --------------------------------------------

def decode(video_path):
    """
    Decodificación simple (baseline limpia).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

    cap.release()

'''
from queue import Queue
from threading import Thread

def decode_thread(video_path, queue):
    """
    Versión experimental (no usada en pipeline final).
    """
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        queue.put(frame)

    queue.put(None)
    cap.release()
'''
