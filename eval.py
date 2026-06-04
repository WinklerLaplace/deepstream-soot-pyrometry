#!/usr/bin/env -S bash -c '"`dirname $(dirname $(dirname $0))`/env/bin/python" "$0" "$@"'
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.functions import destandarize
from utils.load_data import MyDataLoader
from matplotlib.ticker import MaxNLocator
import argparse
import time
from matplotlib.gridspec import GridSpec
from pathlib import Path
import cv2
from flask import Flask, Response
import threading

from utils.processing import process_llamas_image
from utils.processing import decode
from utils.processing import process_llamas_frame
from utils.processing import process_llamas_frame_prof

from PIL import Image

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from utils.utils import *

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Using CPU')
else:
    print('CUDA is available. Using GPU')
device = torch.device("cuda:0" if train_on_gpu else "cpu")
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

# --------------------------------------------
#  VARIABLES GLOBALES 
# --------------------------------------------

# Aumentar tamaño de fuente globalmente
plt.rcParams.update({
    'font.size': 14,  # Tamaño de fuente global
    'axes.titlesize': 16,  # Tamaño de los títulos de los ejes
    'axes.labelsize': 16,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 14,  # Tamaño de los valores en el eje X
    'ytick.labelsize': 14,  # Tamaño de los valores en el eje Y
    'legend.fontsize': 14,  # Tamaño de la fuente en la leyenda
    'figure.titlesize': 14  # Tamaño del título de la figura
})

base_out = Path("outputs/visual")
render_ts_dir = base_out / "render_ts_rgb"
render_centerline_dir = base_out / "render_centerline"
display_ts_dir = base_out / "display_ts_rgb"
display_centerline_dir = base_out / "display_centerline"

TS_MEAN = 1861.5075235004497
TS_STD  = 296.8565934989852
T_MIN = 1500.0
T_MAX = 2150.0

# --------------------------------------------
#  EVALUACIÓN: DATASET SINTÉTICO
# --------------------------------------------

def eval(opt, model):
    # LOAD DATA
    my_data_loader = MyDataLoader()
    x_test, y_test, y_mean, y_std, fs_test, r_test, z_test= my_data_loader.load_test_data()

    x_test = torch.tensor(x_test).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)

    # Eval 
    with torch.no_grad():
        output = model(x_test)

    x_test = x_test.cpu()
    y_test = y_test.cpu()

    y_test_pred = destandarize(output, y_mean, y_std)
    y_test_pred = y_test_pred.squeeze(1)

    y_test_pred = y_test_pred.cpu()

    #N = random.randint(0,len(x_test)-1)
    N = 11

    y_test[N][fs_test[N] <= 0.05e-6] = 0
    y_test_pred[N][fs_test[N] <= 0.05e-6] = 0
    print(N)

    plt.rcParams['figure.figsize'] = [12, 4]

    plt.subplot(171)
    plt.imshow(x_test[0,0,:,:], cmap = 'jet')#, vmax=x_test.max() , vmin=x_test.min())
    plt.title('$U-Net$')
    plt.colorbar()
    plt.subplots_adjust()

    plt.subplot(172)
    plt.imshow(x_test[0,1,:,:], cmap = 'jet')#, vmax=x_test.max(), vmin=x_test.min())
    plt.title('$U-Net$')
    plt.colorbar()
    plt.subplots_adjust()

    plt.subplot(173)
    plt.imshow(x_test[0,2,:,:], cmap = 'jet')#, vmax=x_test.max(), vmin=x_test.min())
    plt.title('$U-Net$')
    plt.colorbar()
    plt.subplots_adjust()

    plt.subplot(174)
    plt.imshow(y_test[N,:,:], cmap = 'jet', vmin=1500, vmax=2205)
    plt.title('$Groundtruth$')
    plt.colorbar()

    plt.subplot(175)
    plt.imshow(y_test_pred[N], cmap = 'jet', vmin=1500, vmax=2205)
    plt.title('$U-Net$')
    plt.colorbar()
    plt.subplots_adjust()

    abs_err = np.abs(y_test[N] - y_test_pred[N])
    plt.subplot(176)
    plt.imshow(abs_err, cmap = 'jet', vmin=0, vmax=50)
    plt.title('$\Delta T_{s}$')
    plt.colorbar(ticks=MaxNLocator(6))

    print('Abs. error max:', abs_err.max())
    print('Abs. error min:', abs_err.min())
    print('Abs. error mean:', abs_err.mean())
    print('Abs. error stddev:', abs_err.std())
    print('Abs. error %:', abs_err.mean()*100/y_test[N].mean())

    plt.savefig('outputs/img/eval.png')
    plt.show()

# --------------------------------------------
#  EVALUACIÓN: COMPARACIÓN ENTRE MODELOS
# --------------------------------------------

def compare(opt,model1,model2):
    print("compare two models")
    data = preprocess_data(opt)    
    #---------------------------  model 1 -------------------------------------#
    with torch.no_grad():
        start_time = time.time()
        output_1 = model1(torch.tensor(data["Py_exp_interp"]).float().to(device))
        end_time = time.time()
    print(f"Tiempo de ejecución modelo 1: {end_time - start_time} segundos")

    t_cgan_caseC_1 = destandarize(output_1, data["y_mean"], data["y_std"])[0,0,:,:].cpu().numpy()
    t_cgan_caseC_1 = np.ma.masked_where(data["mask"], t_cgan_caseC_1[::-1])
    
    rmse = root_mean_squared_error(data["t_emi"], t_cgan_caseC_1)
    print("RMSE 1: ", rmse)

    # ----------------- model 2 ---------------------------------------------------#
    with torch.no_grad():
        start_time = time.time() 
        output_2 = model2(torch.tensor(data["Py_exp_interp"]).float().to(device))
        end_time = time.time() 

    print(f"Tiempo de ejecución modelo 2: {end_time - start_time} segundos")
    
    t_cgan_caseC_2 = destandarize(output_2, data["y_mean"], data["y_std"])[0,0,:,:].cpu().numpy()
    t_cgan_caseC_2 = np.ma.masked_where(data["mask"], t_cgan_caseC_2[::-1])
    
    rmse = root_mean_squared_error(data["t_emi"], t_cgan_caseC_2)
    print("RMSE 2: ", rmse)
    
    #-------------------- PLOTS --------------------------------------------------#

    if opt.case == 'A':
        y_min, y_max, t_max = 1, 3.0, 2200
    elif opt.case == 'B':
        y_min, y_max, t_max = 1, 3.5, 2200
    elif opt.case == 'C':
        y_min, y_max, t_max = 1, 5.5, 2100

    # Graficar
    abs_err = t_cgan_caseC_1 - t_cgan_caseC_2

    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(1, 5, width_ratios=[0.1, 0.2, 0.5, 0.5, 0.5], wspace=0.1, hspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    axes[1].axis("off")  
    axes[1].set_frame_on(False)
    for ax in axes:
        ax.set_facecolor("darkgrey")

    referencia = axcontourf(axes[2], data["r_emi"], data["z_emi"], t_cgan_caseC_1, rf'$T_{{s}}$ model 1', levels=np.linspace(1500, t_max, 50), Y_MIN=y_min, Y_MAX=y_max)
    axcontourf(axes[3], data["r"], data["z"], t_cgan_caseC_2, rf'$T_{{s}}$ model 2', levels=np.linspace(1500, t_max, 50), Y_MIN=y_min, Y_MAX=y_max, show_axes=False)
    diferencia = axcontourf(axes[4], data["r"], data["z"], abs_err, r'$\Delta_{T_{s}}$', levels=np.linspace(-100, 100, 50), CMAP='bwr', Y_MIN=y_min, Y_MAX=y_max, show_axes=False)
    
    cbar_ref = fig.colorbar(referencia, cax=axes[0], location='left', ticks=MaxNLocator(6))
    cbar_ref.ax.yaxis.set_ticks_position('left')
    fig.colorbar(diferencia, ticks=MaxNLocator(6))

    plt.savefig(f'outputs/img/diferencia_modelos_{opt.case}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# --------------------------------------------
#  EVALUACIÓN: COMPARACIÓN EXTENDIDA
# --------------------------------------------

def compare_extended(opt, model_unet, model_attention_unet, 
                     model_unet_trt_fp32, model_unet_trt_fp16, model_unet_trt_int8, 
                     model_attention_unet_trt_fp32,
                     model_attention_unet_trt_fp16,
                     model_attention_unet_trt_int8):
    print("Comparando modelos base y optimizados")

    # LOAD DATA
    my_data_loader = MyDataLoader()
    _, _, y_mean, y_std, _, _, _ = my_data_loader.load_test_data()
    if opt.case == 'A':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_A()
    elif opt.case == 'B':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Py, r, z = my_data_loader.load_data_exp_B()
    elif opt.case == 'C':
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Sy_cal, r, z = my_data_loader.load_data_exp_C()
    #elif opt.case == 'D':
    #    Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, Sy_cal, r, z = my_data_loader.load_data_exp_D()     
    else: # data test
        Py_exp_interp,t_emi,t_bemi, r_emi, z_emi, t_emi, r, z = my_data_loader.load_data_exp()
    Py_exp_interp = torch.tensor(Py_exp_interp).float().to(device)

    mask = t_emi<1
    t_emi = np.ma.masked_where(mask, t_emi)
    t_bemi = np.ma.masked_where(mask, t_bemi)

    def model_inference(model, data, name=""):
        with torch.no_grad():
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            print(f"Tiempo de ejecución {name}: {end_time - start_time:.4f} s")
        output = destandarize(output, y_mean, y_std)[0, 0, :, :].cpu().numpy()[::-1]
        return np.ma.masked_where(mask, output)

    # Inferencias UNet
    unet_output = model_inference(model_unet, Py_exp_interp, "UNet Base")
    unet_trt_fp32_output = model_inference(model_unet_trt_fp32, Py_exp_interp, "UNet TRT fp32")
    unet_trt_fp16_output = model_inference(model_unet_trt_fp16, Py_exp_interp, "UNet TRT fp16")
    unet_trt_int8_output = model_inference(model_unet_trt_int8, Py_exp_interp, "UNet TRT int8")

    # ===== DEBUG FP32 vs FP16 UNet =====
    diff_unet = np.max(np.abs(unet_trt_fp32_output - unet_trt_fp16_output))
    print(f"[DEBUG] Max |UNet fp32 - fp16| = {diff_unet:.6e}")

    # Inferencias Attention UNet
    attention_unet_output = model_inference(model_attention_unet, Py_exp_interp, "Attention UNet Base")
    attention_unet_trt_fp32_output = model_inference(model_attention_unet_trt_fp32, Py_exp_interp, "Attention UNet TRT FP32")
    attention_unet_trt_fp16_output = model_inference(model_attention_unet_trt_fp16, Py_exp_interp, "Attention UNet TRT fp16")
    attention_unet_trt_int8_output = model_inference(model_attention_unet_trt_int8, Py_exp_interp, "Attention UNet TRT int8")

    # ===== DEBUG FP32 vs FP16 Attention UNet =====
    diff_att = np.max(np.abs(attention_unet_trt_fp32_output - attention_unet_trt_fp16_output))
    print(f"[DEBUG] Max |AttUNet fp32 - fp16| = {diff_att:.6e}")

    # Función para calcular error absoluto
    def abs_error(base, optimized):
        error = base - optimized
        error = np.clip(error, -100, 100)
        return error

    # Configurar figura
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 8, width_ratios=[0.1,0.3,0.5, 0.5, 0.5, 0.5,0.2,0.1],wspace=0.1,hspace=0.35)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(8)] for i in range(2)]

    for row in axes:
        for ax in row:
            ax.set_facecolor("darkgray")

    for i in [0, 1]:
        for j in [0,1,6,7]:
            axes[i][j].axis("off")  
            axes[i][j].set_frame_on(False)

    if opt.case == 'A':
        y_min, y_max, t_max = 1, 3.0, 2200
    elif opt.case == 'B':
        y_min, y_max, t_max = 1, 3.5, 2200
    elif opt.case == 'C':
        y_min, y_max, t_max = 1, 5.5, 2100
 
    # Plot para UNet
    unet = axcontourf(axes[0][2], r, z, unet_output, 'Modelo Base', levels=np.linspace(1500, t_max, 50),CMAP='inferno',Y_MAX=y_max, Y_MIN=y_min)
    unet_fp32 = axcontourf(axes[0][3], r, z, abs_error(unet_output, unet_trt_fp32_output), '$\Delta_t$ TRT fp32', levels=np.linspace(-100, 100, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False)
    unet_fp16 = axcontourf(axes[0][4], r, z, abs_error(unet_output, unet_trt_fp16_output), '$\Delta_t$ TRT fp16', levels=np.linspace(-100, 100, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False)
    unet_int8 = axcontourf(axes[0][5], r, z, abs_error(unet_output, unet_trt_int8_output), '$\Delta_t$ TRT int8', levels=np.linspace(-100, 100, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False,ftitle="U-Net")

    # Plot para Attention UNet
    attunet = axcontourf(axes[1][2], r, z, attention_unet_output, 'Modelo Base', levels=np.linspace(1500, t_max, 50),CMAP='inferno',Y_MAX=y_max, Y_MIN=y_min)
    attunet_fp32 = axcontourf(axes[1][3], r, z, abs_error(attention_unet_output, attention_unet_trt_fp32_output), '$\Delta_t$ TRT fp32', levels=np.linspace(-100, 100, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False)
    attunet_fp16 = axcontourf(axes[1][4], r, z, abs_error(attention_unet_output, attention_unet_trt_fp16_output), '$\Delta_t$ TRT fp16', levels=np.linspace(-100, 100, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False)
    attunet_int8 = axcontourf(axes[1][5], r, z, abs_error(attention_unet_output, attention_unet_trt_int8_output), '$\Delta_t$ TRT int8', levels=np.linspace(-100, 100, 50), CMAP='bwr',Y_MAX=y_max, Y_MIN=y_min,show_axes=False,ftitle="Att. U-Net")

    # Colorbars
    cbar_ax_left = fig.add_subplot(gs[:, 0])  # Colorbar izquierdo en ambas filas
    cbar_ax_right = fig.add_subplot(gs[:, 7])  # Colorbar derecho en ambas filas
    
    cbar_unet = fig.colorbar(unet, cax=cbar_ax_left, ticks=MaxNLocator(6))
    cbar_unet.ax.yaxis.set_ticks_position('left') 
    cbar_unet.ax.set_title(r'$T$ [K]', fontsize=16)
    cbar_trt = fig.colorbar(unet_int8, cax=cbar_ax_right, ticks=MaxNLocator(6))
    cbar_trt.ax.set_title(r'$\Delta T$ [K]', fontsize=16)

    fig.tight_layout()
    #plt.savefig('outputs/img/compare_extended.png')
    plt.savefig(f'outputs/img/compare_extended_case_{opt.case}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def compare_all(opt):
    model_unet = load_model(opt,'unet', 'weights/unet.pth')
    model_attention_unet = load_model(opt,'attunet', 'weights/attunet.pth')
    unet_fp32 = load_model(opt,'tensorrt', 'weights/unet_fp32.engine')
    unet_fp16 = load_model(opt,'tensorrt', 'weights/unet_fp16.engine')
    unet_int8 = load_model(opt,'tensorrt', 'weights/unet_int8.engine')

    attunet_fp32 = load_model(opt,'tensorrt', 'weights/attunet_fp32.engine')
    attunet_fp16 = load_model(opt,'tensorrt', 'weights/attunet_fp16.engine')
    attunet_int8 = load_model(opt,'tensorrt', 'weights/attunet_int8.engine')

    compare_extended(opt, model_unet, model_attention_unet, 
                     unet_fp32, unet_fp16, unet_int8, 
                     attunet_fp32,attunet_fp16,attunet_int8)

# --------------------------------------------
#  MÉTRICA DE PRECISIÓN: REGRESSION ACCURACY 
# --------------------------------------------

def eval_exp(opt, model):
    model.eval()
    # Data experimental de la condicion de llama (ya sea EMI o MAE)
    data = preprocess_data(opt) 
    # Definir valores de Y_MIN, Y_MAX y t_max según el tipo de experimento y el caso
    exp_name = "EMI"
    if opt.case == 'A':
        y_min, y_max, t_max = 1, 3.0, 2200
    elif opt.case == 'B':
        y_min, y_max, t_max = 1, 3.5, 2200
    elif opt.case == 'C':
        y_min, y_max, t_max = 1, 5.5, 2100
        exp_name = "MAE"

    if "experimental" in opt.dataset:
        rmse_values = []
        imagenes = sorted(os.listdir(opt.dataset))
        for img_name in imagenes:
            img_path = os.path.join(opt.dataset, img_name)
            img_data = process_experimental_input(opt, img_path)
            img_data = torch.tensor([img_data]).float().to(device)
            if torch.isnan(img_data[0]).any() or torch.isinf(img_data[0]).any():
                #print("\n-------------------------\ncontinue\n--------------------------------\n")
                continue

            with torch.no_grad():
                output = model(img_data)
            t_cgan_caseC = destandarize(output, data["y_mean"], data["y_std"])[0,0,:,:].cpu().numpy()
            t_cgan_caseC = np.ma.masked_where(data["mask"], t_cgan_caseC[::-1])

            #print("Valores de salida (mín, máx):", t_cgan_caseC.min().item(), t_cgan_caseC.max().item())
            rmse = root_mean_squared_error(data["t_emi"], t_cgan_caseC)
            rmse_values.append(rmse)

        # Cálculo de estadísticas
        rmse_values = np.array(rmse_values)
        rmse_promedio = np.mean(rmse_values)
        rmse_std = np.std(rmse_values)
        rmse_max = np.max(rmse_values)
        rmse_min = np.min(rmse_values)

        print(f"RMSE Promedio: {rmse_promedio:.4f}")
        print(f"Desviación Estándar del RMSE: {rmse_std:.4f}")
        print(f"Máximo RMSE: {rmse_max:.4f}")
        print(f"Mínimo RMSE: {rmse_min:.4f}")

    else:
        # Realiza una evaluacion unicamente sobre una imagen experimental contenida en data.
        with torch.no_grad():
            output = model(torch.tensor(data["Py_exp_interp"]).float().to(device))
        t_cgan_caseC = destandarize(output, data["y_mean"], data["y_std"])[0,0,:,:].cpu().numpy()
        t_cgan_caseC = np.ma.masked_where(data["mask"], t_cgan_caseC[::-1])
        
        rmse = root_mean_squared_error(data["t_emi"], t_cgan_caseC)
        print("RMSE: ", rmse)

    # Graficar solo la última figura evaluada si se evaluan todas las figuras del dataset experimental
    abs_err = t_cgan_caseC - data["t_emi"]
    abs_err = np.clip(abs_err, -100, 100)
            
    # Graficar
    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(1, 5, width_ratios=[0.1, 0.2, 0.5, 0.5, 0.5], wspace=0.1, hspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    axes[1].axis("off")  
    axes[1].set_frame_on(False)
    for ax in axes:
        ax.set_facecolor("darkgrey")

    title = rf'$T_{{s}}$ ' + opt.weights.split("/")[-1].split(".")[0]
    referencia = axcontourf(axes[2], data["r_emi"], data["z_emi"], data["t_emi"], rf'$T_{{s}}$ {exp_name}', levels=np.linspace(1500, t_max, 50), Y_MIN=y_min, Y_MAX=y_max)
    axcontourf(axes[3], data["r"], data["z"], t_cgan_caseC, title, levels=np.linspace(1500, t_max, 50), Y_MIN=y_min, Y_MAX=y_max, show_axes=False)
    diferencia = axcontourf(axes[4], data["r"], data["z"], abs_err, r'$\Delta_{T_{s}}$', levels=np.linspace(-100, 100, 50), CMAP='bwr', Y_MIN=y_min, Y_MAX=y_max, show_axes=False)
    
    cbar_ref = fig.colorbar(referencia, cax=axes[0], ticks=MaxNLocator(6))
    cbar_ref.ax.yaxis.set_ticks_position('left')
    fig.colorbar(diferencia, ticks=MaxNLocator(6))

    plt.savefig(f'outputs/img/eval_experiment_{title}_{opt.case}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# --------------------------------------------
#  MÉTRICA DE PRECISIÓN: REGRESSION CLOSENESS
# --------------------------------------------

def closeness(opt, model, engines):
    model.eval()
    for engine in engines.values():
        engine.eval()

    porcentajes = [0.005, 0.01, 0.1, 0.2, 0.5, 1]

    engine_stats = {
        name: {
            'total': 0,
            'close_counts': [0] * len(porcentajes)
        }
        for name in engines.keys()
    }

    # Encontrar imágenes
    dataset_path = Path(opt.dataset)
    imagenes = sorted(dataset_path.rglob("*.tiff"))

    print(f"[INFO] Imágenes encontradas: {len(imagenes)}")
    print("[INFO] Preprocesando imágenes.")

    # Cacheo para procesar una única vez
    inputs_cache = []

    for img_path in imagenes:
        data = process_llamas_image(str(img_path))
        data = torch.tensor([data]).float().to(device)
        inputs_cache.append(data)
    
    print("[INFO] Imágenes procesadas.")
    print("[INFO] Caché de tensores listo.")
    print("[INFO] Iniciando cálculo de tolerancias.")

    # Primer recorrido: Cálculo de tolerancias
    outputs_all_list = []
    with torch.no_grad():
        for data in inputs_cache:
            out_base = model(data)
            outputs_all_list.append(out_base.cpu())
            for engine in engines.values():
                out_eng = engine(data)
                outputs_all_list.append(out_eng.cpu())
    outputs_all = torch.cat(outputs_all_list).flatten().numpy()
    max_value = np.percentile(np.abs(outputs_all), 90)
    rtols = [p * max_value for p in porcentajes]
    print("[INFO] Iniciando cálculo de regression closeness.")

    # Segundo recorrido: Cálculo de regression closeness
    with torch.no_grad():
        for data in inputs_cache:
            out_base = model(data)
            num_elementos_por_imagen = out_base.numel()
            for name, engine in engines.items():
                out_eng = engine(data)
                engine_stats[name]['total'] += 1
                for idx, rtol in enumerate(rtols):
                    close_values = torch.isclose(
                        out_base, out_eng, atol=rtol, rtol=0
                    )
                    engine_stats[name]['close_counts'][idx] += close_values.sum().item()

    # Tabla final
    header = [f"atol {porcentajes[i]}={rtols[i]:.5f}" for i in range(len(porcentajes))]
    table = "| engine | " + " | ".join(header) + " |\n"
    table += "|" + "--------|" * (len(header) + 1) + "\n"
    for name, stats in engine_stats.items():
        total = stats['total'] * num_elementos_por_imagen
        percentages = [
            f"{100.0 * c / total:.2f}%" if total > 0 else "0.00%"
            for c in stats['close_counts']
        ]
        table += f"| {name} | " + " | ".join(percentages) + " |\n"

    print("\nRegression Closeness (img_preprocess)")
    print(table)

def load_closeness(opt):
    model_unet = load_model(opt,'unet', 'weights/unet.pth')
    model_attention_unet = load_model(opt,'attunet', 'weights/attunet.pth')
    unet_fp32 = load_model(opt,'tensorrt', 'weights/unet_fp32.engine')
    unet_fp16 = load_model(opt,'tensorrt', 'weights/unet_fp16.engine')
    unet_int8 = load_model(opt,'tensorrt', 'weights/unet_int8.engine')

    attunet_fp32 = load_model(opt,'tensorrt', 'weights/attunet_fp32.engine')
    attunet_fp16 = load_model(opt,'tensorrt', 'weights/attunet_fp16.engine')
    attunet_int8 = load_model(opt,'tensorrt', 'weights/attunet_int8.engine')

    closeness(opt, model_unet, {'fp32':unet_fp32,'fp16':unet_fp16,'int8':unet_int8})  
    closeness(opt, model_attention_unet, {'fp32':attunet_fp32,'fp16':attunet_fp16,'int8':attunet_int8})

# --------------------------------------------
#  MÉTRICA DE RENDIMIENTO: LATENCIA
# --------------------------------------------

def latency(opt):
    model = load_model(opt, opt.model, opt.weights)
    model.eval()

    dataset_path = Path(opt.dataset)
    videos = sorted(dataset_path.rglob("*.mp4"))

    if not videos:
        print("No se encontraron archivos válidos.")
        return None, None

    device = torch.device("cuda")

    # Eventos CUDA para profiling interno en procesos de GPU
    start_h2d = torch.cuda.Event(enable_timing=True)
    end_h2d = torch.cuda.Event(enable_timing=True)

    start_inf = torch.cuda.Event(enable_timing=True)
    end_inf = torch.cuda.Event(enable_timing=True)

    tiempos = []
    total_frames = 0

    for video_path in videos:

        decoder = decode(str(video_path))

        while True:
            try:
                t0 = time.perf_counter()

                # Decode
                frame = next(decoder)
                t_decode = time.perf_counter()

                # Preprocess
                tensor = process_llamas_frame_prof(frame)
                t_pre = time.perf_counter()

                # Batch (size = 1)
                batch_array = np.expand_dims(tensor, axis=0).astype(np.float32)
                t_np = time.perf_counter()

                batch_tensor = torch.from_numpy(batch_array) 
                t_from_numpy = time.perf_counter()

                # Transferencia CPU -> GPU
                start_h2d.record()
                batch_tensor = batch_tensor.to(device)  
                end_h2d.record()

                # Inferencia
                start_inf.record()
                with torch.no_grad():
                    out = model(batch_tensor)
                end_inf.record()

                # Sincronización global 
                torch.cuda.synchronize()
                t_inf = time.perf_counter()

                # Postprocesamiento + display
                out_np = out.cpu().numpy()
                Ts = out_np[0, 0]
                mask = compute_mask_from_tensor(tensor)
                panel_ts = display_ts_rgb(tensor, Ts, mask, total_frames)
                panel_cl = display_centerline(Ts, mask, total_frames)
                panel_cl = cv2.resize(panel_cl, (panel_cl.shape[1], panel_ts.shape[0]))
                combined = np.hstack([panel_ts, panel_cl])
                latest_frame = combined
                t_end = time.perf_counter()

                # Métricas
                h2d_ms = start_h2d.elapsed_time(end_h2d)
                inf_ms = start_inf.elapsed_time(end_inf)

                tiempos.append(t_end - t0)
                total_frames += 1

                if total_frames % 50 == 0:

                    print("\n[PROFILING LATENCIA - END TO END]")
                    print(f"Decode:      {(t_decode - t0):.6f}s")
                    print(f"Preprocess:  {(t_pre - t_decode):.6f}s")
                    print(f"Expand+Cast: {(t_np - t_pre):.6f}s")
                    print(f"From numpy:  {(t_from_numpy - t_np):.6f}s")
                    print(f"CPU->GPU:    {h2d_ms/1000:.6f}s")
                    print(f"Inferencia:  {inf_ms/1000:.6f}s")
                    print(f"Post+Disp:   {(t_end - t_inf):.6f}s")
                    print(f"Total:       {(t_end - t0):.6f}s")

            except StopIteration:
                break

            except Exception as e:
                print(f"Error en frame {total_frames}: {e}")
                continue

    if not tiempos:
        return None, None

    lat_max = max(tiempos)
    lat_avg = sum(tiempos) / len(tiempos)

    return lat_max, lat_avg

# --------------------------------------------
#  MÉTRICA DE RENDIMIENTO: THROUGHPUT
# --------------------------------------------

def throughput(opt):
    model = load_model(opt, opt.model, opt.weights)
    model.eval()

    dataset_path = Path(opt.dataset)
    videos = sorted(dataset_path.rglob("*.mp4"))

    if not videos:
        print("No se encontraron archivos .mp4")
        return None, None

    batch_size = opt.batch_size
    device = torch.device("cuda")

    total_frames = 0
    batch_data = []

    start_global = time.perf_counter()

    for video_path in videos:

        for frame in decode(str(video_path)):

            try:
                tensor = process_llamas_frame(frame)
                batch_data.append(tensor)
                total_frames += 1

                if len(batch_data) == batch_size:
                    batch_array = np.stack(batch_data).astype(np.float32)
                    batch_tensor = torch.from_numpy(batch_array).pin_memory()
                    batch_tensor_gpu = batch_tensor.to(device, non_blocking=True)

                    with torch.no_grad():
                        out = model(batch_tensor_gpu)

                    # Post-procesamiento
                    out_np = out.cpu().numpy()
                    start_id = total_frames - len(batch_data)

                    # Display
                    for i in range(len(batch_data)):
                        frame_id = start_id + i
                        tensor_i = batch_data[i]
                        Ts_i = out_np[i, 0]
                        mask_i = compute_mask_from_tensor(tensor_i)
                        panel_ts = display_ts_rgb(tensor_i, Ts_i, mask_i, frame_id)
                        panel_cl = display_centerline(Ts_i, mask_i, frame_id)
                        panel_cl = cv2.resize(panel_cl, (panel_cl.shape[1], panel_ts.shape[0]))
                        combined = np.hstack([panel_ts, panel_cl])
                        latest_frame = combined

                    batch_data = []

            except Exception as e:
                print(f"Error en frame {total_frames}: {e}")
                continue

    # Último batch
    if batch_data:
        batch_array = np.stack(batch_data).astype(np.float32)
        batch_tensor = torch.from_numpy(batch_array).pin_memory()
        batch_tensor_gpu = batch_tensor.to(device, non_blocking=True)

        with torch.no_grad():
            out = model(batch_tensor_gpu)

        out_np = out.cpu().numpy()
        start_id = total_frames - len(batch_data)

        for i in range(len(batch_data)):
            frame_id = start_id + i
            tensor_i = batch_data[i]
            Ts_i = out_np[i, 0]
            mask_i = compute_mask_from_tensor(tensor_i)
            panel_ts = display_ts_rgb(tensor_i, Ts_i, mask_i, frame_id)
            panel_cl = display_centerline(Ts_i, mask_i, frame_id)
            panel_cl = cv2.resize(panel_cl, (panel_cl.shape[1], panel_ts.shape[0]))
            combined = np.hstack([panel_ts, panel_cl])
            latest_frame = combined

    # Sincronización final
    torch.cuda.synchronize()

    end_global = time.perf_counter()
    total_time = end_global - start_global

    if total_time == 0 or total_frames == 0:
        return None, None

    thr_avg = total_frames / total_time

    print(f"Throughput: {thr_avg:.2f} fps")
    print(f"({total_frames} frames en {total_time:.2f}s)")

    return thr_avg, None
    
# --------------------------------------------
#  EJECUCIÓN: VISUALIZACIÓN PERSISTENTE
# --------------------------------------------    

# ===================== TEMP =====================
 
app = Flask(__name__)

def generate():
    global latest_frame
    latest_frame = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(latest_frame, "Waiting for frames...",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255,255,255), 2)
    import time

    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        
        _, buffer = cv2.imencode('.jpg', latest_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_server():
    app.run(host='0.0.0.0', port=5000, threaded=True)

threading.Thread(target=start_server, daemon=True).start()

import io
from PIL import Image

def fig_to_rgb_array(fig):

    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white"
    )

    buf.seek(0)

    img = np.array(
        Image.open(buf).convert("RGB")
    )

    buf.close()

    return img

# ===================== TEMP =====================

scale_z = 5.5 / 128
scale_r = 0.6 / 32
n_ticks_z = 12
n_ticks_r = 3
    
def visualization(opt):
    global latest_frame # TEMP
    model = load_model(opt, opt.model, opt.weights)
    model.eval()

    dataset_path = Path(opt.dataset)
    videos = sorted(dataset_path.rglob("*.mp4"))

    if not videos:
        print("No se encontraron archivos .mp4")
        return

    batch_size = opt.batch_size
    device = torch.device("cuda")

    total_frames = 0
    batch_data = []

    print("[SIMULATION] Iniciando ejecución...")

    for video_path in videos:

        for frame in decode(str(video_path)):

            try:
                tensor = process_llamas_frame(frame)
                batch_data.append(tensor)
                total_frames += 1

                if len(batch_data) == batch_size:
                    batch_array = np.stack(batch_data).astype(np.float32)
                    batch_tensor = torch.from_numpy(batch_array).pin_memory()
                    batch_tensor_gpu = batch_tensor.to(device, non_blocking=True)

                    with torch.no_grad():
                        output = model(batch_tensor_gpu)

                    output_np = output.cpu().numpy()

                    for i in range(len(batch_data)):
                        tensor_i = batch_data[i]
                        Ts_i = output_np[i, 0]
                        frame_id = total_frames - len(batch_data) + i
                        mask_i = compute_mask_from_tensor(tensor_i)
                        if frame_id % 25 == 0:
                            # render_rgb_ts(tensor_i, Ts_i, mask_i, frame_id, render_ts_dir)
                            # render_centerline(Ts_i, mask_i, frame_id, render_centerline_dir)
                            
                            # ===================== TEMP =====================

                            panel_ts = render_rgb_ts(
                                tensor_i,
                                Ts_i,
                                mask_i,
                                frame_id,
                                render_ts_dir
                            )

                            panel_cl = render_centerline(
                                Ts_i,
                                mask_i,
                                frame_id,
                                render_centerline_dir
                            )

                            h1, w1 = panel_ts.shape[:2]
                            h2, w2 = panel_cl.shape[:2]

                            target_h = max(h1, h2)

                            def pad_to_height(img, target_h):
                                h, w = img.shape[:2]

                                if h >= target_h:
                                    return img

                                pad_top = (target_h - h) // 2
                                pad_bottom = target_h - h - pad_top

                                return cv2.copyMakeBorder(
                                    img,
                                    pad_top,
                                    pad_bottom,
                                    0,
                                    0,
                                    cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255)
                                )

                            panel_ts = pad_to_height(panel_ts, target_h)
                            panel_cl = pad_to_height(panel_cl, target_h)

                            combined = np.concatenate(
                                [panel_ts, panel_cl],
                                axis=1
                            )

                            # Flask stream frame
                            latest_frame = cv2.cvtColor(
                                combined,
                                cv2.COLOR_RGB2BGR
                            )
                            
                            # ===================== TEMP =====================
                            
                            

                    batch_data = []

            except Exception as e:
                print(f"Error en frame {total_frames}: {e}")
                continue

    # Último batch
    if batch_data:
        batch_array = np.stack(batch_data).astype(np.float32)
        batch_tensor = torch.from_numpy(batch_array).pin_memory()
        batch_tensor_gpu = batch_tensor.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(batch_tensor_gpu)

        output_np = output.cpu().numpy()

        for i in range(len(batch_data)):
            tensor_i = batch_data[i]
            Ts_i = output_np[i, 0]
            frame_id = total_frames - len(batch_data) + i
            mask_i = compute_mask_from_tensor(tensor_i)
            if frame_id % 25 == 0:
                # render_rgb_ts(tensor_i, Ts_i, mask_i, frame_id, render_ts_dir)
                # render_centerline(Ts_i, mask_i, frame_id, render_centerline_dir)
                            
                # ===================== TEMP =====================
                
                panel_ts = render_rgb_ts(
                    tensor_i,
                    Ts_i,
                    mask_i,
                    frame_id,
                    render_ts_dir
                )

                panel_cl = render_centerline(
                    Ts_i,
                    mask_i,
                    frame_id,
                    render_centerline_dir
                )

                h1, w1 = panel_ts.shape[:2]
                h2, w2 = panel_cl.shape[:2]

                target_h = max(h1, h2)

                def pad_to_height(img, target_h):
                    h, w = img.shape[:2]

                    if h >= target_h:
                        return img

                    pad_top = (target_h - h) // 2
                    pad_bottom = target_h - h - pad_top

                    return cv2.copyMakeBorder(
                        img,
                        pad_top,
                        pad_bottom,
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=(255, 255, 255)
                    )

                panel_ts = pad_to_height(panel_ts, target_h)
                panel_cl = pad_to_height(panel_cl, target_h)

                combined = np.concatenate(
                    [panel_ts, panel_cl],
                    axis=1
                )

                # Flask stream frame
                latest_frame = cv2.cvtColor(
                    combined,
                    cv2.COLOR_RGB2BGR
                )
                
                # ===================== TEMP =====================

    torch.cuda.synchronize()

    print(f"[SIMULATION] Finalizado. Total frames procesados: {total_frames}")    

def render_rgb_ts(tensor_chw, Ts, mask, frame_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    H, W = tensor_chw.shape[1:]

    # RGB normalizado
    rgb = tensor_chw[[2, 1, 0], :, :]
    rgb_masked = rgb.copy()
    rgb_masked[:, ~mask] = np.nan  

    if np.all(np.isnan(rgb_masked)):
        rgb_norm = np.zeros_like(rgb)
    else:
        rgb_min = np.nanmin(rgb_masked)
        rgb_max = np.nanmax(rgb_masked)
        rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)

    rgb_norm = np.nan_to_num(rgb_norm)

    # Figura
    fig = plt.figure(figsize=(8.5, 4.8), dpi=120)
    gs = GridSpec(2, 4, height_ratios=[20, 0.4], wspace=0.02, hspace=0.08)

    # Orden
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_r  = fig.add_subplot(gs[0, 1])
    ax_g  = fig.add_subplot(gs[0, 2])
    ax_b  = fig.add_subplot(gs[0, 3])

    # Ts
    Ts_vis = Ts.copy()
    Ts_vis[~mask] = np.nan
    Ts_kelvin = Ts_vis * TS_STD + TS_MEAN
    
    im_ts = ax_ts.imshow(Ts_kelvin, cmap="inferno", vmin=T_MIN, vmax=T_MAX)
    ax_ts.set_xlabel("r (cm)")
    ax_ts.set_ylabel("z (cm)")
    ax_ts.set_xlim(0, W)
    ax_ts.set_ylim(H, 0)
    xticks = np.linspace(0, W, n_ticks_r)
    yticks = np.linspace(0, H, n_ticks_z)
    ax_ts.set_xticks(xticks)
    ax_ts.set_yticks(yticks)
    ax_ts.set_xticklabels([f"{x*scale_r:.2f}" for x in xticks])
    ax_ts.set_yticklabels([f"{(H - y)*scale_z:.2f}" for y in yticks])

    # Canales RGB
    rgb_axes = [ax_r, ax_g, ax_b]
    rgb_titles = [r"$P_R$", r"$P_G$", r"$P_B$"]

    for i, ax in enumerate(rgb_axes):
        ch = rgb_norm[i].copy()
        ch[~mask] = np.nan

        im = ax.imshow(ch, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(rgb_titles[i])
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbars
    cb_height = 0.012     
    cb_offset_top = 0.14

    # Ts colorbar 
    pos_ts = ax_ts.get_position()
    pos_r  = ax_r.get_position()
    pos_b  = ax_b.get_position()
    
    cax_ts = fig.add_axes([pos_ts.x0 + 0.1 * pos_ts.width, pos_ts.y1 + cb_offset_top, 0.8 * pos_ts.width, cb_height])

    cbar_ts = fig.colorbar(im_ts, cax=cax_ts, orientation="horizontal")
    cbar_ts.set_ticks([T_MIN, T_MAX])
    cbar_ts.set_ticklabels([f"{T_MIN:.0f}", f"{T_MAX:.0f}"])
    cbar_ts.set_label(r"$T_s$ [K]", fontsize=7)
    cbar_ts.ax.xaxis.set_label_position('top')
    cbar_ts.ax.xaxis.set_ticks_position('bottom') 
    cbar_ts.ax.tick_params(labelsize=6)

    # RGB colorbar 
    left = pos_r.x0
    right = pos_b.x1
    width = right - left

    cax_rgb = fig.add_axes([left + 0.1 * width, pos_r.y1 + cb_offset_top, 0.8 * width, cb_height])

    cbar_rgb = fig.colorbar(im, cax=cax_rgb, orientation="horizontal")
    cbar_rgb.set_ticks([0, 1])
    cbar_rgb.set_ticklabels(["0", "1"])
    cbar_rgb.set_label(r"$P_{R,G,B}$ Normalized", fontsize=7)
    cbar_rgb.ax.xaxis.set_label_position('top')
    cbar_rgb.ax.xaxis.set_ticks_position('bottom')
    cbar_rgb.ax.tick_params(labelsize=6)

    # Ajuste de ejes
    ax_ts.tick_params(labelsize=8)
    ax_ts.xaxis.label.set_size(9)
    ax_ts.yaxis.label.set_size(9)

    # Guardar
    # plt.savefig(os.path.join(out_dir, f"{frame_id}_rgb_ts.png"), dpi=300, bbox_inches="tight")

    # plt.close(fig)

    # ===================== TEMP =====================
    
    img = fig_to_rgb_array(fig)
    
    plt.close(fig)
    
    return img

    # ===================== TEMP =====================

def render_centerline(Ts, mask, frame_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    H, W = Ts.shape

    # Ts en Kelvin
    Ts_vis = Ts.copy()
    Ts_vis[~mask] = np.nan
    Ts_kelvin = Ts_vis * TS_STD + TS_MEAN

    # Centerline
    col = 0
    centerline = Ts_kelvin[:, col].copy()
    mask_line = mask[:, col]
    centerline[~mask_line] = np.nan

    z = np.arange(H) * scale_z

    # Rango dinámico en Z
    valid_idx = np.where(~np.isnan(centerline))[0]

    if valid_idx.size > 0:
        z_min = scale_z * (10 * np.floor(valid_idx.min() / 10))
        z_max = scale_z * (10 * np.ceil(valid_idx.max() / 10))
    else:
        z_min, z_max = 0, H

    # Rango dinámico en T
    valid_vals = centerline[~np.isnan(centerline)]

    if valid_vals.size > 0:
        t_min = np.nanmin(valid_vals)
        t_max = np.nanmax(valid_vals)

        T_MIN_VIS = 100 * np.floor(t_min / 100)
        T_MAX_VIS = 100 * np.ceil(t_max / 100)
    else:
        T_MIN_VIS, T_MAX_VIS = 1700, 2000

    # Figura
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)

    if valid_vals.size > 0:
        ax.plot(z, centerline, linewidth=2.5, label="Centerline")

    # Ejes
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Centerline Temperature Profile")

    ax.set_xlim(z_min, z_max)
    ax.set_ylim(T_MIN_VIS, T_MAX_VIS)

    # Estética
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()

    # Guardar
    # plt.savefig(os.path.join(out_dir, f"{frame_id}_centerline.png"), dpi=300, bbox_inches="tight")

    # plt.close(fig)    
    
    # ===================== TEMP =====================
    
    img = fig_to_rgb_array(fig)
    
    plt.close(fig)

    return img

    # ===================== TEMP =====================
    
# --------------------------------------------
#  EJECUCIÓN: SIMULACIÓN EN TIEMPO REAL
# --------------------------------------------    

app = Flask(__name__)

def generate():
    global latest_frame
    latest_frame = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(latest_frame, "Waiting for frames...",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255,255,255), 2)
    import time

    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        
        _, buffer = cv2.imencode('.jpg', latest_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_server():
    app.run(host='0.0.0.0', port=5000, threaded=True)

threading.Thread(target=start_server, daemon=True).start()
    
def simulate(opt):
    global latest_frame
    
    model = load_model(opt, opt.model, opt.weights)
    model.eval()

    dataset_path = Path(opt.dataset)
    videos = sorted(dataset_path.rglob("*.mp4"))

    if not videos:
        print("No se encontraron archivos .mp4")
        return

    batch_size = opt.batch_size
    device = torch.device("cuda")

    total_frames = 0
    batch_data = []

    print("[SIMULATION] Iniciando ejecución...")

    for video_path in videos:

        for frame in decode(str(video_path)):

            try:
                tensor = process_llamas_frame(frame)
                batch_data.append(tensor)
                total_frames += 1

                if len(batch_data) == batch_size:
                    batch_array = np.stack(batch_data).astype(np.float32)
                    batch_tensor = torch.from_numpy(batch_array).pin_memory()
                    batch_tensor_gpu = batch_tensor.to(device, non_blocking=True)

                    with torch.no_grad():
                        output = model(batch_tensor_gpu)
                        
                    output_np = output.cpu().numpy()

                    for i in range(len(batch_data)):
                        frame_id = total_frames - len(batch_data) + i
                        tensor_i = batch_data[i]
                        Ts_i = output_np[i, 0]
                        mask_i = compute_mask_from_tensor(tensor_i)
                        # display_ts_rgb(tensor_i, Ts_i, mask_i, frame_id)
                        # display_centerline(Ts_i, mask_i, frame_id)
                        panel_ts = display_ts_rgb(tensor_i, Ts_i, mask_i, frame_id)
                        panel_cl = display_centerline(Ts_i, mask_i, frame_id)
                        panel_cl = cv2.resize(panel_cl, (panel_cl.shape[1], panel_ts.shape[0]))
                        combined = np.hstack([panel_ts, panel_cl])
                        latest_frame = combined

                    batch_data = []

            except Exception as e:
                print(f"Error en frame {total_frames}: {e}")
                continue

    if batch_data:
        batch_array = np.stack(batch_data).astype(np.float32)
        batch_tensor = torch.from_numpy(batch_array).pin_memory()
        batch_tensor_gpu = batch_tensor.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(batch_tensor_gpu)
            
        output_np = output.cpu().numpy()

        for i in range(len(batch_data)):
            frame_id = total_frames - len(batch_data) + i
            tensor_i = batch_data[i]
            Ts_i = output_np[i, 0]
            mask_i = compute_mask_from_tensor(tensor_i)
            # display_ts_rgb(tensor_i, Ts_i, mask_i, frame_id)
            # display_centerline(Ts_i, mask_i, frame_id)
            panel_ts = display_ts_rgb(tensor_i, Ts_i, mask_i, frame_id)
            panel_cl = display_centerline(Ts_i, mask_i, frame_id)
            combined = np.hstack([panel_ts, panel_cl])
            latest_frame = combined

    torch.cuda.synchronize()

    print(f"[SIMULATION] Finalizado. Total frames procesados: {total_frames}")    

def display_ts_rgb(tensor_chw, Ts, mask, frame_id):
    H, W = tensor_chw.shape[1:]

    # RGB Normalizado
    rgb = tensor_chw[[2, 1, 0], :, :].copy()
    rgb[:, ~mask] = np.nan

    if np.all(np.isnan(rgb)):
        rgb_norm = np.zeros_like(rgb)
    else:
        rgb_min = np.nanmin(rgb)
        rgb_max = np.nanmax(rgb)
        rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)

    rgb_norm = np.nan_to_num(rgb_norm)

    # Ts 
    Ts_vis = Ts.copy()
    Ts_vis[~mask] = np.nan
    Ts_kelvin = Ts_vis * TS_STD + TS_MEAN

    Ts_norm = (Ts_kelvin - T_MIN) / (T_MAX - T_MIN)
    Ts_norm = np.clip(Ts_norm, 0, 1)
    Ts_norm = np.nan_to_num(Ts_norm)

    # Conversión a imagen
    Ts_img = (Ts_norm * 255).astype(np.uint8)
    Ts_color = cv2.applyColorMap(Ts_img, cv2.COLORMAP_INFERNO)
    Ts_color[~mask] = (255, 255, 255)

    rgb_imgs = []
    for i in range(3):
        ch = (rgb_norm[i] * 255).astype(np.uint8)
        ch_color = cv2.applyColorMap(ch, cv2.COLORMAP_VIRIDIS)
        ch_color[~mask] = (255, 255, 255)
        rgb_imgs.append(ch_color)

    # Panel
    panel = np.hstack([Ts_color, rgb_imgs[0], rgb_imgs[1], rgb_imgs[2]])

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(panel, "Ts", (10, 20), font, 0.5, (0,0,0), 1)
    cv2.putText(panel, "R", (W + 10, 20), font, 0.5, (0,0,0), 1)
    cv2.putText(panel, "G", (2*W + 10, 20), font, 0.5, (0,0,0), 1)
    cv2.putText(panel, "B", (3*W + 10, 20), font, 0.5, (0,0,0), 1)

    # Métricas
    if np.any(~np.isnan(Ts_kelvin)):
        valid_vals = Ts_kelvin[~np.isnan(Ts_kelvin)]
        t_mean = np.mean(valid_vals)
        cv2.putText(panel, f"Mean: {t_mean:.0f}K", (10, H-10), font, 0.4, (0,0,0), 1)

    # Colorbars
    cb_width = panel.shape[1]
    cb_height = 20
    gradient = np.linspace(0, 255, cb_width, dtype=np.uint8)
    gradient = np.tile(gradient, (cb_height, 1))
    cb_ts = cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)
    cb_rgb = cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
    
    cv2.putText(cb_ts, f"{T_MIN:.0f}", (5, 15), font, 0.4, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(cb_ts, f"{T_MAX:.0f}", (cb_width - 60, 15), font, 0.4, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(cb_rgb, "0", (5, 15), font, 0.4, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(cb_rgb, "1", (cb_width - 20, 15), font, 0.4, (255,255,255), 1, cv2.LINE_AA)

    # Output
    final = np.vstack([panel, cb_ts, cb_rgb])
    # os.makedirs(display_ts_dir, exist_ok=True)
    # cv2.imwrite(os.path.join(display_ts_dir, f"{frame_id}_rgb_ts.png"), final)
    return final

def display_centerline(Ts, mask, frame_id):
    H, W = Ts.shape

    # Centerline real
    col = W // 2
    line = Ts[:, col].copy()
    mask_line = mask[:, col]
    line[~mask_line] = np.nan

    # Conversión a Kelvin
    line = line * TS_STD + TS_MEAN

    # Valores válidos
    valid = ~np.isnan(line)
    if not np.any(valid):
        return

    z = np.arange(H)[valid]
    T = line[valid]

    # Rango 
    z_min, z_max = z.min(), z.max()
    T_min, T_max = T_MIN, T_MAX

    # Canvas
    width = 300
    height = 200
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin = 30

    # Escalas
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)
    T_norm = (T - T_min) / (T_max - T_min + 1e-8)

    # Convertir a coordenadas de imagen
    x = (z_norm * (width - 2*margin) + margin).astype(int)
    y = height - (T_norm * (height - 2*margin) + margin).astype(int)

    # Dibujar curva
    for i in range(len(x)-1):
        cv2.line(canvas, (x[i], y[i]), (x[i+1], y[i+1]), (0,0,255), 2)

    # Dibujar ejes
    cv2.line(canvas, (margin, height-margin), (width-margin, height-margin), (0,0,0), 1)  # X
    cv2.line(canvas, (margin, margin), (margin, height-margin), (0,0,0), 1)              # Y

    # Labels simples
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "z", (width//2, height-5), font, 0.5, (0,0,0), 1)
    cv2.putText(canvas, "T", (5, height//2), font, 0.5, (0,0,0), 1)

    # Valores extremos
    cv2.putText(canvas, f"{T_min:.0f}", (5, height-margin), font, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"{T_max:.0f}", (5, margin+5), font, 0.4, (0,0,0), 1)

    # Guardar
    # os.makedirs(display_centerline_dir, exist_ok=True)
    # cv2.imwrite(os.path.join(display_centerline_dir, f"{frame_id}_centerline.png"), canvas)
    
    return canvas
    
def compute_mask_from_tensor(tensor_chw):
    # Canal G
    G = tensor_chw[1]

    # Llevar a rango positivo 
    G_shift = G - G.min()
    if G_shift.max() > 0:
        G_norm = G_shift / G_shift.max()
    else:
        G_norm = G_shift
        
    G_uint8 = (G_norm * 255).astype(np.uint8)

    # Otsu
    _, mask = cv2.threshold(G_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask.astype(bool)    
    
# --------------------------------------------
#  CONFIGURACIÓN 
# --------------------------------------------
    
def parse_opt():
    parser = argparse.ArgumentParser()

    # Configuración básica
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--dataset', default='datasets/img_preprocess', type=str,
                        help='path a dataset')
    parser.add_argument('--model', default='attunet', type=str,
                        help='modelo a evaluar')
    parser.add_argument('--weights', default='weights/attunet.pth', type=str,
                        help='path a los pesos')

    # Modos de ejecución
    parser.add_argument('--eval', action='store_true',
                        help='Evaluación del dataset sintético')
    parser.add_argument('--compare', action='store_true',
                        help='Comparar dos modelos')
    parser.add_argument('--compare_all', action='store_true',
                        help='Comparar todos los modelos')
    parser.add_argument('--experiment', action='store_true',
                        help='Evaluación de Regression Accuracy')
    parser.add_argument('--closeness', action='store_true',
                        help='Evaluación de Regression Closeness')
    parser.add_argument('--latency', action='store_true',
                        help='Evaluación de latencia (usa batch_size=1)')
    parser.add_argument('--throughput', action='store_true',
                        help='Evaluación de throughput (batch_size>1 recomendado)')
    parser.add_argument('--visualization', action='store_true',
                        help='Ejecución con visualizaciones persistentes')
    parser.add_argument('--simulate', action='store_true',
                        help='Simulación de ejecución en tiempo real')

    # Especificaciones
    parser.add_argument('--case', default='A', type=str,
                        help='condicion de llama')

    return parser.parse_args()    

def main(opt):
    if opt.eval:
        model = load_model(opt, opt.model, opt.weights)
        if model is None:
            print("Error en la carga del modelo.")
            return
        eval(opt, model)
        return

    if opt.compare:
        models = opt.model.split()
        weights = opt.weights.split()
        if len(models) != 2 or len(weights) != 2:
            print("ERROR: Debes especificar exactamente dos modelos y pesos.")
            return
        model1 = load_model(opt, models[0], weights[0])
        model2 = load_model(opt, models[1], weights[1])
        if model1 is None or model2 is None:
            print("Error en la carga de modelos.")
            return
        compare(opt, model1, model2)
        return
    
    if opt.compare_all:
        compare_all(opt)
        return
    
    if opt.experiment:
        model = load_model(opt, opt.model, opt.weights)
        if model is None:
            print("Error en la carga del modelo.")
            return
        eval_exp(opt, model)
        return    
    
    if opt.closeness:
        load_closeness(opt)
        return
    
    if opt.latency:
        if opt.batch_size != 1:
            print("Medir latencia requiere batch_size=1.")
            print("Forzando batch_size=1.")
            opt.batch_size = 1
        l_max, l_ave = latency(opt)
        print("Latencia max: ", l_max, "s")
        print("Latencia ave: ", l_ave, "s")
        return

    if opt.throughput:
        if opt.batch_size <= 0:
            print("Error: Ingrese un batch size válido.")
            return
        if opt.batch_size == 1:
            print("Se recomienda batch_size > 1 para la medición de throughput.")
            print("Reanudando medición.")
        _, thr = throughput(opt)
        print("Throughput ave: ", thr, "inf/s")
        return
    
    if opt.visualization:
        if opt.batch_size <= 0:
            print("Error: Ingrese un batch size válido.")
            return
        else:
            print(f"Generando visualizaciones")
        visualization(opt)
        return
    
    if opt.simulate:
        if opt.batch_size <= 0:
            print("Error: Ingrese un batch size válido.")
            return
        if opt.batch_size == 1:
            print("Aviso: Comportamiento similar a latencia con batch_size={opt.batch_size}")
        else:
            print(f"Simulación con batch_size={opt.batch_size}")
        simulate(opt)
        return

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
