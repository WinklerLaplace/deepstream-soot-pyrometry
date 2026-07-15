#pragma once
#include <cuda_runtime.h>

// Static background of the Ts-RGB panel: channel labels ("Ts/R/G/B"),
// both full colorbars (gradient + range values), and the fixed
// "Mean:"/"K" text. Built ONCE in Init(); if the panel dimensions
// change, simply call this again with the new values.
void build_panels_static_background(
    uchar4* d_bg, int w, int h,
    int label_h, int img_h, int footer_h,
    int panel_sub_w, float t_min, float t_max);

// Mean temperature (Kelvin) within the mask. Single-block reduction,
// same pattern as minmax_reduce_kernel.
__global__ void mean_reduce_kernel(
    const float* __restrict__ ts_raw,
    const unsigned char* __restrict__ mask,
    float ts_mean, float ts_std,
    int N,
    float* __restrict__ out_mean);

// Converts the mean (float, Kelvin) into glyph codes for stamping.
__global__ void format_mean_kernel(
    const float* __restrict__ mean_val,
    int* __restrict__ codes, int* __restrict__ n_chars, int max_chars);
    