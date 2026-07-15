#pragma once
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <string>

// =================================================================
// Unified text style
// =================================================================
// The single place where the style of ALL text in the pipeline is
// defined: both the static text drawn with OpenCV (axes, channel
// labels, colorbars, via put_text_autofit) and the dynamic glyph
// atlas (build_glyph_atlas, below). Changing these constants affects
// both systems equally -- previously there were two separate style
// criteria (one for the z-axis via the atlas, another for everything
// else via direct OpenCV calls), which produced visually inconsistent
// text across elements of the same canvas.
static constexpr int FONT_FACE      = cv::FONT_HERSHEY_SIMPLEX;
static constexpr int FONT_THICKNESS = 2;
static constexpr int FONT_AA        = cv::LINE_AA;

// Draws `text` horizontally centered at (cx, cy_baseline), automatically
// shrinking the scale until it fits within a max_w x max_h box. Used for
// ALL static text (host/OpenCV): axes, channel names, colorbar range
// values, etc. Returns the final scale used, useful if the caller needs
// to position something relative to the actual rendered text size.
inline double put_text_autofit(
    cv::Mat& canvas, const std::string& text,
    int cx, int cy_baseline, int max_w, int max_h)
{
    double font_scale = 1.5;
    int baseline = 0;
    cv::Size sz;
    for (int iter = 0; iter < 30; iter++) {
        sz = cv::getTextSize(text, FONT_FACE, font_scale, FONT_THICKNESS, &baseline);
        if (sz.width <= max_w && sz.height <= max_h) break;
        font_scale *= 0.9;
    }
    int tx = cx - sz.width / 2;
    cv::putText(canvas, text, {tx, cy_baseline}, FONT_FACE, font_scale, {0,0,0}, FONT_THICKNESS, FONT_AA);
    return font_scale;
}

// =================================================================
// Glyph atlas — digits 0-9 and '.' (index 10) ONLY
// =================================================================
// Fixed labels (T-axis, "z [mm]", "T[K]", channel names, "No data")
// are drawn with OpenCV directly onto the static background in Init()
// via put_text_autofit -- they don't use this atlas, since they never
// change per frame. The atlas exists only for text whose content
// varies per frame: the z-axis tick labels and the mean temperature,
// which must be stamped on GPU without any host roundtrip.
//
// N_GLYPHS and the cell size are derived from the output canvas
// dimensions at init time (see thermal_render.cu), so increasing the
// output resolution simply regenerates the atlas at the right scale --
// there are no fixed bitmaps baked into the binary.
static constexpr int N_GLYPHS = 11; // '0'..'9' + '.'

// Builds the atlas via OpenCV (host) and uploads it to the GPU. Uses
// the same FONT_FACE/FONT_THICKNESS/FONT_AA as put_text_autofit, so
// dynamic text (z-axis, mean value) looks visually identical to the
// static text. glyph_w/glyph_h: cell size in pixels, computed by the
// caller based on the current canvas resolution.
void build_glyph_atlas(unsigned char* d_atlas /* N_GLYPHS*glyph_w*glyph_h, pre-allocated device ptr */,
                        int glyph_w, int glyph_h);

// Stamps a string of glyph codes (from the atlas) onto the canvas,
// horizontally centered at px_center, with its baseline at py_baseline.
// One CUDA block per label; used for GPU-resident dynamic text.
__global__ void draw_glyphs_kernel(
    uchar4*              __restrict__ canvas,
    int canvas_w, int canvas_h,
    const unsigned char* __restrict__ atlas,
    int glyph_w, int glyph_h,
    const int*           __restrict__ codes,   // n_labels * max_chars, 255 = empty slot
    const int*           __restrict__ n_chars, // n_labels
    const int*           __restrict__ px_center,
    const int*           __restrict__ py_baseline,
    int n_labels, int max_chars);

// =================================================================
// Numeric formatting -> glyph codes (device-side)
// =================================================================
// Converts dynamic values (mean temperature, z-axis positions) into
// glyph code sequences consumable by draw_glyphs_kernel. There is no
// snprintf-equivalent usable inside a kernel, so digit formatting is
// implemented by hand.

// Bridge kernel: formats the z-axis tick labels (up to N_Z_LABELS
// float values, see thermal_centerline.cuh) in a single launch, one
// thread per label.
__global__ void format_z_labels_kernel(
    const float* __restrict__ values,
    const int*   __restrict__ n_labels,
    int* __restrict__ codes,
    int* __restrict__ n_chars,
    int max_chars = 6);

// Both formatting functions below are defined inline in the header
// (rather than as normal __global__/extern functions) so they can be
// called from any .cu file without requiring relocatable device code
// compilation (-rdc=true), which would force a build-flag change
// across every object in the project.

// Converts an integer (e.g. rounded mean temperature) to glyph codes,
// digit by digit, no sign or decimals.
static __device__ __forceinline__ int format_int(int value, int* out_codes, int max_chars)
{
    int tmp[6], n = 0, v = value;
    if (v == 0) tmp[n++] = 0;
    while (v > 0 && n < 6) { tmp[n++] = v % 10; v /= 10; }
    int m = 0;
    for (int i = n-1; i >= 0 && m < max_chars; i--) out_codes[m++] = tmp[i];
    for (int i = m; i < max_chars; i++) out_codes[i] = 255; // 255 = empty slot (padding)
    return m;
}

// Converts a float to glyph codes with a single fixed decimal (e.g.
// "3.2" for z-axis positions). Rounds the decimal digit and carries
// into the integer part when needed (e.g. 2.96 -> "3.0").
static __device__ __forceinline__ int format_float_1decimal(float value, int* out_codes, int max_chars)
{
    int int_part  = (int)value;
    int frac_part = (int)((value - int_part) * 10.f + 0.5f);
    if (frac_part >= 10) { frac_part = 0; int_part += 1; }

    int tmp[3], n_int = 0;
    int v = int_part;
    if (v == 0) { tmp[n_int++] = 0; }
    while (v > 0 && n_int < 3) { tmp[n_int++] = v % 10; v /= 10; }

    int n = 0;
    for (int i = n_int - 1; i >= 0 && n < max_chars; i--) out_codes[n++] = tmp[i];
    if (n < max_chars) out_codes[n++] = 10; // code 10 = '.'
    if (n < max_chars) out_codes[n++] = frac_part;
    for (int i = n; i < max_chars; i++) out_codes[i] = 255;
    return n;
}
