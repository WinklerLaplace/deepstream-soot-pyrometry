#include "thermal_text.cuh"
#include <cstring>
#include <vector>

// =================================================================
// build_glyph_atlas
// =================================================================
// For each glyph (digits 0-9 and '.'), finds the largest font scale
// that fits within a glyph_w x glyph_h cell (2px horizontal / 4px
// vertical margin), centers it, and draws it with the unified style
// (FONT_FACE/FONT_THICKNESS/FONT_AA). The result is uploaded to the
// GPU as a contiguous array of N_GLYPHS cells.
void build_glyph_atlas(unsigned char* d_atlas, int glyph_w, int glyph_h)
{
    std::vector<unsigned char> h_atlas(N_GLYPHS * glyph_w * glyph_h, 0);
    const char* chars = "0123456789.";

    for (int g = 0; g < N_GLYPHS; g++) {
        cv::Mat cell(glyph_h, glyph_w, CV_8UC1, cv::Scalar(0));
        char buf[2] = { chars[g], '\0' };

        // Auto-fit: same mechanism as put_text_autofit, but measured
        // against the atlas's fixed cell size instead of a continuous
        // canvas.
        double font_scale = 1.2;
        int baseline = 0;
        cv::Size sz;
        for (int iter = 0; iter < 30; iter++) {
            sz = cv::getTextSize(buf, FONT_FACE, font_scale, FONT_THICKNESS, &baseline);
            if (sz.width <= glyph_w - 2 && sz.height <= glyph_h - 4) break;
            font_scale *= 0.9;
        }

        int tx = (glyph_w - sz.width) / 2;
        int ty = (glyph_h + sz.height) / 2;
        cv::putText(cell, buf, {tx, ty}, FONT_FACE, font_scale, cv::Scalar(255), FONT_THICKNESS, FONT_AA);

        std::memcpy(h_atlas.data() + g * glyph_w * glyph_h, cell.data, glyph_w * glyph_h);
    }

    cudaMemcpy(d_atlas, h_atlas.data(), h_atlas.size(), cudaMemcpyHostToDevice);
}

// =================================================================
// draw_glyphs_kernel
// =================================================================
// One CUDA block per text label; the block's threads split the total
// area (nc * glyph_w x glyph_h) to stamp each character by reading its
// bitmap from the atlas. Text is horizontally centered on px_center,
// with its baseline at py_baseline (same convention as the host-side
// put_text_autofit, so both systems render consistently).
__global__ void draw_glyphs_kernel(
    uchar4*              __restrict__ canvas,
    int canvas_w, int canvas_h,
    const unsigned char* __restrict__ atlas,
    int glyph_w, int glyph_h,
    const int*           __restrict__ codes,
    const int*           __restrict__ n_chars,
    const int*           __restrict__ px_center,
    const int*           __restrict__ py_baseline,
    int n_labels, int max_chars)
{
    int label = blockIdx.x;
    if (label >= n_labels) return;

    int nc = n_chars[label];
    int total_w = nc * glyph_w;
    int x0 = px_center[label] - total_w / 2;
    int y0 = py_baseline[label] - glyph_h;

    int tid = threadIdx.x;
    int area = glyph_w * glyph_h;
    for (int c = 0; c < nc; c++) {
        int code = codes[label * max_chars + c];
        if (code == 255) continue; // padding slot, nothing to draw

        for (int p = tid; p < area; p += blockDim.x) {
            int gy = p / glyph_w, gx = p % glyph_w;
            unsigned char a = atlas[code * area + p];
            if (a < 128) continue; // simple threshold, no destination AA

            int px = x0 + c * glyph_w + gx;
            int py = y0 + gy;
            if (px >= 0 && px < canvas_w && py >= 0 && py < canvas_h)
                canvas[py * canvas_w + px] = make_uchar4(0, 0, 0, 255);
        }
    }
}

// =================================================================
// format_z_labels_kernel
// =================================================================
// Bridge between the float values computed by draw_z_ticks_kernel
// (thermal_centerline.cu) and the glyph codes consumed by
// draw_glyphs_kernel. One thread per label; labels beyond *n_labels
// (sampled column had insufficient data this frame) are marked with
// n_chars=0 so draw_glyphs_kernel draws nothing for them.
__global__ void format_z_labels_kernel(
    const float* __restrict__ values,
    const int*   __restrict__ n_labels,
    int* __restrict__ codes,
    int* __restrict__ n_chars,
    int max_chars)
{
    int i = threadIdx.x;
    if (i >= *n_labels) { n_chars[i] = 0; return; }
    n_chars[i] = format_float_1decimal(values[i], codes + i*max_chars, max_chars);
}
