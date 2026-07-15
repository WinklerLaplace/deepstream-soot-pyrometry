#include "thermal_panels.cuh"
#include "thermal_colormaps.cuh"
#include "thermal_text.cuh"
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cstring>

// =================================================================
// build_panels_static_background
// =================================================================
// Static background of the Ts-RGB panel: channel labels ("Ts R G B"),
// and the footer with the "Mean: ... K" text plus both colorbars
// (INFERNO for Ts, VIRIDIS for R/G/B) with their range values. The
// footer is divided into explicit vertical bands within footer_h,
// instead of hand-accumulated offsets, so the content never overflows
// the assigned area even if footer_h changes (e.g. when the output
// resolution changes).
void build_panels_static_background(
    uchar4* d_bg, int w, int h,
    int label_h, int img_h, int footer_h,
    int panel_sub_w, float t_min, float t_max)
{
    // The colormap LUTs live in __constant__ memory and cannot be
    // indexed directly from host code, so they're copied to host
    // arrays once, here.
    unsigned char h_inferno[256][3], h_viridis[256][3];
    cudaMemcpyFromSymbol(h_inferno, kInferno, sizeof(h_inferno));
    cudaMemcpyFromSymbol(h_viridis, kViridis, sizeof(h_viridis));

    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(255,255,255));

    // Channel labels above each sub-panel. Box: sub-panel width minus
    // margin, height capped to the top label band.
    const char* names[4] = {"Ts","R","G","B"};
    for (int i = 0; i < 4; i++) {
        int cx = i*panel_sub_w + panel_sub_w/2;
        put_text_autofit(canvas, names[i], cx, label_h - 10, panel_sub_w - 10, label_h - 16);
    }

    // ── Footer: explicit vertical budget ──
    // footer_h is split into 5 fixed bands (proportions can be tuned
    // if any element looks too tight/loose once rendered):
    //   20% "Mean: ... K"
    //   14% Ts colorbar
    //   15% Ts range labels
    //   14% R/G/B colorbar (with 6% breathing room from the row above)
    //   15% 0/1 labels
    int footer_y0 = label_h + img_h;
    int bar_x0 = 20, bar_w = w - 40;

    int mean_h     = (int)(footer_h * 0.20f);
    int bar_h      = (int)(footer_h * 0.14f);
    int gap_h      = (int)(footer_h * 0.06f);
    int labelrow_h = (int)(footer_h * 0.15f);

    int y_mean = footer_y0 + mean_h - 6;
    int y_bar1 = footer_y0 + mean_h + gap_h;
    int y_lbl1 = y_bar1 + bar_h + labelrow_h - 4;
    int y_bar2 = y_lbl1 + 6;
    int y_lbl2 = y_bar2 + bar_h + labelrow_h - 4;
    // y_lbl2 must stay <= h; if footer_h is too small for this budget,
    // content would clip against the bottom edge -- revisit the
    // proportions above if that ever happens.

    // "Mean:" + a reserved gap for the dynamic number + "K".
    // mean_label_w and gap_w define where mean_px must point in
    // thermal_render.cu -- kept in sync by hand (see the comment in
    // GpuBuffers::init).
    int mean_label_w = 130;
    int gap_w = 90;
    put_text_autofit(canvas, "Mean:", bar_x0 + mean_label_w/2, y_mean, mean_label_w, mean_h);
    int k_x0 = bar_x0 + mean_label_w + gap_w;
    put_text_autofit(canvas, "K", k_x0 + 15, y_mean, 30, mean_h);

    // Ts colorbar (INFERNO) + [t_min, t_max] range
    for (int x = 0; x < bar_w; x++) {
        unsigned int bin = (unsigned int)((float)x/(bar_w-1) * 255.f + 0.5f);
        cv::Vec3b col(h_inferno[bin][0], h_inferno[bin][1], h_inferno[bin][2]);
        for (int y = 0; y < bar_h; y++) canvas.at<cv::Vec3b>(y_bar1+y, bar_x0+x) = col;
    }
    char buf[16];
    snprintf(buf, sizeof(buf), "%.0f", t_min);
    put_text_autofit(canvas, buf, bar_x0 + 24, y_lbl1, 60, labelrow_h);
    snprintf(buf, sizeof(buf), "%.0f", t_max);
    put_text_autofit(canvas, buf, bar_x0 + bar_w - 24, y_lbl1, 60, labelrow_h);

    // R/G/B colorbar (VIRIDIS) + normalised [0, 1] range
    for (int x = 0; x < bar_w; x++) {
        unsigned int bin = (unsigned int)((float)x/(bar_w-1) * 255.f + 0.5f);
        cv::Vec3b col(h_viridis[bin][0], h_viridis[bin][1], h_viridis[bin][2]);
        for (int y = 0; y < bar_h; y++) canvas.at<cv::Vec3b>(y_bar2+y, bar_x0+x) = col;
    }
    put_text_autofit(canvas, "0", bar_x0 + 12,         y_lbl2, 30, labelrow_h);
    put_text_autofit(canvas, "1", bar_x0 + bar_w - 12, y_lbl2, 30, labelrow_h);

    // BGR (OpenCV) -> BGRA and upload to device
    std::vector<uchar4> h_bg(w*h);
    for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        cv::Vec3b p = canvas.at<cv::Vec3b>(y,x);
        h_bg[y*w+x] = make_uchar4(p[0],p[1],p[2],255);
    }
    cudaMemcpy(d_bg, h_bg.data(), h_bg.size()*sizeof(uchar4), cudaMemcpyHostToDevice);
}

// =================================================================
// mean_reduce_kernel
// =================================================================
// Single-block reduction, same pattern as minmax_reduce_kernel: each
// thread accumulates a partial sum/count over a strided slice of the
// data, then a tree reduction combines all partial results. Only
// pixels where mask != 0 contribute to the average.
__global__ void mean_reduce_kernel(
    const float* __restrict__ ts_raw,
    const unsigned char* __restrict__ mask,
    float ts_mean, float ts_std,
    int N,
    float* __restrict__ out_mean)
{
    __shared__ float s_sum[256];
    __shared__ int   s_cnt[256];
    int tid = threadIdx.x;
    float lsum = 0.f; int lcnt = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        if (mask[i]) { lsum += ts_raw[i]*ts_std + ts_mean; lcnt++; }
    }
    s_sum[tid]=lsum; s_cnt[tid]=lcnt;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid]+=s_sum[tid+s]; s_cnt[tid]+=s_cnt[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) *out_mean = (s_cnt[0] > 0) ? s_sum[0]/s_cnt[0] : 0.f;
}

// =================================================================
// format_mean_kernel
// =================================================================
// Single-thread bridge between the reduced mean value and the glyph
// codes consumed by draw_glyphs_kernel. Rounds to the nearest integer
// Kelvin (no decimals needed for this display).
__global__ void format_mean_kernel(
    const float* __restrict__ mean_val,
    int* __restrict__ codes, int* __restrict__ n_chars, int max_chars)
{
    if (threadIdx.x == 0)
        n_chars[0] = format_int((int)(*mean_val + 0.5f), codes, max_chars);
}
