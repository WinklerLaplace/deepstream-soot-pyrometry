#include "thermal_centerline.cuh"
#include "thermal_text.cuh"
#include <opencv2/imgproc.hpp>
#include <climits>
#include <vector>
#include <cstring>

// =================================================================
// build_centerline_static_background
// =================================================================
// Fixed background of the centerline plot: axes, T-axis ticks and
// labels (constant, frame-independent), and the fixed "z [mm]"/"T[K]"
// axis labels. All text uses put_text_autofit to guarantee the same
// visual style as the dynamic z-axis labels (rendered via the glyph
// atlas with the same FONT_FACE/THICKNESS) and the Ts-RGB panel labels.
void build_centerline_static_background(
    uchar4* d_bg, int cl_w, int cl_h, int margin,
    float t_min, float t_max, int n_ticks_r)
{
    cv::Mat canvas(cl_h, cl_w, CV_8UC3, cv::Scalar(255,255,255));

    cv::line(canvas, {margin, cl_h-margin}, {cl_w-margin, cl_h-margin}, {0,0,0}, 2);
    cv::line(canvas, {margin, margin},      {margin, cl_h-margin},      {0,0,0}, 2);

    // T-axis ticks at fixed physical steps (rather than n_ticks_r evenly
    // spaced divisions of the range) so the labels land on round numbers
    // (e.g. 1500, 1650, 1800...) instead of arbitrary fractions of the
    // range. n_ticks_r is kept as a parameter for interface compatibility
    // but is no longer used to derive the step.
    float tick_step = 150.0f;
    for (float T_t = t_min; T_t <= t_max + 0.5f; T_t += tick_step) {
        int py = cl_h - margin - (int)(((T_t - t_min)/(t_max - t_min)) * (cl_h - 2*margin));
        cv::line(canvas, {margin-6, py}, {margin, py}, {0,0,0}, 2);
        char buf[8]; snprintf(buf, sizeof(buf), "%.0f", T_t);
        put_text_autofit(canvas, buf, margin/2 - 3, py+6, margin - 10, 20);
    }

    // Fixed axis labels. Generous box (100x24 / margin x24) since these
    // are short strings that don't compete for space with anything else.
    put_text_autofit(canvas, "z [mm]", cl_w/2, cl_h - 6, 100, 24);
    put_text_autofit(canvas, "T[K]",   margin/2, margin - 12, margin - 6, 24);

    std::vector<uchar4> h_bg(cl_w * cl_h);
    for (int y = 0; y < cl_h; y++)
        for (int x = 0; x < cl_w; x++) {
            cv::Vec3b p = canvas.at<cv::Vec3b>(y,x);
            h_bg[y*cl_w+x] = make_uchar4(p[0], p[1], p[2], 255);
        }
    cudaMemcpy(d_bg, h_bg.data(), h_bg.size()*sizeof(uchar4), cudaMemcpyHostToDevice);
}

// =================================================================
// build_no_data_sprite
// =================================================================
void build_no_data_sprite(uchar4* d_sprite, int cl_w, int cl_h)
{
    cv::Mat canvas(cl_h, cl_w, CV_8UC3, cv::Scalar(255,255,255));
    put_text_autofit(canvas, "No data", cl_w/2, cl_h/2, cl_w - 40, 40);

    std::vector<uchar4> h_sp(cl_w * cl_h);
    for (int y = 0; y < cl_h; y++)
        for (int x = 0; x < cl_w; x++) {
            cv::Vec3b p = canvas.at<cv::Vec3b>(y,x);
            h_sp[y*cl_w+x] = make_uchar4(p[0], p[1], p[2], 255);
        }
    cudaMemcpy(d_sprite, h_sp.data(), h_sp.size()*sizeof(uchar4), cudaMemcpyHostToDevice);
}

// =================================================================
// column_span_kernel
// =================================================================
// Single-block min/max reduction over row indices, but only counting
// rows where the sampled column is masked. Same shared-memory tree
// reduction pattern as minmax_reduce_kernel, adapted to track indices
// (rows) instead of values.
__global__ void column_span_kernel(
    const unsigned char* __restrict__ mask,
    int H, int W, int col,
    int* __restrict__ z_span)
{
    __shared__ int s_min[256], s_max[256];
    int tid = threadIdx.x;
    s_min[tid] = (tid < H && mask[tid*W+col]) ? tid : INT_MAX;
    s_max[tid] = (tid < H && mask[tid*W+col]) ? tid : -1;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = min(s_min[tid], s_min[tid+s]);
            s_max[tid] = max(s_max[tid], s_max[tid+s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        z_span[0] = (s_min[0]==INT_MAX) ? -1 : s_min[0];
        z_span[1] = s_max[0];
    }
}

// Maps a row index (z) to a horizontal plot pixel, given the frame's
// masked row range [z_min, z_max].
__device__ __forceinline__ int px_x_dev(int z, int z_min, int z_max, int W, int M) {
    int range = max(z_max - z_min, 1);
    return M + (z - z_min) * (W - 2*M) / range;
}

// Maps a temperature (Kelvin) to a vertical plot pixel, given the
// fixed physical range [t_min, t_max].
__device__ __forceinline__ int px_y_dev(float T, float t_min, float t_max, int H, int M) {
    float v = (T - t_min) / (t_max - t_min);
    return H - M - (int)(v * (H - 2*M));
}

// =================================================================
// draw_z_ticks_kernel
// =================================================================
// One thread per tick candidate. Draws the tick mark geometry for
// every tick, and for every label_stride-th tick additionally records
// its position/value for later glyph rendering.
__global__ void draw_z_ticks_kernel(
    uchar4* __restrict__ canvas,
    int canvas_w, int canvas_h, int margin,
    const int* __restrict__ z_span,
    int n_ticks, int label_stride,
    float scale_z,
    int* __restrict__ label_px_center,
    int* __restrict__ label_py_baseline,
    float* __restrict__ label_values,
    int* __restrict__ n_labels_out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t > n_ticks) return;

    int z_min = z_span[0], z_max = z_span[1];
    if (z_min < 0) { if (t==0) *n_labels_out = 0; return; }

    int z_t = z_min + t * (z_max - z_min) / n_ticks;
    int px  = px_x_dev(z_t, z_min, z_max, canvas_w, margin);

    for (int dy = 0; dy < 5; dy++) {
        int py = canvas_h - margin + dy;
        if (px >= 0 && px < canvas_w && py >= 0 && py < canvas_h)
            canvas[py*canvas_w+px] = make_uchar4(0,0,0,255);
    }

    if (t % label_stride == 0) {
        int label_idx = t / label_stride;

        // Clamp: prevents the centered label text from invading the
        // left margin (where the T-axis lives) or clipping against the
        // right edge. half_text_w is a conservative estimate of the
        // widest label ("5.5", ~3 chars); adjust if GLYPH_W changes.
        const int half_text_w = 20;
        int px_clamped = px;
        if (px_clamped - half_text_w < margin)
            px_clamped = margin + half_text_w;
        if (px_clamped + half_text_w > canvas_w - 4)
            px_clamped = canvas_w - 4 - half_text_w;

        label_px_center[label_idx]   = px_clamped;
        label_py_baseline[label_idx] = canvas_h - margin + 28;
        label_values[label_idx]      = z_t * scale_z;
        if (t == (n_ticks / label_stride) * label_stride)
            atomicMax(n_labels_out, label_idx + 1);
    }
}

// =================================================================
// draw_curve_kernel
// =================================================================
// One thread per row. Draws the segment connecting row `row` to row
// `row+1` (both masked) via a DDA line: since z is mapped to plot
// width, the horizontal distance between consecutive rows can exceed
// one pixel, so a naive per-row single-pixel draw would leave gaps.
__global__ void draw_curve_kernel(
    uchar4* __restrict__ canvas,
    const float* __restrict__ ts_raw,
    const unsigned char* __restrict__ mask,
    float ts_mean, float ts_std,
    float t_min, float t_max,
    int H, int W, int col,
    int canvas_w, int canvas_h, int margin,
    const int* __restrict__ z_span)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H - 1) return;
    int z_min = z_span[0], z_max = z_span[1];
    if (z_min < 0) return;
    if (!mask[row*W+col] || !mask[(row+1)*W+col]) return;

    float T0 = ts_raw[row*W+col]     * ts_std + ts_mean;
    float T1 = ts_raw[(row+1)*W+col] * ts_std + ts_mean;
    int x0 = px_x_dev(row,   z_min, z_max, canvas_w, margin);
    int y0 = px_y_dev(T0, t_min, t_max, canvas_h, margin);
    int x1 = px_x_dev(row+1, z_min, z_max, canvas_w, margin);
    int y1 = px_y_dev(T1, t_min, t_max, canvas_h, margin);

    int dx = x1-x0, dy = y1-y0;
    int steps = max(abs(dx), abs(dy)); if (steps==0) steps=1;
    // Simple thickness: paint a 3x3 block per DDA step so the curve
    // remains visible at native output resolution.
    for (int s = 0; s <= steps; s++) {
        int x = x0 + dx*s/steps, y = y0 + dy*s/steps;
        for (int ox = -1; ox <= 1; ox++)
            for (int oy = -1; oy <= 1; oy++) {
                int px = x+ox, py = y+oy;
                if (px>=0 && px<canvas_w && py>=0 && py<canvas_h)
                    canvas[py*canvas_w+px] = make_uchar4(0,0,200,255);
            }
    }
}

// =================================================================
// blit_no_data_kernel
// =================================================================
__global__ void blit_no_data_kernel(
    uchar4* __restrict__ canvas,
    const uchar4* __restrict__ sprite,
    int canvas_w, int canvas_h,
    const int* __restrict__ z_span)
{
    if (z_span[0] >= 0) return; // data exists this frame, sprite not needed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= canvas_w * canvas_h) return;
    canvas[idx] = sprite[idx];
}
