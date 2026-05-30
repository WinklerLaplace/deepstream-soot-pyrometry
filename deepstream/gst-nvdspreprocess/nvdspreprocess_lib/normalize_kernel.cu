#include <cuda_runtime.h>

__constant__ float c_means[3];
__constant__ float c_stds[3];
__constant__ float c_inv_xmax;

__global__
void rgba_to_chw_norm_kernel(
    const unsigned char* __restrict__ rgba,
    float*               __restrict__ chw,
    int H, int W,
    int pitch_bytes)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const unsigned char* row = rgba + y * pitch_bytes;
    int hw = H * W;
    int idx = y * W + x;

    float rgba_r = (float)row[x * 4 + 0];
    float rgba_g = (float)row[x * 4 + 1];
    float rgba_b = (float)row[x * 4 + 2];

    chw[0 * hw + idx] = (rgba_b * c_inv_xmax - c_means[0]) / c_stds[0];
    chw[1 * hw + idx] = (rgba_g * c_inv_xmax - c_means[1]) / c_stds[1];
    chw[2 * hw + idx] = (rgba_r * c_inv_xmax - c_means[2]) / c_stds[2];
}

extern "C"
void init_preprocess_constants(
    float inv_xmax,
    const float* means,   // host pointer, 3 floats
    const float* stds)    // host pointer, 3 floats
{
    cudaMemcpyToSymbol(c_inv_xmax, &inv_xmax, sizeof(float));
    cudaMemcpyToSymbol(c_means,    means,     3 * sizeof(float));
    cudaMemcpyToSymbol(c_stds,     stds,      3 * sizeof(float));
}

extern "C"
void launch_preprocess(
    const unsigned char* rgba_dev,
    float*               chw_dev,
    int H, int W,
    int pitch_bytes)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y);

    rgba_to_chw_norm_kernel<<<blocks, threads>>>(
        rgba_dev, chw_dev, H, W, pitch_bytes);
}
