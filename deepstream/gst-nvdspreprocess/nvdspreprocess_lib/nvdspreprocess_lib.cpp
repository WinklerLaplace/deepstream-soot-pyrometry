/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <condition_variable>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include "nvdspreprocess_lib.h"
#include "nvdspreprocess_impl.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

extern "C"
void launch_normalize(float* data,
                     int C, int H, int W,
                     float inv_xmax,
                     const float* means,
                     const float* stds);
                     
static CustomCtx *g_ctx = nullptr;

struct CustomCtx
{
  /** Custom initialization parameters */
  CustomInitParams initParams;
  /** Custom mean subtraction and normalization parameters */
  CustomMeanSubandNormParams custom_mean_norm_params;
  /** unique pointer to tensor_impl class instance */
  std::unique_ptr <NvDsPreProcessTensorImpl> tensor_impl;

  NvBufSurface *prealloc_crop_surf = nullptr;  // 2048×150 NV12
  NvBufSurface *prealloc_rot_surf  = nullptr;  // 150×2048 NV12
};

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */
static gboolean
get_absolute_file_path (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[_PATH_MAX + 1];
  gchar abs_real_file_path[_PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      return FALSE;
    }
    g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
  }

  /* Get the absolute path of the config file. */
  if (!realpath (cfg_file_path, abs_cfg_path)) {
    return FALSE;
  }

  /* Remove the file name from the absolute path to get the directory of the
   * config file. */
  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  /* Get the absolute file path from the config file's directory path and
   * relative file path. */
  abs_file_path = g_strconcat (abs_cfg_path, file_path, nullptr);

  /* Resolve the path.*/
  if (realpath (abs_file_path, abs_real_file_path) == nullptr) {
    /* Ignore error if file does not exist and use the unresolved path. */
    if (errno == ENOENT)
      g_strlcpy (abs_real_file_path, abs_file_path, _PATH_MAX);
    else {
      g_free (abs_file_path);
      return FALSE;
    }
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}

NvDsPreProcessStatus
CustomTensorPreparation(CustomCtx* ctx, NvDsPreProcessBatch* batch,
                        NvDsPreProcessCustomBuf*& buf,
                        CustomTensorParams& tensorParam,
                        NvDsPreProcessAcquirer* acquirer)
{
    buf = acquirer->acquire();

    NvDsPreProcessUnit& unit = batch->units[0];

    launch_preprocess(
        reinterpret_cast<const unsigned char*>(unit.converted_frame_ptr),
        reinterpret_cast<float*>(buf->memory_ptr),
        128, 32,
        static_cast<int>(batch->pitch));

    cudaDeviceSynchronize();

    tensorParam.params.network_input_shape[0] =
        static_cast<int>(batch->units.size());

    return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus
CustomAsyncTransformation(NvBufSurface *in_surf,
                          NvBufSurface *out_surf,
                          CustomTransformParams &params)
{
    NvBufSurfTransform_Error err;

    params.transform_config_params.compute_mode = NvBufSurfTransformCompute_GPU;
    params.transform_config_params.gpu_id       = 0;
    params.transform_config_params.cuda_stream  = NULL;

    err = NvBufSurfTransformSetSessionParams(&params.transform_config_params);
    if (err != NvBufSurfTransformError_Success)
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;

    NvBufSurface *crop_surf = g_ctx->prealloc_crop_surf;
    NvBufSurface *rot_surf  = g_ctx->prealloc_rot_surf;

    // CROP
    NvBufSurfTransformRect crop_src = {1061, 0, 2048, 150};
    NvBufSurfTransformRect crop_dst = {0,   0, 2048, 150};

    NvBufSurfTransformParams crop_params;
    memset(&crop_params, 0, sizeof(crop_params));
    crop_params.src_rect       = &crop_src;
    crop_params.dst_rect       = &crop_dst;
    crop_params.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC;

    err = NvBufSurfTransform(in_surf, crop_surf, &crop_params);
    if (err != NvBufSurfTransformError_Success)
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;

    // ROTATE
    NvBufSurfTransformRect rot_src = {0, 0, 2048, 150};
    NvBufSurfTransformRect rot_dst = {0, 0, 150,  2048};

    NvBufSurfTransformParams rot_params;
    memset(&rot_params, 0, sizeof(rot_params));
    rot_params.src_rect       = &rot_src;
    rot_params.dst_rect       = &rot_dst;
    rot_params.transform_flag = NVBUFSURF_TRANSFORM_FLIP;
    rot_params.transform_flip = NvBufSurfTransform_Rotate90;

    err = NvBufSurfTransform(crop_surf, rot_surf, &rot_params);
    if (err != NvBufSurfTransformError_Success)
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;

    // RESIZE
    NvBufSurfTransformRect resize_src = {0, 0, 150,  2048};
    NvBufSurfTransformRect resize_dst = {0, 0, 32,   128};

    params.transform_params.src_rect        = &resize_src;
    params.transform_params.dst_rect        = &resize_dst;
    params.transform_params.transform_flag  = NVBUFSURF_TRANSFORM_CROP_SRC
                                            | NVBUFSURF_TRANSFORM_CROP_DST
                                            | NVBUFSURF_TRANSFORM_FILTER;
    params.transform_params.transform_filter = NvBufSurfTransformInter_Bilinear;

    err = NvBufSurfTransform(rot_surf, out_surf, &params.transform_params);
    if (err != NvBufSurfTransformError_Success)
        return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;

    return NVDSPREPROCESS_SUCCESS;
}

CustomCtx *initLib(CustomInitParams initparams)
{
  auto ctx = std::make_unique<CustomCtx>();
  NvDsPreProcessStatus status;

  ctx->custom_mean_norm_params.pixel_normalization_factor =
      std::stof(initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_PIXEL_NORMALIZATION_FACTOR]);

  if (!initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_MEAN_FILE].empty()) {
    char abs_path[_PATH_MAX] = {0};
    if (!get_absolute_file_path (initparams.config_file_path,
          initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_MEAN_FILE].c_str(), abs_path)) {
      printf("Error: Could not parse mean image file path\n");
      return nullptr;
    }
    if (!ctx->custom_mean_norm_params.meanImageFilePath.empty()) {
      ctx->custom_mean_norm_params.meanImageFilePath.clear();
    }
    ctx->custom_mean_norm_params.meanImageFilePath.append(abs_path);
  }

  std::string offsets_str = initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_OFFSETS];

  if (!offsets_str.empty()) {
    std::string delimiter = ";";
    size_t pos = 0;
    std::string token;

    while ((pos = offsets_str.find(delimiter)) != std::string::npos) {
        token = offsets_str.substr(0, pos);
        ctx->custom_mean_norm_params.offsets.push_back(std::stof(token));
        offsets_str.erase(0, pos + delimiter.length());
    }
    ctx->custom_mean_norm_params.offsets.push_back(std::stof(offsets_str));

    printf("Using offsets : %f,%f,%f\n", ctx->custom_mean_norm_params.offsets[0],
          ctx->custom_mean_norm_params.offsets[1], ctx->custom_mean_norm_params.offsets[2]);
  }

  status = normalization_mean_subtraction_impl_initialize(&ctx->custom_mean_norm_params,
          &initparams.tensor_params, ctx->tensor_impl, initparams.unique_id);

  if (status != NVDSPREPROCESS_SUCCESS) {
    printf("normalization_mean_subtraction_impl_initialize failed\n");
    return nullptr;
  }

  ctx->initParams = initparams;

  // constantes de normalización 

  static const float INV_XMAX = 1.0f / 2005.2559796039604f;
  
  static const float MEANS[3] = {
      0.003930921760659862f,
      0.019108146307200424f,
      0.01762230914738107f
  };
  
  static const float STDS[3] = {
      0.01091675497061941f,
      0.051508108170714f,
      0.046621915109236536f
  };
  
  init_preprocess_constants(INV_XMAX, MEANS, STDS);

  // Preallocar superficies intermedias de transformación

  NvBufSurfaceCreateParams p;
  memset(&p, 0, sizeof(p));
  p.gpuId = 0;
  p.layout = NVBUF_LAYOUT_PITCH;
  p.memType = NVBUF_MEM_CUDA_UNIFIED;
  p.colorFormat = NVBUF_COLOR_FORMAT_NV12;  

  p.width  = 2048; p.height = 150;
  if (NvBufSurfaceCreate(&ctx->prealloc_crop_surf, 1, &p) != 0) {
      printf("Failed to preallocate crop surface\n");
      return nullptr;
  }

  p.width  = 150; p.height = 2048;
  if (NvBufSurfaceCreate(&ctx->prealloc_rot_surf, 1, &p) != 0) {
      NvBufSurfaceDestroy(ctx->prealloc_crop_surf);
      printf("Failed to preallocate rot surface\n");
      return nullptr;
  }

  g_ctx = ctx.get();

  return ctx.release();
}

void deInitLib(CustomCtx *ctx)
{
  g_ctx = nullptr;
  if (ctx->prealloc_crop_surf) NvBufSurfaceDestroy(ctx->prealloc_crop_surf);
  if (ctx->prealloc_rot_surf)  NvBufSurfaceDestroy(ctx->prealloc_rot_surf);
  delete ctx;
}
