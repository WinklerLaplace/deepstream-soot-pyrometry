#pragma once

#include <cuda_runtime.h>
#include <nvbufsurface.h>
#include <nvdsmeta.h>

extern "C" {
    void ThermalRender_Init();

    void ThermalRender_Shutdown();

    void ThermalRender_RenderFrame(
        NvBufSurface*  surf,
        guint          batch_id,
        void*          ts_host_ptr,    
        void*          chw_device_ptr, 
        NvDsFrameMeta* frame_meta,
        cudaStream_t   stream);
}
