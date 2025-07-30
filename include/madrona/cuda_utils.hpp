/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once
#include <cstddef>
#include <dlfcn.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <madrona/macros.hpp>

namespace madrona {

class CudaDynamicLoader {
public:
    static void ensureLoaded();

    // CUDA Driver API function pointers
    inline static CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    inline static CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    inline static CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    inline static CUresult (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
    inline static CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    inline static CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) = nullptr;
    inline static CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    inline static CUresult (*cuMemAllocManaged)(CUdeviceptr*, size_t, unsigned int) = nullptr;
    inline static CUresult (*cuMemAdvise)(CUdeviceptr, size_t, unsigned int, int) = nullptr;
    inline static CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    inline static CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    inline static CUresult (*cuCtxGetDevice)(CUdevice*) = nullptr;
    inline static CUresult (*cuModuleGetGlobal)(CUdeviceptr*, size_t*, CUmodule, const char*) = nullptr;
    inline static CUresult (*cuMemCreate)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
    inline static CUresult (*cuMemMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    inline static CUresult (*cuMemRelease)(CUmemGenericAllocationHandle) = nullptr;
    inline static CUresult (*cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    inline static CUresult (*cuMemAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    inline static CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
    inline static CUresult (*cuMemGetAllocationGranularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    inline static CUresult (*cuMemUnmap)(CUdeviceptr, size_t) = nullptr;
    inline static CUresult (*cuMemAddressFree)(CUdeviceptr, size_t) = nullptr;
    inline static CUresult (*cuGraphCreate)(CUgraph*, unsigned int) = nullptr;
    inline static CUresult (*cuGraphAddKernelNode)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_KERNEL_NODE_PARAMS*) = nullptr;
    inline static CUresult (*cuGraphInstantiate)(CUgraphExec*, CUgraph, unsigned long long) = nullptr;
    inline static CUresult (*cuGraphDestroy)(CUgraph) = nullptr;
    inline static CUresult (*cuGraphAddEventRecordNode)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUevent) = nullptr;
    inline static CUresult (*cuGraphExecDestroy)(CUgraphExec) = nullptr;
    inline static CUresult (*cuEventCreate)(CUevent*, unsigned int) = nullptr;
    inline static CUresult (*cuEventSynchronize)(CUevent) = nullptr;
    inline static CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent) = nullptr;
    inline static CUresult (*cuModuleUnload)(CUmodule) = nullptr;
    inline static CUresult (*cuGraphLaunch)(CUgraphExec, CUstream) = nullptr;
    inline static CUresult (*cuMemcpy)(CUdeviceptr, CUdeviceptr, size_t) = nullptr;
    inline static CUresult (*cuGetErrorName)(CUresult, const char **) = nullptr;
    inline static CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;

    // NVRTC API function pointers
    inline static nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*) = nullptr;
    inline static nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
    inline static nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
    inline static nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
    inline static nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t*) = nullptr;
    inline static nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
    inline static nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
    inline static nvrtcResult (*nvrtcAddNameExpression)(nvrtcProgram, const char*) = nullptr;
    inline static nvrtcResult (*nvrtcGetLoweredName)(nvrtcProgram, const char*, const char**) = nullptr;
    inline static nvrtcResult (*nvrtcGetLTOIRSize)(nvrtcProgram, size_t*) = nullptr;
    inline static nvrtcResult (*nvrtcGetLTOIR)(nvrtcProgram, char*) = nullptr;
    inline static nvrtcResult (*nvrtcGetCUBINSize)(nvrtcProgram, size_t*) = nullptr;
    inline static nvrtcResult (*nvrtcGetCUBIN)(nvrtcProgram, char*) = nullptr;
    inline static nvrtcResult (*nvrtcGetErrorString)(nvrtcResult) = nullptr;

private:
    static inline void* cuda_handle_;
    static inline void* nvrtc_handle_;
};

namespace cu {

inline void *allocGPU(size_t num_bytes);

inline void deallocGPU(void *ptr);

inline void *allocStaging(size_t num_bytes);

inline void *allocReadback(size_t num_bytes);

inline void deallocCPU(void *ptr);

inline void cpyCPUToGPU(cudaStream_t strm, void *gpu, void *cpu, size_t num_bytes);

inline void cpyGPUToCPU(cudaStream_t strm, void *cpu, void *gpu, size_t num_bytes);

inline cudaStream_t makeStream();

[[noreturn]] void cudaRuntimeError(
        cudaError_t err, const char *file,
        int line, const char *funcname) noexcept;
[[noreturn]] void cuDrvError(
        CUresult err, const char *file,
        int line, const char *funcname) noexcept;

inline void checkCuda(cudaError_t res, const char *file,
                      int line, const char *funcname) noexcept;
inline void checkCuDrv(CUresult res, const char *file,
                       int line, const char *funcname) noexcept;

}
}

#define ERR_CUDA(err) ::madrona::cu::cudaError((err), __FILE__, __LINE__,\
                                               MADRONA_COMPILER_FUNCTION_NAME)
#define ERR_CU(err) ::madrona::cu::cuDrvError((err), __FILE__, __LINE__,\
                                              MADRONA_COMPILER_FUNCTION_NAME)

#define REQ_CUDA(expr) ::madrona::cu::checkCuda((expr), __FILE__, __LINE__,\
                                                MADRONA_COMPILER_FUNCTION_NAME)
#define REQ_CU(expr) ::madrona::cu::checkCuDrv((expr), __FILE__, __LINE__,\
                                               MADRONA_COMPILER_FUNCTION_NAME)

#include "cuda_utils.inl"
