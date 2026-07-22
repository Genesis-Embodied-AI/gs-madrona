/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <dlfcn.h>
#include <iostream>

#include <madrona/crash.hpp>
#include <madrona/cuda_utils.hpp>

namespace madrona {

void CudaDynamicLoader::ensureLoaded() {
  static std::once_flag flag;

  std::call_once(flag, [] {
    cuda_handle_ = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
    if (!cuda_handle_) {
        fprintf(stderr, "Failed to load Nvidia CUDA driver library: %s\n", dlerror());
        std::abort();
    }
#define LOAD_SYM(handle, name)                                 \
    name = (decltype(name))dlsym(handle, #name);               \
    if (!name) {                                               \
        fprintf(stderr, "Failed to load symbol %s\n", #name);  \
        std::abort();                                          \
    }

#define LOAD_CUDA_SYM(name) LOAD_SYM(cuda_handle_, name)
    LOAD_CUDA_SYM(cuDeviceGet);
    LOAD_CUDA_SYM(cuDevicePrimaryCtxRetain);
    LOAD_CUDA_SYM(cuCtxSetCurrent);
    LOAD_CUDA_SYM(cuModuleLoadData);
    LOAD_CUDA_SYM(cuModuleGetFunction);
    LOAD_CUDA_SYM(cuLaunchKernel);
    LOAD_CUDA_SYM(cuMemcpyHtoD);
    LOAD_CUDA_SYM(cuMemAllocManaged);
    LOAD_CUDA_SYM(cuMemAdvise);
    LOAD_CUDA_SYM(cuMemFree);
    LOAD_CUDA_SYM(cuDeviceGetAttribute);
    LOAD_CUDA_SYM(cuCtxGetDevice);
    LOAD_CUDA_SYM(cuModuleGetGlobal);
    LOAD_CUDA_SYM(cuMemCreate);
    LOAD_CUDA_SYM(cuMemMap);
    LOAD_CUDA_SYM(cuMemRelease);
    LOAD_CUDA_SYM(cuMemSetAccess);
    LOAD_CUDA_SYM(cuMemAddressReserve);
    LOAD_CUDA_SYM(cuMemAlloc);
    LOAD_CUDA_SYM(cuMemGetAllocationGranularity);
    LOAD_CUDA_SYM(cuMemUnmap);
    LOAD_CUDA_SYM(cuMemAddressFree);
    LOAD_CUDA_SYM(cuGraphCreate);
    LOAD_CUDA_SYM(cuGraphAddKernelNode);
    LOAD_CUDA_SYM(cuGraphInstantiate);
    LOAD_CUDA_SYM(cuGraphDestroy);
    LOAD_CUDA_SYM(cuGraphAddEventRecordNode);
    LOAD_CUDA_SYM(cuGraphExecDestroy);
    LOAD_CUDA_SYM(cuEventCreate);
    LOAD_CUDA_SYM(cuEventSynchronize);
    LOAD_CUDA_SYM(cuEventElapsedTime);
    LOAD_CUDA_SYM(cuModuleUnload);
    LOAD_CUDA_SYM(cuGraphLaunch);
    LOAD_CUDA_SYM(cuMemcpy);
    LOAD_CUDA_SYM(cuGetErrorName);
    LOAD_CUDA_SYM(cuGetErrorString);
#undef LOAD_CUDA_SYM

    // nvrtc is a normal (dynamic) link-time dependency of this library
    // (DT_NEEDED libnvrtc.so.12, resolved via RPATH from the pinned cu12
    // package). Bind the entry points directly to those linked symbols rather
    // than dlopen'ing a host nvrtc: dlopen ignores DT_RUNPATH, so a plain
    // dlopen("libnvrtc.so") could pick up a mismatched host nvrtc (e.g. .so.13)
    // whose PTX output the cu12 nvJitLink cannot link. Binding to the linked
    // symbols keeps nvrtc and nvJitLink on the same cu12 version.
#define BIND_NVRTC_SYM(name) name = &(::name)
    BIND_NVRTC_SYM(nvrtcCreateProgram);
    BIND_NVRTC_SYM(nvrtcDestroyProgram);
    BIND_NVRTC_SYM(nvrtcCompileProgram);
    BIND_NVRTC_SYM(nvrtcGetPTX);
    BIND_NVRTC_SYM(nvrtcGetPTXSize);
    BIND_NVRTC_SYM(nvrtcGetProgramLog);
    BIND_NVRTC_SYM(nvrtcGetProgramLogSize);
    BIND_NVRTC_SYM(nvrtcAddNameExpression);
    BIND_NVRTC_SYM(nvrtcGetLoweredName);
    BIND_NVRTC_SYM(nvrtcGetLTOIRSize);
    BIND_NVRTC_SYM(nvrtcGetLTOIR);
    BIND_NVRTC_SYM(nvrtcGetCUBINSize);
    BIND_NVRTC_SYM(nvrtcGetCUBIN);
    BIND_NVRTC_SYM(nvrtcGetErrorString);
#undef BIND_NVRTC_SYM

#undef LOAD_SYM

    std::atexit([]() {
        dlclose(cuda_handle_);
        cuda_handle_ = nullptr;
    });
  });
}

namespace cu {

[[noreturn]] void cudaRuntimeError(
        cudaError_t err, const char *file,
        int line, const char *funcname) noexcept
{
    fatal(file, line, funcname, "%s", cudaGetErrorString(err));
}

[[noreturn]] void cuDrvError(
        CUresult err, const char *file,
        int line, const char *funcname) noexcept
{
    const char *name, *desc;
    CudaDynamicLoader::cuGetErrorName(err, &name);
    CudaDynamicLoader::cuGetErrorString(err, &desc);
    fatal(file, line, funcname, "%s: %s", name, desc);
}

}
}