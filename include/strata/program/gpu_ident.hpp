#pragma once
// strata/program/gpu_ident.hpp - which card is which: `abyss --list-gpus` (the launchers turn a config's GPU UUID into
// today's device number with it) and the start line naming each device a run uses.  Device numbers change when a
// card is attached or removed (a USB4 card at PCI bus 03 numbered before an x16 card at c5); the UUID does not.
//
// The listing is the runtime's own (cudaGetDeviceProperties' uuid / cudaDeviceGetPCIBusId; HIP: hipDeviceGetUuid /
// hipDeviceGetPCIBusId) in the order the runtime numbers the devices - CUDA_DEVICE_ORDER=PCI_BUS_ID, which the
// launchers also set for the run, so the number printed is the one CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES take.
#include <cuda_runtime.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::program {

// the card's UUID as nvidia-smi -L prints it (GPU-88044da7-d824-7056-3fbf-40d39acad923); AMD: as rocminfo prints it
// (GPU- and the 16 hex digits of the unique id rocm-smi --showuniqueid prints, which hipDeviceGetUuid hands back as
// text).  "" when the card reports none (all zero - unchecked on AMD under Windows).
inline std::string gpu_uuid(int dev) {
    unsigned char b[16] = {};
#if defined(STRATA_USE_HIP)
    hipDevice_t d{};
    hipUUID u{};
    if (hipDeviceGet(&d, dev) != hipSuccess || hipDeviceGetUuid(&u, d) != hipSuccess) {
        (void) hipGetLastError();
        return "";
    }
    std::memcpy(b, u.bytes, 16);
    bool text = true;
    for (unsigned char c : b) text = text && std::isxdigit(c);
    if (text) return "GPU-" + std::string(reinterpret_cast<const char*>(b), 16);
#else
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return "";
    }
    std::memcpy(b, p.uuid.bytes, 16);
#endif
    bool zero = true;
    for (unsigned char c : b) zero = zero && c == 0;
    if (zero) return "";
    char s[64];
    std::snprintf(s, sizeof s, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2],
                  b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return s;
}

inline std::string gpu_pci(int dev) {
    char s[64] = {};
    if (cudaDeviceGetPCIBusId(s, (int) sizeof s, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return "?";
    }
    for (char* c = s; *c; ++c) *c = (char) std::tolower((unsigned char) *c);
    return s;
}

inline std::string gpu_name(int dev) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return "?";
    }
    return p.name;
}

// `abyss --list-gpus`: one row a device, tab-separated: index, UUID ("-" when none), PCI bus id, name
inline int list_gpus() {
#if defined(_WIN32)
    _putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
#else
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
#endif
#if defined(STRATA_USE_HIP)
    const char* rt = "HIP";
#else
    const char* rt = "CUDA";
#endif
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    std::printf("# abyss --list-gpus: %s devices as the run numbers them (index, uuid, pci, name)\n", rt);
    if (e != cudaSuccess) {
        std::printf("# no %s device (%s)\n", rt, cudaGetErrorString(e));
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        const std::string u = gpu_uuid(i);
        std::printf("%d\t%s\t%s\t%s\n", i, u.empty() ? "-" : u.c_str(), gpu_pci(i).c_str(), gpu_name(i).c_str());
    }
    return 0;
}

// the start line: "CUDA0 = NVIDIA GeForce RTX 5090 (UUID GPU-88044da7, PCI 0000:c5:00.0)" - the number is the
// run's own (within CUDA_VISIBLE_DEVICES)
inline void print_gpu_line(int dev) {
#if defined(STRATA_USE_HIP)
    const char* rt = "HIP";
#else
    const char* rt = "CUDA";
#endif
    std::string u = gpu_uuid(dev);
    if (u.size() > 12) u = u.substr(0, 12);
    std::fprintf(stderr, "%s%d = %s (UUID %s, PCI %s)\n", rt, dev, gpu_name(dev).c_str(), u.empty() ? "none" : u.c_str(),
                 gpu_pci(dev).c_str());
}

}  // namespace strata::program
