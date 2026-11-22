// src/prefill/moe_mmq_none.cu - include/strata/prefill/moe_mmq.hpp for a build without llama.cpp's MMQ sources
// (CMake STRATA_MMQ=OFF): no MMQ products; the sizes stay what the MMQ build plans its buffers with.  The prompt
// path's experts run on Strata's own GEMM (moe_own.hpp) for the types it covers and on the FP16 path for the rest.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qtypes.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace {
[[noreturn]] void none(const char* what) {
    std::fprintf(stderr, "prefill mmq: %s: this build has no MMQ kernels (STRATA_MMQ=OFF)\n", what);
    std::exit(1);
}
constexpr size_t kQ8Block = 4 * sizeof(float) + 128;   // (MMQ's activation block: 4 scales, 128 int8)
}  // namespace

bool built() { return false; }
bool optimized() { return true; }
bool supported(int) { return false; }
size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    const int64_t qk = strata::qt::blck_size(t);
    return qk > 0 ? (size_t) rows * (size_t) (cols / qk) * strata::qt::type_size(t) : 0;
}
size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) ((cols + 511) / 512 * 512) * kQ8Block / 128 + 128 * kQ8Block;
}
size_t q8_block_bytes() { return kQ8Block; }
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*, float*) { none("quantize"); }
bool glu_supported(int, int64_t) { return false; }
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) { none("product"); }
void Context::run_glu(const Product&, float, void*) { none("gate/up product"); }

}  // namespace strata::prefill::mmq
