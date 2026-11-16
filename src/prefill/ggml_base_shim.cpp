// src/prefill/ggml_base_shim.cpp - the three ggml-base functions the MMQ prompt kernels (ggml-cuda's mmq sources,
// compiled into strata_mmq) call, for a build without ggml's libraries (STRATA_GGML=OFF): the block geometry from
// Strata's own type table and the abort.  C linkage with ggml's names; an enum ggml_type argument is passed as an int.
#include "strata/qtypes.hpp"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {

[[noreturn]] void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fflush(stdout);
    std::fprintf(stderr, "%s:%d: ", file, line);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::abort();
}

int64_t ggml_blck_size(int type) { return strata::qt::blck_size(type); }

size_t ggml_type_size(int type) { return strata::qt::type_size(type); }

}  // extern "C"
