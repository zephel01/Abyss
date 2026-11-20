"""tools/vision/hy4v_patch.py - Tencent Hy4-flash's vision tower in llama.cpp's mtmd, at the commit setup pins.

The projector type "hy_v4_flash" (the name the published mmproj files carry in clip.projector_type) with its own graph
(hy4v/hy4v.cpp): raster patches, the learned position grid resampled with the reference's scale factor (the weights
are filled here, in clip.cpp's input step), a pre-LayerNorm ViT without rope, then RMSNorm, a 2x2 conv, GELU, a 1x1
conv and one image_newline row after every merged row.  The output grid is (merged columns + 1) x merged rows, so a
text model with M-RoPE gets nx = columns + 1, ny = rows (the engine's GENI positions).  No begin / end markers: the
chat template writes them.  Runs after glm5next_patch.py (its anchors include that patch's lines).  Every edit is
anchored and idempotent, as in glm5next_patch.py.

    python hy4v_patch.py <llama.cpp dir>
"""
from __future__ import annotations

import pathlib
import shutil
import sys

MARK = "hy4v"


def edit(path: pathlib.Path, pairs: list[tuple[str, str]], count: int | None = 1):
    s = path.read_text(encoding="utf-8")
    for old, new in pairs:
        if new in s:
            continue                                   # already applied
        n = s.count(old)
        if n == 0 or (count is not None and n != count):
            raise SystemExit(f"hy4v patch: {path}: anchor found {n} times:\n{old}")
        s = s.replace(old, new)
    path.write_text(s, encoding="utf-8", newline="\n")


# the position grid's resampling weights (torch.nn.functional.interpolate, bilinear, align_corners=False, with an
# explicit scale_factor (n + 0.1) / 128: source = (o + 0.5) / scale - 0.5, clamped at 0), [n][128] row-major
WEIGHTS = """
static std::vector<float> hy4v_axis_weights(int n, int edge) {
    std::vector<float> w((size_t) n * edge, 0.0f);
    const double scale = ((double) n + 0.1) / (double) edge;
    for (int o = 0; o < n; ++o) {
        double src = ((double) o + 0.5) / scale - 0.5;
        if (src < 0.0) src = 0.0;
        int i0 = (int) src;
        if (i0 > edge - 1) i0 = edge - 1;
        const int i1 = i0 + 1 < edge ? i0 + 1 : edge - 1;
        const double l1 = src - (double) i0, l0 = 1.0 - l1;
        w[(size_t) o * edge + i0] += (float) l0;
        w[(size_t) o * edge + i1] += (float) l1;
    }
    return w;
}

"""


def main():
    root = pathlib.Path(sys.argv[1])
    m = root / "tools" / "mtmd"
    here = pathlib.Path(__file__).resolve().parent

    edit(m / "clip-impl.h", [
        ("    PROJECTOR_TYPE_GLM5NEXT,\n", "    PROJECTOR_TYPE_GLM5NEXT,\n    PROJECTOR_TYPE_HY4V,\n"),
        ('    { PROJECTOR_TYPE_GLM5NEXT,          "glm5next"},\n',
         '    { PROJECTOR_TYPE_GLM5NEXT,          "glm5next"},\n    { PROJECTOR_TYPE_HY4V,              "hy_v4_flash"},\n'),
    ])
    edit(m / "models" / "models.h", [
        ("struct clip_graph_glm5next : clip_graph {\n"
         "    clip_graph_glm5next(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}\n"
         "    ggml_cgraph * build() override;\n"
         "};\n",
         "struct clip_graph_glm5next : clip_graph {\n"
         "    clip_graph_glm5next(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}\n"
         "    ggml_cgraph * build() override;\n"
         "};\n\n"
         "struct clip_graph_hy4v : clip_graph {\n"
         "    clip_graph_hy4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}\n"
         "    ggml_cgraph * build() override;\n"
         "};\n"),
    ])
    edit(m / "CMakeLists.txt", [
        ("            models/glm5next.cpp\n", "            models/glm5next.cpp\n            models/hy4v.cpp\n"),
    ])
    shutil.copy2(here / "hy4v" / "hy4v.cpp", m / "models" / "hy4v.cpp")

    clip = m / "clip.cpp"
    edit(clip, [
        # the graph
        ("        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "            {\n"
         "                builder = std::make_unique<clip_graph_glm5next>(ctx, img);\n"
         "            } break;\n",
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "            {\n"
         "                builder = std::make_unique<clip_graph_glm5next>(ctx, img);\n"
         "            } break;\n"
         "        case PROJECTOR_TYPE_HY4V:\n"
         "            {\n"
         "                builder = std::make_unique<clip_graph_hy4v>(ctx, img);\n"
         "            } break;\n"),
        # hyperparameters: the processor's pixel limits (smart_resize with factor 32), bicubic resize
        ("                case PROJECTOR_TYPE_GLM5NEXT:\n"
         "                    {\n"
         "                        hparams.rope_theta = 10000.0f;\n",
         "                case PROJECTOR_TYPE_HY4V:\n"
         "                    {\n"
         "                        hparams.n_merge = 2;\n"
         "                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);\n"
         "                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;\n"
         "                        hparams.image_min_pixels = 4096;\n"
         "                        hparams.image_max_pixels = 2048 * 2048;\n"
         "                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels, false);\n"
         "                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels, false);\n"
         "                        hparams.set_warmup_n_tokens(32*32); // avoid OOM on warmup\n"
         "                    } break;\n"
         "                case PROJECTOR_TYPE_GLM5NEXT:\n"
         "                    {\n"
         "                        hparams.rope_theta = 10000.0f;\n"),
        # the projector tensors
        ("            case PROJECTOR_TYPE_GLM4V:\n"
         "            case PROJECTOR_TYPE_GLM5NEXT:\n"
         "                {\n"
         "                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, \"weight\"));\n",
         "            case PROJECTOR_TYPE_HY4V:\n"
         "                {\n"
         "                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, \"weight\"));\n"
         "                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, \"bias\"));\n"
         "                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, \"weight\"));\n"
         "                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, \"bias\"));\n"
         "                    model.mm_pre_norm_w = get_tensor(string_format(TN_MM_PRE_NORM, \"weight\"));\n"
         "                    model.image_newline = get_tensor(TN_IMAGE_NEWLINE);\n"
         "                } break;\n"
         "            case PROJECTOR_TYPE_GLM4V:\n"
         "            case PROJECTOR_TYPE_GLM5NEXT:\n"
         "                {\n"
         "                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, \"weight\"));\n"),
        # embedding size (mm.2.weight is [1, 1, 4608, 3072])
        ("        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "            return ctx->model.mm_ffn_down_w->ne[1];\n",
         "        case PROJECTOR_TYPE_HY4V:\n"
         "            return ctx->model.mm_2_w->ne[3];\n"
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "            return ctx->model.mm_ffn_down_w->ne[1];\n"),
        # the output grid: one newline column after the merged columns
        ("int clip_n_output_tokens_x(const clip_ctx * ctx, const clip_image_f32 * img) {\n"
         "    const auto & params = ctx->model.hparams;\n"
         "    const int n_total = clip_n_output_tokens(ctx, img);\n"
         "    const auto & proj = ctx->proj_type();\n"
         "    switch (proj) {\n",
         "int clip_n_output_tokens_x(const clip_ctx * ctx, const clip_image_f32 * img) {\n"
         "    const auto & params = ctx->model.hparams;\n"
         "    const int n_total = clip_n_output_tokens(ctx, img);\n"
         "    const auto & proj = ctx->proj_type();\n"
         "    switch (proj) {\n"
         "        case PROJECTOR_TYPE_HY4V:\n"
         "            return (img->nx() / params.patch_size) / params.n_merge + 1;\n"),
        ("int clip_n_output_tokens_y(const clip_ctx * ctx, const clip_image_f32 * img) {\n"
         "    const auto & params = ctx->model.hparams;\n"
         "    const auto & proj = ctx->proj_type();\n"
         "    switch (proj) {\n",
         "int clip_n_output_tokens_y(const clip_ctx * ctx, const clip_image_f32 * img) {\n"
         "    const auto & params = ctx->model.hparams;\n"
         "    const auto & proj = ctx->proj_type();\n"
         "    switch (proj) {\n"
         "        case PROJECTOR_TYPE_HY4V:\n"
         "            return (img->ny() / params.patch_size) / params.n_merge;\n"),
        ("        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "        case PROJECTOR_TYPE_YOUTUVL:\n"
         "        case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "            {\n"
         "                // dynamic size (2 conv, so double patch size)\n",
         "        case PROJECTOR_TYPE_HY4V:\n"
         "            {\n"
         "                const int m = params.n_merge;\n"
         "                n_patches = (img->nx() / (params.patch_size * m) + 1) * (img->ny() / (params.patch_size * m));\n"
         "            } break;\n"
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "        case PROJECTOR_TYPE_YOUTUVL:\n"
         "        case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "            {\n"
         "                // dynamic size (2 conv, so double patch size)\n"),
        # the inputs: the position grid's resampling weights
        ("        case PROJECTOR_TYPE_QWEN2VL:\n"
         "        case PROJECTOR_TYPE_QWEN3VL:\n"
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "            {\n"
         "                const int merge_ratio = hparams.n_merge;\n",
         "        case PROJECTOR_TYPE_HY4V:\n"
         "            {\n"
         "                set_input_f32(\"hy4v_wx\", hy4v_axis_weights(image_size_width / patch_size, 128));\n"
         "                set_input_f32(\"hy4v_wy\", hy4v_axis_weights(image_size_height / patch_size, 128));\n"
         "            } break;\n"
         "        case PROJECTOR_TYPE_QWEN2VL:\n"
         "        case PROJECTOR_TYPE_QWEN3VL:\n"
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "            {\n"
         "                const int merge_ratio = hparams.n_merge;\n"),
        ("bool clip_image_batch_encode(clip_ctx * ctx, int n_threads, const clip_image_f32_batch * imgs_c_ptr, "
         "std::vector<float> & out_batch_embd) {\n",
         WEIGHTS.lstrip("\n") +
         "bool clip_image_batch_encode(clip_ctx * ctx, int n_threads, const clip_image_f32_batch * imgs_c_ptr, "
         "std::vector<float> & out_batch_embd) {\n"),
    ])
    edit(m / "mtmd.cpp", [
        ("            case PROJECTOR_TYPE_GLM4V:\n"
         "            case PROJECTOR_TYPE_GLM5NEXT:\n"
         "                {\n"
         "                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>\n",
         "            case PROJECTOR_TYPE_HY4V:\n"
         "                {\n"
         "                    // the chat template writes <｜hy_image▁start｜> / <｜hy_image▁end｜> around the pads itself\n"
         "                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);\n"
         "                } break;\n"
         "            case PROJECTOR_TYPE_GLM4V:\n"
         "            case PROJECTOR_TYPE_GLM5NEXT:\n"
         "                {\n"
         "                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>\n"),
    ])
    print(f"llama.cpp at {root}: {MARK} vision added")


if __name__ == "__main__":
    main()
