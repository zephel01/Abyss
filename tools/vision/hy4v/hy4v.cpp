// Tencent Hy4-flash's vision tower (transformers' HYV4FlashVisionModel), added to llama.cpp's mtmd by
// tools/vision/hy4v_patch.py as the projector "hy_v4_flash" (the mmproj's own clip.projector_type).
//   16x16 patches (one conv2d with bias) in raster order + the learned 128 x 128 position grid resampled bilinearly
//   to the image's grid (with the reference's (n + 0.1) / 128 scale factor: the weights come in as the inputs
//   hy4v_wx / hy4v_wy, filled by clip.cpp) -> 27 pre-LayerNorm blocks (full bidirectional attention, no rope, GELU
//   (erf) MLP) -> RMSNorm -> 2x2 conv to 4608 -> GELU -> 1x1 conv to 3072 -> one image_newline row after every merged
//   row.  Out: (merged columns + 1) x merged rows embeddings, raster order.
#include "models.h"

ggml_cgraph * clip_graph_hy4v::build() {
    GGML_ASSERT(model.patch_bias != nullptr && model.position_embeddings != nullptr);
    GGML_ASSERT(model.mm_0_w && model.mm_0_b && model.mm_2_w && model.mm_2_b && model.mm_pre_norm_w && model.image_newline);
    const int px = n_patches_x, py = n_patches_y;
    GGML_ASSERT(px % 2 == 0 && py % 2 == 0);

    // the patches: [px, py, C] -> [C, px * py] (patch y * px + x)
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    inp = ggml_add(ctx0, inp, model.patch_bias);

    // the position grid [C, 128 x, 128 y] resampled: first along x (wx [128, px]), then along y (wy [128, py])
    const int edge = 128;
    GGML_ASSERT(model.position_embeddings->ne[1] == edge * edge);
    ggml_tensor * wx = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, edge, px);
    ggml_set_name(wx, "hy4v_wx");
    ggml_set_input(wx);
    ggml_tensor * wy = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, edge, py);
    ggml_set_name(wy, "hy4v_wy");
    ggml_set_input(wy);
    ggml_tensor * pe = ggml_reshape_3d(ctx0, model.position_embeddings, n_embd, edge, edge);   // [C, x, y]
    pe = ggml_cont(ctx0, ggml_permute(ctx0, pe, 1, 0, 2, 3));                                  // [x, C, y]
    pe = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, pe, edge, n_embd * edge), wx);              // [C * y, px]
    pe = ggml_reshape_3d(ctx0, pe, n_embd, edge, px);                                         // [C, y, px]
    pe = ggml_cont(ctx0, ggml_permute(ctx0, pe, 1, 0, 2, 3));                                  // [y, C, px]
    pe = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, pe, edge, n_embd * px), wy);                // [C * px, py]
    pe = ggml_reshape_2d(ctx0, pe, n_embd, px * py);
    cb(pe, "pos_embed_resampled", -1);

    ggml_tensor * cur = build_vit(inp, n_patches, NORM_TYPE_NORMAL, FFN_GELU_ERF, pe, nullptr);
    cb(cur, "vit_out", -1);

    // the merger
    cur = build_norm(cur, model.mm_pre_norm_w, nullptr, NORM_TYPE_RMS, 1e-5f, -1);
    cur = ggml_reshape_3d(ctx0, cur, n_embd, px, py);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));                               // [px, py, C]
    cur = ggml_conv_2d(ctx0, model.mm_0_w, cur, 2, 2, 0, 0, 1, 1);                            // [px/2, py/2, 4C]
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.mm_0_b, 1, 1, model.mm_0_b->ne[0]));
    cur = ggml_gelu_erf(ctx0, cur);
    const int64_t mid = cur->ne[2], out = model.mm_2_w->ne[3];
    cur = ggml_reshape_2d(ctx0, cur, (px / 2) * (py / 2), mid);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));                                         // [4C, px/2 * py/2]
    cur = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, model.mm_2_w, mid, out), cur);            // [out, cells]
    cur = ggml_add(ctx0, cur, model.mm_2_b);
    cur = ggml_reshape_3d(ctx0, cur, out, px / 2, py / 2);
    ggml_tensor * nl = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, model.image_newline, out, 1, 1),
                                   ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, out, 1, py / 2));
    cur = ggml_concat(ctx0, cur, nl, 1);                                                      // [out, px/2 + 1, py/2]
    cur = ggml_reshape_2d(ctx0, cur, out, (px / 2 + 1) * (py / 2));
    cb(cur, "hy4v_out", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
