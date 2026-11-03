# Optional DeepSeek V4.1 Decode expert helper

This CUDA prototype uses a second GPU as an extra resident expert tier for Decode. The primary GPU keeps attention, Engram, the output head, and its existing Prefill implementation. Host copies of the helper's experts stay in the existing RAM tier; this does not reduce host memory requirements.

It is off by default. With the variable absent or `STRATA_GLM_EXPERT_HELPER=0`, no helper is constructed and the original CPU-lane computation is used. The default path's numerical parity and helper performance must be checked against the same upstream base and workload.

## Run

Example for a primary at visible GPU 0 and a helper at visible GPU 1:

```sh
CUDA_VISIBLE_DEVICES=0,1 \
STRATA_GLM_EXPERT_HELPER=1 \
STRATA_GLM_EXPERT_HELPER_GIB=13 \
STRATA_DS4_SPEC=0 \
STRATA_GLM_NO_SPEC=1 \
STRATA_GLM_KDA_GRAPH=0 \
./build/strata --serve --glm-pack /path/to/pack --layer-split 0 --spec 0
```

`STRATA_GLM_EXPERT_HELPER` is the helper's index in the visible GPU list. Zero disables the feature; the helper must be a separate visible GPU. `STRATA_GLM_EXPERT_HELPER_GIB` is its expert-pool budget in GiB (default 13, accepted range 0.25–128). The helper also keeps 512 MiB free for its small buffers and runtime allocations. A budget that cannot fit disables the helper instead of failing the model load or request.

Supported configurations are an unsplit DeepSeek V4.1 native pack, with a nonzero CPU-lane plan and no iGPU lane. Speculative decoding (the DSpark drafter, `--dspark`) uses it too: a verify window's helper-owned experts go to the helper as one call for all its columns while the CPU computes the rest (v1.3.94, #24). The feature is inactive with layer graphs. HIP keeps its existing path; AMD helper execution has not been implemented or tested in this prototype. Unsupported configurations print a diagnostic and retain the existing lane.

After Prefill, the helper selects RAM-resident experts not in primary VRAM, ranked by existing usage counts, with equal byte budgets per layer. It copies them once and retains selected slots between requests. The primary's experts have priority. Decode routes dispatch eligible remaining experts to the helper while the CPU computes the rest, then merge both contributions through the existing CPU answer. The usual residual processing, including the runtime refusal-direction edit when enabled, remains downstream of that merged answer.

`STRATA_GLM_EXPERT_HELPER_PROMOTE=N` (v1.3.96, #24; default 0 = off) lets the primary keep adapting while the helper is on. Normally a helper-owned expert is never promoted into primary VRAM, so with the helper on far fewer experts move into VRAM during Decode (about 0.3-0.5 promotions per token instead of 2.2-2.5 in #24) and the next prompt runs with a VRAM tier that has not followed the conversation (Prefill about 10% slower after a Decode). With N > 0, a helper-owned expert whose route count reaches N in a route is fetched into a free VRAM spare instead (at most one per route, only when a spare is left after the route's other promotions). Try 2 or 4; it trades some helper hits for a primary that keeps up. No effect when the helper is off or VRAM promotion is disabled.

`STRATA_GLM_EXPERT_HELPER_RESYNC=N` (v1.3.97, #24; default 0 = off) does the catching-up between requests instead: after a helper decode, once the request's `DONE` is out, up to N of the hottest RAM-held experts are promoted into VRAM (each layer's spares, refilled by demoting its coldest residents, while a RAM-held expert's aged count is at least 2 x the coldest resident's + 4). Decode keeps the helper's full gain; the next prompt starts with a VRAM tier that has followed the conversation. `PROMOTE` traded Decode for Prefill in #24 (PROMOTE=2: Prefill +7%, Decode -10%; helper on / PROMOTE=0 kept the shortest request time). The resync prints `expert helper resync: <n> promotions in <ms>` (one RTX 5090 at Gen4 x4: 50-210 promotions in 0.2-0.7 s). From v1.3.99 it stops as soon as the next request arrives (`(stopped: the next request)` in the log); from v1.3.102 it lands one copy at a time and checks before each, so the request waits for about two expert copies (~1 ms at Gen5 x8, 3.5-5 ms at Gen4 x4; the log's `longest gap between next-request checks` is that bound), so it only uses the idle time between requests: back-to-back requests pay ~nothing, and with seconds between turns (a person typing) it runs until every layer is in step. In #24 a budget of 96/192 was too small to bring the Prefill back (+0-1.4%, PROMOTE=2 moved ~4,400 experts per 2K-token answer) and its wait cost 33-106 ms a request; so try a large budget, e.g. 8192, with pauses between requests.

Allocation and placement happen after the prompt; their costs belong to the request's Decode/first-token latency. The helper has no peer-to-peer requirement: it transfers small activation and result vectors through host memory. Its pool can remain allocated while subsequent prompts run, but the Prefill kernels and routing never read its ownership table.

## Failure and request lifetime

`STRATA_GLM_EXPERT_HELPER_SAME_GPU=1` (debug) puts the helper on the primary GPU itself: no speed-up, it only checks the helper path on a one-GPU box, and `STRATA_GLM_EXPERT_HELPER_CHECK=1` (slow) recomputes every helper-served verify window on the CPU and prints the largest difference (the worst column so far, so it grows with the number of columns) and the RMS difference over all checked columns. Every 256th checked column is also computed exactly (dequantized weights, float activations), and the helper's and the CPU lane's distances to that reference are printed side by side: the helper is fine when its numbers match the CPU lane's (on one RTX 5090: helper vs CPU RMS 1e-4; both 0.022 RMS from the exact result). A missing/invalid helper device, a same-device selection, insufficient VRAM, allocation failure, or preparation/upload failure leaves the existing CPU lane in charge. Partial allocations and uploads are drained and released before Decode starts. An error in a helper launch, input transfer, or completion recomputes the entire affected CPU-lane route on the CPU; no partial helper answer is published. The helper is disabled for the remaining and subsequent requests. Its optional launch errors do not clear or replace the primary GPU's error reporting.

A request scope drains the primary/service routes and turns helper ownership off on normal completion, cancellation, or an early error. Reset does the same. Runtime-failed helper buffers are released only after those routes drain, so already-queued kernels can still safely read the ownership table. As with the existing engine, an unrecoverable driver/device failure affecting the primary cannot be made recoverable by this optional tier.

The `expert helper prepared`, `request_end`, and `destroy` diagnostics report pool bytes, resident experts, calls, upload bytes, timings, and fallback counts. Existing `cpu lane` statistics include helper work, so they are not CPU-only measurements when this feature is on.

## Validation and capacity comparisons

Use a baseline built from the same upstream commit and a patched build with the helper unset and explicitly zero. Compare full precision output/state or logits in addition to generated text. Exercise single-GPU visibility, invalid device, insufficient pool budget, setup/upload/runtime failure, cancellation, repeated requests, and normal exit. Compare helper-on quality separately: changing the partition changes floating-point summation order, so helper-on output is not promised bit-identical to CPU execution.

For a primary-memory comparison, the existing `STRATA_GLM_VRAM_GB=16` limits the primary GPU's memory budget without changing Prefill code. Benchmark helper off/on under both the native memory budget and that cap, using matched fresh processes, input token IDs, warm-up, sampling, and request lengths. Record actual peak memory, swaps, first-token latency, wall time, and helper setup costs. A memory-limited RTX PRO 4500 remains a PRO 4500: it approximates a 16 GiB capacity constraint, not the compute or memory-bandwidth performance of a second RTX 5070 Ti.

The Linux hardware regression runner accepts existing token IDs, avoiding tokenizer or model downloads:

```sh
c++ -std=c++17 -shared -fPIC -I"$CUDA_HOME/include" \
    tools/expert_helper_faults.cpp -ldl -o /tmp/expert-helper-faults.so
python tools/check_expert_helper.py \
    --baseline /path/to/upstream/strata --exe /path/to/patched/strata \
    --pack /path/to/pack --ids bench/expert-helper/prompt-16384.json \
    --output /path/to/new-test-results --context 32768 \
    --fault-library /tmp/expert-helper-faults.so
```

The ID file is a JSON array or an object with an `ids` array. Use a prompt that produces more than four tokens so cancellation is exercised. The default modes check upstream/off/zero hidden-state and token equality; helper execution; single-visible-GPU and oversized-pool fallback; allocation, upload, input, completion and kernel-launch faults; and cancellation during Prefill or Decode followed by two requests. Each mode loads a fresh process, sends three requests and requires normal `QUIT` exit. `--modes` selects a comma-separated subset. The fault library uses `LD_PRELOAD` only in its test child processes, never production engine flags; CUDA 13 was used for its hardware validation. It simulates recoverable API/launch errors, not an actual GPU reset or a fatal driver fault. Arrange exclusive GPU access and a suitable RAM budget before running it; the runner does not manage other services.

The included 16,384-ID fixture is the synthetic warehouse/inventory context and Python-generation request used for the matched measurements. It uses the DeepSeek V4.1 tokenizer from the benchmark pack. SHA256 of its comma-separated IDs (without a trailing newline): `73be3223a935b3eeef64f6e73263fa7128bbe88f8fdab13d522cf9b559ef2cf9`. For the performance protocol, send `GEN 2048 temperature=0 top_p=1 top_k=0 min_p=0 seed=42 spec=0 <IDs>` once as warm-up and then three times for measurement in each fresh process; the regression runner's three short requests serve a separate correctness purpose.

## Measured on v1.3.73

Base `30795a60c1642fc484f46110f0b62233c0b5d33e`, with the official Engram fix. Ryzen 9 9900X, 96 GB DDR5, RTX PRO 4500 Blackwell 32 GB + RTX 5070 Ti 16 GB; both links ran at PCIe 5.0 x8. CUDA 13.0, driver 595.99.02, Release sm_120a native-expert builds. The same DeepSeek V4.1 Flash 1017 ISTA IQ2_XXS / GSQ-RCO pack with the Huihui attention-output replacement was used throughout (as in #21). The runtime refusal-direction edit was not enabled.

Fresh process per mode, one 16,384-input / 2,048-output warm-up followed by three measured requests; medians below. Context 32,768, Prefill chunk 4,096, 80 decimal GB RAM cache, 12 CPU workers and the same fixed CPU plan `011233456` in all four modes. Greedy, seed 42; prompt reuse, speculative decoding and approximate skips off. Helper pool budget 13 GiB. The second card stays installed but unused in off modes. Placement costs stay in request Decode/TTFT and wall time. This is the custom workload above, not the standard `--bench` suite.

| Primary / helper | Prefill tok/s | Decode tok/s | TTFT s | Wall s | Sampled RSS GB |
|---|---:|---:|---:|---:|---:|
| PRO 4500 native / off | 1306.68 | 45.18 | 12.54 | 57.87 | 90.95 |
| PRO 4500 native / 5070 Ti | 1289.27 | 49.75 | 12.75 | 53.88 | 91.42 |
| PRO 4500 16 GiB cap / off | 939.82 | 28.75 | 17.43 | 88.66 | 90.95 |
| PRO 4500 16 GiB cap / 5070 Ti | 935.02 | 36.76 | 17.67 | 73.34 | 91.42 |

Native primary memory: Decode +10.1%, wall time -6.9%. 16 GiB primary cap: Decode +27.9%, wall time -17.3%.

The 16 GiB cap is a capacity comparison on a PRO 4500, not a measurement of two RTX 5070 Ti cards. Floating-point partitioning can change generated sequences; this is not fixed-token replay. Three repeats do not establish statistical significance or general model-quality equivalence. Short requests can spend more on helper placement than they save in Decode; the feature remains off by default.

Validation: upstream, helper-unset and helper-zero raw hidden-state dumps matched byte for byte across ten request cases. Helper-on text matched all ten cases (basic responses and Prefill boundaries), without promising bitwise equality. Eleven hardware configurations completed 82 checked requests, including missing/invalid GPU, oversized pool, partial allocation and preparation/input/completion faults; separate kernel-fault, Decode-cancel and Prefill-cancel modes completed nine more requests. Faulted/cancelled processes continued into subsequent requests and all exited normally on `QUIT`. These are recoverable API/launch-error simulations, not physical GPU resets. Existing repository suites passed: server 148 run / 3 skipped, tools 236 run / 27 skipped; CUDA build and HIP-name mapping check passed. AMD helper execution and Windows execution were not tested.

Patched engine SHA256: `0da58c900f9905d73071bad628ba05927c939eccdde37d969b1f4a43e90f967f`. Input comma-separated-ID SHA256: `73be3223a935b3eeef64f6e73263fa7128bbe88f8fdab13d522cf9b559ef2cf9`.
