#!/bin/bash

# actkv variant of startup.sh: runs the actkv build with the native RAM-KV tier and its LRFU
# recency/frequency history over the mean-key retrieval, so repeatedly-relevant history stays in the
# resident GPU window across turns instead of a single-shot score that drops it once the query moves
# on. The actkv build lives at /media/disk_0/ninfer-actkv (kept off the full nvme root); FFmpeg/curl
# shared libraries still come from the nvme build/_deps install.
#
# Raw traces write straight to stderr and can glue to the in-place status line; enable only while
# debugging: NINFER_REUSE_TRACE=1 NINFER_RAM_KV_TRACE=1 NINFER_VISION_SWAP_TRACE=1
#
# RAM-KV supports multiple active requests: the GPU window is ONE pool split evenly across
# --max-concurrency lanes, so each lane keeps window/max-concurrency = 40960 tokens resident here and
# archives the rest of its conversation in host RAM (bounded by --ram-kv-budget-bytes). Every lane
# needs at least prefill-chunk*(PP2 pipeline chunks) + 8192 tokens, so shrinking the window or raising
# --max-concurrency further must respect that floor.
#
# Maxima on this 2x16 GiB box (27B, int8 KV, TP2):
#   --max-context 262144 is the artifact's native ceiling (kNativeContext); going beyond it needs
#                  --rope yarn, which changes the rotary regime and the prompt template.
#   --ram-kv-window is GPU-memory bound, not context bound. At a 98304-token window the KV runtime
#                  measured 3.52 GiB with 3.00 GiB slack after weights, and one 64-token page costs
#                  ~2.19 MiB (Text + MTP), so 163840 tokens (~5.9 GiB of KV) leaves roughly 450 MiB
#                  for graphs and workspace. A 262144-token window would need ~9 GiB and cannot fit;
#                  if startup reports an allocation failure, drop this to 131072 or 147456.
#   --ram-kv-budget-bytes only has to clear the startup check (one archive x tp is ~17.5 GiB at a
#                  262144-token context). The host has 62 GiB, and --vision-kv-swap copies a rank's
#                  KV (~6 GiB at this window) to host RAM during image encode, so 36 GiB is the
#                  practical maximum. The budget is not a runtime cap: each lane maps its own
#                  full-logical-capacity archive and host RSS follows what conversations actually
#                  archive, so four 256k conversations can exceed it.
#
# actkv knobs (see docs/serving.md):
#   --ram-kv-window             resident GPU KV window, in tokens (multiple of 64)
#   --ram-kv-budget-bytes       host-RAM archive budget across all ranks (not process RSS)
#   --ram-kv-lrfu               rank resident pages by a decaying recency/frequency history
#   --ram-kv-lrfu-lambda        decay in [0,1], default 0.5
#   --ram-kv-lrfu-top-fraction  reinforced top fraction, in (0,1], default 0.9
#
# --kv-capacity is omitted: it defaults to the RAM window, which the engine requires.
# --kv-park-bytes is disabled because the engine rejects parking together with RAM-KV (RAM-KV keeps
# its own host archive).
# --pending-timeout-ms 0: prefill admission is serialized, so with the default 30 s a second long
# request can expire while the first is still prefilling; 0 waits instead.
# To capture one agent turn for diagnosis, append `--capture-turn profiles/turn.capture`: the first
# /v1/messages request carrying tools is written verbatim (request JSON + exact response bytes)
# once, then capture disarms until the next server start.
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /media/disk_0/ninfer-actkv/apps/ninfer-serve /media/disk_0/Huggingface/hfbuild/nvfp4_all/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --port 8080 --tp 2 --devices 0,1 --max-concurrency 4 \
  --max-context $((256*1024)) --kv-dtype int8 --prefill-chunk $((128*7)) \
  --ram-kv-window $((160*1024)) --ram-kv-budget-bytes $((36*1024*1024*1024)) \
  --ram-kv-lrfu --ram-kv-lrfu-lambda 0.8 --ram-kv-lrfu-top-fraction 0.9 \
  --spec mtp --draft-tokens 3 --lm-head-draft --default-max-tokens 65536 \
  --request-log-jsonl profiles/serve.jsonl --log-stats-interval-ms 1000 --status-line --status-line-interval-ms 500 --pending-timeout-ms 0 --preserve-thinking --top-k 40 --top-p 0.95 --min-p 0.0 --vision --vision-device gpu --vision-kv-swap --vision-max-tokens 0 --boundary-checkpoints --boundary-history 8 --shared-prefix \
  --shared-prefix-dir /media/disk_0/ninfer-kv-cache/prefixes --prefill-slice 256 --prefill-decode-rounds 4 --volta-fast-decode \
  --console-log-file profiles/serve.log
  # --kv-park-bytes is rejected together with --ram-kv-window; leave it off.
  # --kv-park-bytes $((42*1024*1024*1024))
