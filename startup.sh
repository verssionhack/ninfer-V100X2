#!/bin/bash

# Raw traces write straight to stderr and can glue to the in-place status line; enable only while
# debugging: NINFER_REUSE_TRACE=1 NINFER_VISION_SWAP_TRACE=1
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /media/disk_0/Huggingface/hfbuild/nvfp4_all/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --port 8080 --tp 2 --devices 0,1 --max-concurrency 2 \
  --max-context $((256*1024)) --kv-dtype int8 --kv-capacity $((256*1024)) --prefill-chunk $((128*7)) \
  --spec mtp --draft-tokens 3 --lm-head-draft --default-max-tokens 65536 \
  --request-log-jsonl profiles/serve.jsonl --log-stats-interval-ms 1000 --status-line --status-line-interval-ms 500 --pending-timeout-ms 0 --preserve-thinking --top-k 40 --top-p 0.95 --min-p 0.0 --kv-park-bytes $((42*1024*1024*1024)) --vision --vision-device gpu --vision-kv-swap --vision-max-tokens 0 --boundary-checkpoints --boundary-history 8 --shared-prefix \
  --shared-prefix-dir /media/disk_0/ninfer-kv-cache/prefixes --prefill-slice 256 --prefill-decode-rounds 4 --volta-fast-decode \
  --console-log-file profiles/serve.log
  # Spilled KV is retained by cost/value by default once a spill dir is set; tune or opt out:
  #--kv-park-spill-dir /media/disk_0/ninfer-kv-cache --kv-park-spill-bytes $((128*1024*1024*1024)) --kv-park-io-bytes $((2*1024*1024*1024)) --kv-park-prefetch 2 --kv-park-max-age-sec 0 \
  #--kv-park-restore-min-saving-ms 750 --kv-park-value-half-life-sec 1800 --kv-park-restore-overhead-ms 250 --kv-park-value-sweep-sec 60 \
  #--kv-park-prefill-tok-s 0 --kv-park-read-bytes-per-sec 0   # 0 = measure at runtime; --kv-park-lru reverts to pure LRU
