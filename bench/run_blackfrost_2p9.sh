#!/bin/bash
# Blackfrost DeepSeek-V4.1-Flash EXL3 2.9bpw (our quant, exllamav3 convert on 78f1 + Mia-1017 graft for Engram/indexers/CSA compressors)
# on 2 Sparks TP2: head spark-b610 (local weights), worker spark-78f1 (local copy). Mia recipe image :2.9bpw, LANGUAGE_MODEL_ONLY=1.
# Same contract as the ds41-1017-exl3 rows: speed probe -> Spark-Bench v7.0 thinking OFF, uncapped, temp 1.0 top_p 0.95, seeds 2001-2002.
set -u
OUT=~/projects/lg-hard-pilot/rec; LOG=$OUT/QUEUE.log; NAME=blackfrost-exl3-2p9
HOST=10.0.0.120; PORT=8889; SSH=raulwesche@$HOST; RD=/home/raulwesche/dsv41-recipe
CTX=600000; TEMP=1.0; SAMPLING_JSON='{"top_p":0.95}'; MODEL_ID=Blackfrost-DSv4.1-Flash-EXL3-2.9bpw
LABEL="Blackfrost DSv4.1-Flash EXL3 2.9bpw (own exllamav3 quant + 1017 graft; Mia recipe, vLLM+EXL3, DSpark k5, 2 Sparks TP2)"
TOPO="2x DGX Spark TP2 (spark-b610 head + spark-78f1)"
NOTES="Blackfrost FP8 -> EXL3 2.9bpw (3.02 avg, head 6b, MTP 4b) via patched exllamav3 on spark-78f1; Engram L1/L14, ratio-1 indexers L20-36 and CSA compressors L2/8/14 grafted from LibertAIDAI 1017 EXL3 (identical upstream weights); 44 unused ape/bias_vl tensors stripped; Mia recipe image :2.9bpw unchanged, LANGUAGE_MODEL_ONLY=1"
source ~/projects/lg-hard-pilot/nemotron/lane_lib.sh
stop_server(){ ssh $SSH "cd $RD && ./start.sh stop" </dev/null >/dev/null 2>&1; }

M=$(curl -s -m 10 http://$HOST:$PORT/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ "$M" = "$MODEL_ID" ] || die "server not serving $MODEL_ID (got '$M')"
say "serving '$MODEL_ID'; speed probe"
python3 ~/projects/lg-hard-pilot/oct17/speed_probe.py http://$HOST:$PORT/v1 "$MODEL_ID" $OUT/$NAME-speed.json > $OUT/$NAME-speed.log 2>&1
say "speed probe done: $(tail -1 $OUT/$NAME-speed.log | cut -c1-200)"
say "full Spark-Bench v7.0 (thinking OFF, uncapped, 2 repeats)"
run_bench $CTX
say "lane done (server left up)"
