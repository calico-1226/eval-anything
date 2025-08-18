#!/bin/bash
export API_KEY='sk-1SS6rRTWAOiJHrKyaHmemnZH3Me5Ev53WRn5NlZBPQ37AOyV'
export API_BASE='https://api3.xhub.chat/v1/chat/completions'

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set the project root directory
PROJECT_ROOT="."

MODEL_PATH=/share/project/juntao/Projects/oss-red-team/models/openai/gpt-oss-20b
MODEL_NAME=gpt-oss-20b
PORT=8000
HOST=0.0.0.0
PP_SIZE=1
TP_SIZE=4
GPU_UTILIZATION=0.95
MAX_SEQ_LEN=32768
ENABLE_PREFIX_CACHING=false
DTYPE=bfloat16
DISABLE_LOG_STATS=true
DISABLE_LOG_REQUESTS=true
DISABLE_FASTAPI_DOCS=true


# benchmark
BENCHMARK=TruthfulQA
DATA_PATH=flag_safety/benchmarks/TruthfulQA/
SPLIT=train

# sampling
TEMPERATURE=0.0
TOP_P=0.1
REPETITION_PENALTY=1.0

REASONING=false

# Set other default parameters
RESULTS_DIR="./results"
CACHE_DIR="./cache/${MODEL_NAME}"
NUM_WORKERS=40



python -m flag_safety \
  --model-path ${MODEL_PATH} \
  --model-name ${MODEL_NAME} \
  --port ${PORT} \
  --host ${HOST} \
  --pipeline-parallel-size ${PP_SIZE} \
  --tensor-parallel-size ${TP_SIZE} \
  --gpu-memory-utilization ${GPU_UTILIZATION} \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  --max-seq-len ${MAX_SEQ_LEN} \
  --dtype ${DTYPE} \
  --benchmark ${BENCHMARK} \
  --split ${SPLIT} \
  --temperature ${TEMPERATURE} \
  --top-p ${TOP_P} \
  --repetition-penalty ${REPETITION_PENALTY} \
  --results-dir ${RESULTS_DIR} \
  --cache-dir ${CACHE_DIR} \
  --num-workers ${NUM_WORKERS} \
  --data-path ${DATA_PATH} \
  $([ "$ENABLE_PREFIX_CACHING" = "true" ] && echo "--enable-prefix-caching" || echo "") \
  $([ "$DISABLE_LOG_STATS" = "true" ] && echo "--disable-log-stats" || echo "") \
  $([ "$DISABLE_LOG_REQUESTS" = "true" ] && echo "--disable-log-requests" || echo "") \
  $([ "$DISABLE_FASTAPI_DOCS" = "true" ] && echo "--disable-fastapi-docs" || echo "") \
  $([ "$REASONING" = "true" ] && echo "--reasoning" || echo "")
