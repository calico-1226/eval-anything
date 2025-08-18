#!/bin/bash

export VLLM_USE_V1=0 # Set environment variable to disable V1 engine, so that disable_frontend_multiprocessing parameter can take effect
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set the project root directory
PROJECT_ROOT="."

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
MODEL_NAME=qwen2_5_vl_7b
PORT=8010
HOST=localhost
TP_SIZE=4
GPU_UTILIZATION=0.9
LIMIT_MM_PER_PROMPT="image=10,video=10"
ENABLE_PREFIX_CACHING=true
DTYPE=bfloat16
DISABLE_LOG_STATS=true
DISABLE_LOG_REQUESTS=true
DISABLE_FASTAPI_DOCS=true


# benchmark
BENCHMARK=mmmu
DATA_PATH=MMMU/MMMU
SPLIT=test

# sampling
TEMPERATURE=0.0
TOP_P=0.1
REPETITION_PENALTY=1.0


# Set other default parameters
RESULTS_DIR="./results"
CACHE_DIR="./cache"
NUM_WORKERS=50

python -m flag_safety \
  --model-path ${MODEL_PATH} \
  --model-name ${MODEL_NAME} \
  --port ${PORT} \
  --host ${HOST} \
  --tensor-parallel-size ${TP_SIZE} \
  --gpu-memory-utilization ${GPU_UTILIZATION} \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  --dtype ${DTYPE} \
  --benchmark ${BENCHMARK} \
  --data-path ${DATA_PATH} \
  --split ${SPLIT} \
  --temperature ${TEMPERATURE} \
  --top-p ${TOP_P} \
  --repetition-penalty ${REPETITION_PENALTY} \
  --results-dir ${RESULTS_DIR} \
  --cache-dir ${CACHE_DIR} \
  --num-workers ${NUM_WORKERS} \
  ${ENABLE_PREFIX_CACHING:+--enable-prefix-caching} \
  ${DISABLE_LOG_STATS:+--disable-log-stats} \
  ${DISABLE_LOG_REQUESTS:+--disable-log-requests} \
  ${DISABLE_FASTAPI_DOCS:+--disable-fastapi-docs}
