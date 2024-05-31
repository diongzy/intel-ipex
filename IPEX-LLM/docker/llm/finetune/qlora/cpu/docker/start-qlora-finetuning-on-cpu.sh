#!/bin/bash
set -x
cd /ipex_llm
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
source /opt/intel/oneapi/setvars.sh
export CPU_CORES=$(nproc)
source ipex-llm-init -t
export WANDB_API_KEY='2bd9da9f8c9031d1a7bdddb45f3bdf84f3139346'
##For AMX and Jemalloc
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI
export LD_PRELOAD=export LD_PRELOAD=/usr/local/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:60000,muzzy_decay_ms:-1"

# Login to wandb
wandb login $WANDB_API_KEY
# Set the Hugging Face token
# HF_TOKEN="--auth-token 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"

# if [ -d "/home/qmed-intel/models/mistral_instruct" ];
# then
#   MODEL_PARAM="--repo-id-or-model-path "/home/qmed-intel/models/mistral_instruct" "  # otherwise, default to download from HF repo
# fi

if [ -d "/home/qmed-intel/models/llama_instruct" ];
then
  MODEL_PARAM="--repo-id-or-model-path "/home/qmed-intel/models/llama_instruct" "  # otherwise, default to download from HF repo
fi

if [ -d "/home/qmed-intel/Desktop/qmed_summarisation_data" ];
then
  DATA_PARAM="--dataset /home/qmed-intel/Desktop/qmed_summarisation_data" # otherwise, default to download from HF dataset
fi

if [ "$STANDALONE_DOCKER" = "TRUE" ]
then
  export CONTAINER_IP=$(hostname -i)
  source /opt/intel/oneapi/setvars.sh
  export CCL_WORKER_COUNT=$WORKER_COUNT_DOCKER
  export CCL_WORKER_AFFINITY=auto
  export MASTER_ADDR=$CONTAINER_IP
  mpirun \
     -n $CCL_WORKER_COUNT \
     -ppn $CCL_WORKER_COUNT \
     -genv OMP_NUM_THREADS=$((CPU_CORES / CCL_WORKER_COUNT)) \
     -genv KMP_AFFINITY="granularity=fine,none" \
     -genv KMP_BLOCKTIME=1 \
     -genv TF_ENABLE_ONEDNN_OPTS=1 \
     python qlora_finetuning_cpu.py $MODEL_PARAM $DATA_PARAM
else
  python qlora_finetuning_cpu.py $MODEL_PARAM $DATA_PARAM
fi

