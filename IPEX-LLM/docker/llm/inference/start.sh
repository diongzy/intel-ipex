
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX 
source ipex-llm-init -t
numactl --cpunodebind=0 -m 0 python ./inference/py