export ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI
export LD_PRELOAD=export LD_PRELOAD=/usr/local/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:60000,muzzy_decay_ms:-1"

n_cores=$(lscpu | awk '/^Core\(s\) per socket/{ print $4 }')
export OMP_NUM_THREADS=$n_cores
source ipex-llm-init -t

numactl --cpunodebind=0 -m 0 python ./alpaca_qlora_finetuning_cpu.py \
--base_model "/home/qmed-intel/models/meta-llama/Meta-Llama-3-8B-Instruct" \
--data_path "/home/qmed-intel/Documents/GitHub/intel-ipex/IPEX-LLM/python/llm/example/CPU/QLoRA-FineTuning/alpaca-qlora/templates/train.json" \
--output_dir "./ipex-llm-qlora-alpaca" \
--prompt_template_name "medical-llama" \
--num_epochs 8