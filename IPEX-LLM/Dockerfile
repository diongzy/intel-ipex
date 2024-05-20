FROM ubuntu:20.04 as key-getter


RUN mkdir -p /ipex_llm/data && mkdir -p /ipex_llm/model && \
    # get qlora example code
    cd /ipex_llm && \
    mv /home/qmed-intel/IPEX_LLM/CPU/QLoRA-FineTuning/* . && \
    mkdir -p /GPU/LLM-Finetuning && \
    mv IPEX-LLM/python/llm/example/GPU/LLM-Finetuning/common /GPU/LLM-Finetuning/common && \
    rm -r IPEX-LLM

# for standalone
COPY ./start-qlora-finetuning-on-cpu.sh /ipex_llm/start-qlora-finetuning-on-cpu.sh
RUN chown -R mpiuser /ipex_llm

USER mpiuser

ENTRYPOINT ["/bin/bash"]
