import pandas as pd
import json
import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from ipex_llm.transformers import AutoModelForCausalLM #Updated line
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes
import accelerate
import transformers
from trl import SFTTrainer
from threading import Thread
import numpy as np
import argparse
from tqdm import tqdm
import time
from datetime import datetime

os.environ["HF_TOKEN"] = "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""

##########################################################################
##### IMPORTANT THAT BNB CONFIG FOLLOWS CONFIG IN TRAINING RUN############
def load_llama3():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    folder_path = "/home/qmed-intel/models/"
    model_path = folder_path+model_name  
    # Qmed/FT_llama_instruct_summarise_merged (copy)
    # 
    # base_model_id = "Qmed/FT_llama_instruct_summarise_merged"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False, #
        bnb_4bit_quant_type="int4", #sym_int4 for CPU
        bnb_4bit_compute_dtype=torch.bfloat16 
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name,add_eos_token=True,add_bos_token=True)

    return model, tokenizer

def pipeline_llama(llama_tokenizer, model):
    model.eval()

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=llama_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    
    df = pd.read_csv('50_samples_less_than_0.8.csv')
    # df = pd.read_excel('test.xlsx') #n=14
    input_cases = df['Input JSON'].tolist()
    original_summaries = df['Generated Summary'].tolist()
    
    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Input Query', 'Generated Summary', 'GPT Summary', 
                                       'Output Length', 'Input Length', 'Total Generation Time (s)', 
                                       'First Token Latency (s)', 'Nth Token Latency (s)', 
                                       'Average first token latency (ms)', 'Average nth token latency (ms)', 
                                       'Token per second (ms)'])
    query = 0

    for patient_case, original_summary in zip(input_cases, original_summaries):
        query += 1
        messages = [
            {"role": "system", "content": "You are a medical doctor who are good in summarizing patient self-assessment forms into clinical notes format. You digest all the patient information, think steps by steps, based on your reasoning make a concise summary, which MUST include age, gender, Chief Complaint (symptom and duration), History of Presenting Illness (symptom elaboration, additional symptoms, important negatives), Past Medical and Surgical History, Drug and Allergy History (medications, allergies), Family and Social History (smoking, alcohol habits), ending with the patient's smoking status. You will not provide medical diagnoses or recommendations. Ensure all elements are covered concisely.\n Follow format: \n 'Patient is a x years old of.... End with he/she does not smoke."},
            {"role": "user", "content": f'{patient_case}' },
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            llama_tokenizer.eos_token_id,
            llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        print(outputs[0]["generated_text"][len(prompt):])
        # Extract the generated text
        generated_text = outputs[0]["generated_text"][len(prompt):]

        current_df = pd.DataFrame({
            'Input Query': [patient_case],
            'Generated Summary': [generated_text],
            'GPT Summary': [original_summary],
        })
        
        # Append the current DataFrame to the results DataFrame using concat
        results_df = pd.concat([results_df, current_df], ignore_index=True)
    
    # Save the DataFrame to a CSV file
    results_df.to_csv('inferencen=50.csv', index=False)

# ##FT MODEL
model, tokenizer = load_llama3()
ft_model = PeftModel.from_pretrained(model, "/home/qmed-intel/Documents/llama_instruct_2")
ft_model.to('cpu')
pipeline_llama(tokenizer, ft_model)
