#!/usr/bin/env python
# coding: utf-8

import os
os.environ['HF_HOME'] = 'your-cache-dir'

from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
import argparse
import yaml

# function to parse arguments
def parse_args():
    arg_parser = argparse.ArgumentParser(description='Run finetuning with specified parameters.')
    arg_parser.add_argument('--hf_token', type=str, required=True, help='HuggingFace token')
    arg_parser.add_argument('--config', type=str, default="unsloth_finetuning_config.yaml", help='Path to the YAML config file')
    return arg_parser.parse_args()

# function to read yaml file
def read_yaml(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

unsloth_template = (
    "{{ bos_token }}"
    "{{ 'You are a helpful assistant to the user\n' }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '>>> User: ' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '>>> Assistant: ' }}"
    "{% endif %}"
)
unsloth_eos_token = "eos_token"

# main function to run the script
def main():
    args = parse_args()
    config = read_yaml(args.config)

    # set variables from config
    system_message = config.get('system_message', 'You are a nice bot!')
    max_seq_length = config.get('max_seq_len', 2048)
    dtype = None
    load_in_4bit = True
    use_gradient_checkpointing = config.get('use_gradient_checkpointing', True)
    random_state = 3407
    r = 16
    lora_alpha = 16
    lora_dropout = 0
    bias = "none"
    use_rslora = False
    loftq_config = None

    model_id = config['model_id']
    output_model_id = config['output_model_id']
    huggingface_dataset = config['huggingface_dataset']
    output_dir = config['output_dir']
    learning_rate = config['learning_rate']
    lr_scheduler_type = config['lr_scheduler_type']
    num_train_epochs = config['num_train_epochs']
    per_device_train_batch_size = config['per_device_train_batch_size']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    optim = config['optim']
    logging_steps = config['logging_steps']
    warmup_steps = config['warmup_steps']
    chat_template = config['chat_template']
    
    merged_16bit = config.get('merged_16bit', False)
    merged_4bit = config.get('merged_4bit', False)
    lora = config.get('lora', False)
    f16 = config.get('f16', False)
    q4_k_m = config.get('q4_k_m', False)
    
    print(f"HF Token: {args.hf_token}")
    print(f"System Message: {system_message}")
    print(f"Max Sequence Length: {max_seq_length}")
    print(f"Model ID: {model_id}")
    print(f"Huggingface Dataset: {huggingface_dataset}")
    print(f"Output Directory: {output_dir}")
    print(f"Learning Rate: {learning_rate}")
    print(f"LR Scheduler Type: {lr_scheduler_type}")
    print(f"Number of Training Epochs: {num_train_epochs}")
    print(f"Batch Size: {per_device_train_batch_size}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"Optimizer: {optim}")
    print(f"Logging Steps: {logging_steps}")
    print(f"Gradient Checkpointing: {use_gradient_checkpointing}")
    print(f"Warm-up steps: {warmup_steps}")
    print(f"Chat template: {chat_template}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=args.hf_token,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=loftq_config,
    )

    def create_conversation(sample):
        if sample["messages"][0]["role"] == "system":
            return sample
        else:
            sample["messages"] = [{"content": system_message, "role": "system"}] + sample["messages"]
            return sample

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts, }
    
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )

    dataset = load_dataset(huggingface_dataset, split="train")

    columns_to_remove = list(dataset.features)
    columns_to_remove.remove("messages")
    dataset = dataset.map(create_conversation, remove_columns=columns_to_remove, batched=False)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            # max_steps=None,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=0.01,
            lr_scheduler_type=lr_scheduler_type,
            seed=random_state,
            output_dir=output_dir,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
        map_eos_token=True,
    )

    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "user", "content": "Que pensás de Cristina? "},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True)
    print(tokenizer.batch_decode(outputs))

    model.save_pretrained("lora_model")
    model.push_to_hub(output_model_id, token=args.hf_token)
    
    # saving to float16 for VLLM
    
    # merge to 16bit
    if merged_16bit: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
    if merged_16bit: model.push_to_hub_merged(f"machinelearnear/{output_model_id}_merged_16bit", tokenizer, save_method = "merged_16bit", token=args.hf_token)

    # merge to 4bit
    if merged_4bit: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
    if merged_4bit: model.push_to_hub_merged(f"machinelearnear/{output_model_id}_merged_4bit", tokenizer, save_method = "merged_4bit", token=args.hf_token)

    # just LoRA adapters
    if lora: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
    if lora: model.push_to_hub_merged(f"machinelearnear/{output_model_id}_lora", tokenizer, save_method = "lora", token=args.hf_token)
    
    # GGUF / llama.cpp conversion
    
    # save to 8bit Q8_0
    if False: model.save_pretrained_gguf("model", tokenizer,)
    if False: model.push_to_hub_gguf(f"machinelearnear/{output_model_id}", tokenizer, token=args.hf_token)

    # save to 16bit GGUF
    if f16: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
    if f16: model.push_to_hub_gguf(f"machinelearnear/{output_model_id}_gguf_f16", tokenizer, quantization_method = "f16", token=args.hf_token)

    # save to q4_k_m GGUF
    if q4_k_m: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    if q4_k_m: model.push_to_hub_gguf(f"machinelearnear/{output_model_id}_gguf_q4_k_m", tokenizer, quantization_method = "q4_k_m", token=args.hf_token)
    

if __name__ == '__main__':
    main()