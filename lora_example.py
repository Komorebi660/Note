import os, random, json
import numpy as np
from tqdm import tqdm

from datasets import Dataset
from torch.utils.data import DataLoader

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def set_seed_(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset(split="train"):
    data = []
    if split == "train":
        with open("./train.jsonl", "r", encoding="utf8") as f:
            for line in f:
                data_dict = json.loads(line)
                inputs = data_dict['input_field'] + str(data_dict['output_field'])
                data.append({"text": inputs})
    elif split == "valid":
        with open("./valid.jsonl", "r", encoding="utf8") as f:
            for line in f:
                data_dict = json.loads(line)
                prompt = data_dict['input_field'] 
                label = data_dict['output_field']
                data.append({"text": prompt, "label": label})
    else:
        raise NotImplementedError
    return Dataset.from_list(data).shuffle(seed=2024)

def create_dataloader(batch_size):
    dataset = get_dataset("valid")
    return DataLoader(dataset, batch_size=batch_size)


def test(tokenizer, model):
    # pad to left for generation
    tokenizer.padding_side = "left"
    # create dataset
    valid_dataset = create_dataloader(batch_size=16)
    
    results = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(valid_dataset):
            labels.extend(batch["label"].numpy().tolist())
            prompts = batch["text"]
            inputs = tokenizer(prompts, padding=True, truncation=True, \
                               max_length=256, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=20, \
                                     pad_token_id=tokenizer.eos_token_id, \
                                     num_beams=1, do_sample=False)
            for prompt, output in zip(prompts, outputs):
                result = tokenizer.decode(output, skip_special_tokens=True)
                results.append(result[len(prompt):].strip())
    for label, result in zip(labels, results):
        print(f"Label: {label}, Prediction: {result}")
    
    # restore padding side
    tokenizer.padding_side = "right"
    
    return results


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='NousResearch/Meta-Llama-3-8B-Instruct', help="model name")
    parser.add_argument("--bsz", type=int, default=16, help="batch size for signal GPU")
    parser.add_argument("--output_dir", type=str, default="./output", help="output directory")
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{args.model.replace('/', '-')}-finetune")
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed_(2024)

    # LLM model
    print("Loading LLM...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto", cache_dir="./huggingface")

    test(tokenizer, model)
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # load LoRA config
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.3,
        r=512,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        #gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=100,
        save_steps=-1,
        num_train_epochs=0.2,
    )

    # SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=get_dataset("train"),
        peft_config=peft_config,
        dataset_text_field="text",  # "text"域作为inputs
        max_seq_length=300,
        tokenizer=tokenizer,
        # 只优化inputs中“Answer:”之后的结果
        data_collator=DataCollatorForCompletionOnlyLM(response_template="Answer:", tokenizer=tokenizer),
        args=training_args,
    )

    trainer.train()

    # save
    trainer.save_model(args.output_dir)
    trainer.push_to_hub()

    del model
    del tokenizer
    torch.cuda.empty_cache()

    # test
    tokenizer = AutoTokenizer.from_pretrained("yoki123/NousResearch-Meta-Llama-3-8B-Instruct-finetune", cache_dir="./huggingface")
    #tokenizer = AutoTokenizer.from_pretrained(args.output_dir)     # offline load
    base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto", cache_dir="./huggingface")
    model = PeftModel.from_pretrained(base_model, "yoki123/NousResearch-Meta-Llama-3-8B-Instruct-finetune", cache_dir="./huggingface")
    #model = PeftModel.from_pretrained(base_model, args.output_dir) # offline load
    model.eval()
    test(tokenizer, model)
