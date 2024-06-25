import os
import torch
import wandb
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig, EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import set_seed

def setup_model():
    set_seed(42)
    train_data = load_dataset('json', data_files='spider_train.json', split='train')
    val_data = load_dataset('json', data_files='spider_val.json', split='train')
    val_size = int(0.1 * len(train_data))
    train_data, eval_data = train_data.train_test_split(test_size=val_size).values()
    bits_config = BitsAndBytesConfig(quantization_dtype=torch.float16)
    lm_model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-34b-hf", quantization_config=bits_config)
    text_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-hf")
    text_tokenizer.add_eos_token = True
    text_tokenizer.pad_token_id = 0
    text_tokenizer.padding_side = "left"
    return train_data, val_data, eval_data, lm_model, text_tokenizer

def tokenize_data(tokenizer, dataset):
    def tokenize(prompt):
        result = tokenizer(prompt, truncation=True, max_length=512, padding=False)
        result["labels"] = result["input_ids"].copy()
        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = f"""  
        Generate a SQL query to answer this question: {user_question}
        Database schema: {schema}
        Provide only the SQL query as your response.
        """
        return tokenize(full_prompt)
    
    processed_dataset = dataset.map(generate_and_tokenize_prompt)
    return processed_dataset

def begin_training(train_dataset, val_dataset, eval_dataset, model, tokenizer):
    model.train()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    trained_model = get_peft_model(model, peft_config)
    
    wandb.init(project="CodeLLama-34B")  
    
    training_args = TrainingArguments(
        output_dir="/home/outputs",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=150,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=10,
        group_by_length=True,
        report_to="wandb",
        lr_scheduler_type='linear', 
        run_name=f"codellamafinalRUN",
        load_best_model_at_end=True,  # Added to resolve EarlyStoppingCallback requirement
    )
    
    trainer = Trainer(
        model=trained_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  
    )
    
    trainer.train()
    
    evaluation_results = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Evaluation Results: {evaluation_results}")

    model_path = "/home/model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

if __name__ == "__main__":
    train_dataset, val_dataset, eval_dataset, model, tokenizer = setup_model()
    tokenized_train = tokenize_data(tokenizer, train_dataset)
    tokenized_val = tokenize_data(tokenizer, val_dataset)
    tokenized_eval = tokenize_data(tokenizer, eval_dataset)
    begin_training(tokenized_train, tokenized_val, tokenized_eval, model, tokenizer)
