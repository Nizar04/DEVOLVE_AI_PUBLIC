import os
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from peft import PeftModel

nltk.download('punkt')

base_model = "codellama/CodeLlama-34b-hf"
output_dir = "/home/model/lstcheck"
eval_data_file = 'spider_val.json'

bits_config = BitsAndBytesConfig(quantization_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bits_config)
model = PeftModel.from_pretrained(model, output_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model)

eval_dataset = load_dataset('json', data_files=eval_data_file, split='train')

def normalize_sql(query):
    return ' '.join(query.lower().strip().split())

bleu_scores = []
rouge_scores = []

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for data_point in eval_dataset:
    question = data_point['question']
    context = data_point['context']
    expected_answer = normalize_sql(data_point['answer'])

    eval_prompt = f"""
    Generate a SQL query to answer this question: {user_question}
    Database schema: {schema}
    Provide only the SQL query as your response.
    """

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        output = model.generate(**model_input, max_new_tokens=100)

    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    response_start = generated_response.find('### Response:') + len('### Response:')
    generated_sql = normalize_sql(generated_response[response_start:])

    bleu_score = sentence_bleu([expected_answer.split()], generated_sql.split())
    bleu_scores.append(bleu_score)

    rouge_score = scorer.score(expected_answer, generated_sql)
    rouge_scores.append(rouge_score)

avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
avg_rouge = {
    'rouge1': sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0,
    'rouge2': sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0,
    'rougeL': sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0,
}

print(f"Total Questions: {len(eval_dataset)}")
print(f"Average BLEU Score: {avg_bleu:.2f}")
print(f"Average ROUGE Scores: {avg_rouge}")