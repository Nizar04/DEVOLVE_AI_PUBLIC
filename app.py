from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
import os
import gc
import re
# Set environment variable for PyTorch memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

app = Flask(__name__)
CORS(app)

# Load the model and tokenizer
base_model = "codellama/CodeLlama-34b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the fine-tuned adapter
output_dir = "/Model/"
model = PeftModel.from_pretrained(model, output_dir)
model.eval()

def extract_schema_sqlite(filepath):
    conn = sqlite3.connect(filepath)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schema = "\n".join(row[0] for row in cursor.fetchall())
    conn.close()
    return schema

def clean_and_format_query(query):
    if not query.strip().endswith(';'):
        query += ';'
    keywords = [
        "select", "from", "where", "insert", "into", "values", "update", "set",
        "delete", "create", "table", "drop", "alter", "join", "inner", "left",
        "right", "full", "on", "group", "by", "having", "order", "asc", "desc",
        "and", "or", "not", "in", "is", "null", "like", "between", "exists", "distinct"
    ]
    formatted_query = " ".join([word.upper() if word.lower() in keywords else word for word in query.split()])
    return formatted_query

# def execute_sql_query(query, conn):
#     query = clean_and_format_query(query)
#     try:
#         query_result = pd.read_sql_query(query, conn)
#         return query_result
#     except pd.io.sql.DatabaseError as e:
#         return f"Database error: {e}"
#     except pd.io.sql.SQLSyntaxError as e:
#         return f"SQL syntax error: {e}"
#     except Exception as e:
#         return f"An unexpected error occurred: {e}"

def generate_text_from_result(query_result, user_question):
    result_str = query_result.to_string(index=False)
    prompt = f"""
    Given the SQL query result and the user's question, generate a natural language description.
    ### User Question:
    {user_question}
    ### SQL Query Result:
    {result_str}
    ### Natural Language Description:
    """
    model_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(**model_input, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return generated_text.split("### Natural Language Description:")[-1].strip()
@app.route('/upload_database', methods=['POST'])
def upload_database():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.db'):
        filename = 'uploaded_database.db'
        file.save(filename)
        return jsonify({'message': 'Database uploaded successfully'}), 200
    return jsonify({'error': 'Invalid file type'}), 400

import re

def extract_sql_query(text):
    # Look for a SQL query starting with SELECT
    match = re.search(r'SELECT\s+.*?;', text, re.IGNORECASE | re.DOTALL)
    if match:
        query = match.group(0)
        # Remove any leading or trailing whitespace and ensure it ends with a semicolon
        query = query.strip()
        if not query.endswith(';'):
            query += ';'
        return query
    return None

@app.route('/process_question', methods=['POST'])
def process_question():
    user_question = request.json['question']
    db_filepath = "uploaded_database.db"
    
    if not os.path.exists(db_filepath):
        return jsonify({'error': 'No database uploaded yet'}), 400
    
    schema = extract_schema_sqlite(db_filepath)
    
    prompt = f"""
    Generate a SQL query to answer this question: {user_question}
    Database schema: {schema}
    Provide only the SQL query as your response, starting with SELECT.
    """
    
    model_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(**model_input, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()

    sql_query = extract_sql_query(generated_text)
    
    if not sql_query:
        return jsonify({'error': 'Failed to generate a valid SQL query'}), 400

    print(f"Generated SQL Query: {sql_query}")
    
    conn = sqlite3.connect(db_filepath)
    result = execute_sql_query(sql_query, conn)
    
    if isinstance(result, str):
        conn.close()
        return jsonify({'error': result})
    else:
        interpreted_result = generate_text_from_result(result, user_question)
        conn.close()
        return jsonify({
            'query': sql_query,
            'result': result.to_dict('records'),
            'interpretation': interpreted_result
        })

def execute_sql_query(query, conn):
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        return f"Error executing query: {str(e)}\nQuery: {query}"
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
