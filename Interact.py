import sqlite3
import mysql.connector
import psycopg2
import cx_Oracle
import pymongo
import pyodbc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd

# Load the model and tokenizer
base_model = "codellama/CodeLlama-34b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")

# Set pad_token_id to eos_token_id to avoid warnings
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the fine-tuned adapter
output_dir = "/home/model/lstcheck/"
model = PeftModel.from_pretrained(model, output_dir)
model.eval()

def extract_schema_sqlite(filepath):
    conn = sqlite3.connect(filepath)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schema = "\n".join(row[0] for row in cursor.fetchall())
    conn.close()
    return schema

def extract_schema_mysql(user, password, host, database):
    conn = mysql.connector.connect(user=user, password=password, host=host, database=database)
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES;")
    tables = cursor.fetchall()
    schema = ""
    for (table_name,) in tables:
        cursor.execute(f"SHOW CREATE TABLE {table_name};")
        schema += cursor.fetchone()[1] + ";\n"
    conn.close()
    return schema

def extract_schema_postgresql(user, password, host, database):
    conn = psycopg2.connect(dbname=database, user=user, password=password, host=host)
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    tables = cursor.fetchall()
    schema = ""
    for (table_name,) in tables:
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table_name}'")
        columns = cursor.fetchall()
        columns_sql = ", ".join(f"{col_name} {data_type}" for col_name, data_type in columns)
        schema += f"CREATE TABLE {table_name} ({columns_sql});\n"
    conn.close()
    return schema

def extract_schema_oracle(user, password, host, service_name):
    dsn = cx_Oracle.makedsn(host, 1521, service_name=service_name)
    conn = cx_Oracle.connect(user, password, dsn)
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM user_tables")
    tables = cursor.fetchall()
    schema = ""
    for (table_name,) in tables:
        cursor.execute(f"SELECT column_name, data_type FROM user_tab_columns WHERE table_name='{table_name}'")
        columns = cursor.fetchall()
        columns_sql = ", ".join(f"{col_name} {data_type}" for col_name, data_type in columns)
        schema += f"CREATE TABLE {table_name} ({columns_sql});\n"
    conn.close()
    return schema

def extract_schema_sqlserver(user, password, host, database):
    conn_string = f'DRIVER={{SQL Server}};SERVER={host};DATABASE={database};UID={user};PWD={password}'
    conn = pyodbc.connect(conn_string)
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_type='BASE TABLE'")
    tables = cursor.fetchall()
    schema = ""
    for (table_name,) in tables:
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table_name}'")
        columns = cursor.fetchall()
        columns_sql = ", ".join(f"{col_name} {data_type}" for col_name, data_type in columns)
        schema += f"CREATE TABLE {table_name} ({columns_sql});\n"
    conn.close()
    return schema

def list_collections_mongodb(uri):
    client = pymongo.MongoClient(uri)
    db = client.get_default_database()
    collections = db.list_collection_names()
    schema = "Collections: " + ", ".join(collections)
    client.close()
    return schema

def clean_and_format_query(query):
    # Add a semicolon at the end if missing
    if not query.strip().endswith(';'):
        query += ';'
    
    # Ensure proper casing (convert keywords to uppercase, columns and table names to lowercase)
    keywords = ["select", "from", "where", "insert", "into", "values", "update", "set", "delete", "create", "table", "drop", "alter", "join", "inner", "left", "right", "full", "on", "group", "by", "having", "order", "asc", "desc", "and", "or", "not", "in", "is", "null", "like", "between", "exists", "distinct"]
    formatted_query = " ".join([word.upper() if word.lower() in keywords else word for word in query.split()])
    
    return formatted_query

def execute_sql_query(query, conn):
    query = clean_and_format_query(query)
    try:
        query_result = pd.read_sql_query(query, conn)
        return query_result
    except pd.io.sql.DatabaseError as e:
        return f"Database error: {e}"
    except pd.io.sql.SQLSyntaxError as e:
        return f"SQL syntax error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

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

print("Select the type of your database:")
print("1. SQLite")
print("2. MySQL")
print("3. PostgreSQL")
print("4. Oracle")
print("5. SQL Server")
print("6. MongoDB")

db_choice = input("Enter the number corresponding to your database type: ").strip()
schema = ""
conn = None
client = None

if db_choice == "1":
    db_path = input("Enter the path to your SQLite database file: ")
    schema = extract_schema_sqlite(db_path)
    conn = sqlite3.connect(db_path)
elif db_choice == "2":
    user = input("Enter your MySQL username: ")
    password = input("Enter your MySQL password: ")
    host = input("Enter your MySQL host, usually localhost: ")
    database = input("Enter your MySQL database name: ")
    schema = extract_schema_mysql(user, password, host, database)
    conn = mysql.connector.connect(user=user, password=password, host=host, database=database)
elif db_choice == "3":
    user = input("Enter your PostgreSQL username: ")
    password = input("Enter your PostgreSQL password: ")
    host = input("Enter your PostgreSQL host, usually localhost: ")
    database = input("Enter your PostgreSQL database name: ")
    schema = extract_schema_postgresql(user, password, host, database)
    conn = psycopg2.connect(dbname=database, user=user, password=password, host=host)
elif db_choice == "4":
    user = input("Enter your Oracle username: ")
    password = input("Enter your Oracle password: ")
    host = input("Enter your Oracle host: ")
    service_name = input("Enter your Oracle service name: ")
    schema = extract_schema_oracle(user, password, host, service_name)
    dsn = cx_Oracle.makedsn(host, 1521, service_name=service_name)
    conn = cx_Oracle.connect(user, password, dsn)
elif db_choice == "5":
    user = input("Enter your SQL Server username: ")
    password = input("Enter your SQL Server password: ")
    host = input("Enter your SQL Server host, usually localhost: ")
    database = input("Enter your SQL Server database name: ")
    schema = extract_schema_sqlserver(user, password, host, database)
    conn_string = f'DRIVER={{SQL Server}};SERVER={host};DATABASE={database};UID={user};PWD={password}'
    conn = pyodbc.connect(conn_string)
elif db_choice == "6":
    uri = input("Enter your MongoDB URI (include the database name in the URI): ")
    schema = list_collections_mongodb(uri)
    client = pymongo.MongoClient(uri)
    db = client.get_default_database()
else:
    print("Invalid database selection. Exiting.")
    exit()

print("Database schema extracted. Please provide a SQL-related question.")

while True:
    user_question = input("Enter your SQL-related question or type 'quit' to exit: ")
    if user_question.lower() == "quit":
        print("Exiting the program.")
        break

    print("Generating the SQL query, please wait...")
    prompt = f"""
    Generate a SQL query to answer this question: {user_question}
    Database schema: {schema}
    Provide only the SQL query as your response.
    """

    model_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(**model_input, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Extract only the SQL query part
    response_start = "### Response:\n"
    sql_query = generated_text.split(response_start)[-1].strip()
    print("\nGenerated SQL Query:")
    print(sql_query)

    if db_choice != "6":
        execute_choice = input("Do you want to execute this SQL query against the database? (yes/no): ").strip().lower()
        if execute_choice == "yes":
            result = execute_sql_query(sql_query, conn)
            if isinstance(result, str) and result.startswith("An error occurred"):
                print("\nQuery Execution Error:")
                print(result)
            else:
                print("\nQuery Result:")
                print(result)
                text_result = generate_text_from_result(result, user_question)
                print("\nInterpreted Result:")
                print(text_result)
    
    continue_choice = input("Do you want to ask another question? (yes/no): ").strip().lower()
    if continue_choice != "yes":
        print("Exiting the program.")
        break

if conn:
    conn.close()
if client:
    client.close()
