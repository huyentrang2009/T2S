import gradio as gr
import os
from groq import Groq
import json
import numpy as np
import duckdb
import sqlparse
import pandas as pd
import re
from collections import Counter

def chat_with_groq(client, prompt, model):
    """
    This function sends a prompt to the Groq API and retrieves the AI's response.

    Parameters:
    client (Groq): The Groq API client.
    prompt (str): The prompt to send to the AI.
    model (str): The AI model to use for the response.

    Returns:
    str: The content of the AI's response.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

def get_json_output(llm_response):
    """
    This function cleans the AI's response, extracts the JSON content, and checks if it contains a SQL query or an error message.

    Parameters:
    llm_response (str): The AI's response.

    Returns:
    tuple: A tuple where the first element is a boolean indicating if the response contains a SQL query (True) or an error message (False), 
           and the second element is the SQL query or the error message.
    """
    llm_response_no_escape = llm_response.replace('\\n', ' ').replace('\n', ' ').replace('\\', '').replace('\\', '').strip()
    
    open_idx = llm_response_no_escape.find('{')
    close_idx = llm_response_no_escape.rindex('}') + 1
    cleaned_result = llm_response_no_escape[open_idx : close_idx]

    json_result = json.loads(cleaned_result)
    if 'sql' in json_result:
        query = json_result['sql']
        return True, sqlparse.format(query, reindent=True, keyword_case='upper')
    elif 'error' in json_result:
        return False, json_result['error']
    
def get_reflection(client, full_prompt, llm_response, model):
    """
    This function generates a reflection prompt when there is an error with the AI's response. 
    It then sends this reflection prompt to the Groq API and retrieves the AI's response.

    Parameters:
    client (Groq): The Groq API client.
    full_prompt (str): The original prompt that was sent to the AI.
    llm_response (str): The AI's response to the original prompt.
    model (str): The AI model to use for the response.

    Returns:
    str: The content of the AI's response to the reflection prompt.
    """
    
    reflection_prompt = '''
    You were giving the following prompt:

    {full_prompt}

    This was your response:

    {llm_response}

    There was an error with the response, either in the output format or the query itself.

    Ensure that the following rules are satisfied when correcting your response:
    1. SQL is valid DuckDB SQL, given the provided metadata and the DuckDB querying rules
    2. The query SPECIFICALLY references the correct tables that already existed or were uploaded by the user
    3. Response is in the correct format ({{sql: <sql_here>}} or {{"error": <explanation here>}}) with no additional text?
    4. All fields are appropriately named
    5. There are no unnecessary sub-queries
    6. ALL TABLES are aliased (extremely important)

    Rewrite the response and respond ONLY with the valid output format with no additional commentary

    '''.format(full_prompt = full_prompt, llm_response=llm_response)

    return chat_with_groq(client, reflection_prompt, model)

def get_summarization(client, user_question, df, model):
    prompt = '''
    A user asked the following question pertaining to local database tables:
    
    {user_question}
    
    To answer the question, a dataframe was returned:

    Dataframe:
    {df}

    In a few sentences, summarize the data in the table as it pertains to the original user question. Avoid qualifiers like "based on the data" and do not comment on the structure or metadata of the table itself
    '''.format(user_question = user_question, df = df)

    return chat_with_groq(client, prompt, model)

def generate_table_metadata(file_paths):
    table_metadata = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        columns = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
        table_metadata.append(f"Table: {table_name}\nColumns: {columns}")
    
    return "\n\n".join(table_metadata)

def update_base_prompt(file_paths, base_prompt_path='prompts/bp.txt'):
    table_metadata = generate_table_metadata(file_paths)
    
    with open(base_prompt_path, 'r') as file:
        base_prompt = file.read()
    
    base_prompt = base_prompt.replace("{table_metadata}", table_metadata)
    
    with open(base_prompt_path, 'w') as file:
        file.write(base_prompt)

    return base_prompt

def handle_question(user_question, model, max_num_reflections, files):
    
    client = Groq(api_key='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')
    
    conn = duckdb.connect(database=':memory:', read_only=False)
    #print("Files received:", files)  # Debug: print files received

    # Load existing CSV files in the data directory
    data_dir = 'data'
    existing_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if files is None:
        all_files = existing_files
    else:
        all_files = [file.name if isinstance(file, gr.File) else file for file in files] + existing_files
    
    #print("All files:", all_files)  # Debug: print all files

    table_names = []
    for file in all_files:
        #print("Processing file:", file)  # Debug: print each file being processed
        if os.path.exists(file):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    # Convert date columns to datetime type
                    for col in df.columns:
                        if "date" in col.lower() or "birth" in col.lower():
                            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                    table_name = os.path.splitext(os.path.basename(file))[0]
                    #print("Registering table:", table_name)  # Debug: print table being registered
                    conn.register(table_name, df)
                    table_names.append(table_name)
                except Exception as e:
                    print(f"Failed to read {file} as CSV: {e}")
            elif file.endswith('.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.json_normalize(data)
                    # Convert date columns to datetime type
                    for col in df.columns:
                        if "date" in col.lower() or "birth" in col.lower():
                            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                    table_name = os.path.splitext(os.path.basename(file))[0]
                    #print("Registering table:", table_name)  # Debug: print table being registered
                    conn.register(table_name, df)
                    table_names.append(table_name)
                except Exception as e:
                    print(f"Failed to read {file} as JSON: {e}")
        else:
            print(f"File not found: {file}")


    table_metadata = generate_table_metadata(all_files)
    
    with open('prompts/base_prompt.txt', 'r') as file:
        base_prompt = file.read()

    full_prompt = base_prompt.format(table_metadata = table_metadata, user_question=user_question)
    llm_response = chat_with_groq(client, full_prompt, model)

    valid_response = False
    i = 0
    while not valid_response and i < max_num_reflections:
        try:
            is_sql, result = get_json_output(llm_response)
            if is_sql:
                if any(table in result for table in table_names):
                    results_df = conn.execute(result).fetchdf().reset_index(drop=True)
                    valid_response = True
                else:
                    raise ValueError("Generated SQL query references tables not in uploaded files.")
            else:
                valid_response = True
        except Exception as e:
            print("Reflection error:", e)  # Debug: print reflection error
            llm_response = get_reflection(client, full_prompt, llm_response, model)
            i += 1

    if valid_response:
        if is_sql:
            summarization = get_summarization(client, user_question, results_df, model)
            return f"```sql\n{result}\n```", results_df, summarization.replace('$', '\\$')
        else:
            return result, None, None
    else:
        return "ERROR: Could not generate valid SQL for this question", None, None
    
def main():
    article = "<h3>How to Use:</h3> " \
        "<ul><li>Open NEWAI's SQL Query Generator.</li> " \
        "<li>Enter your question in the provided question box.</li>" \
        "<li>Choose one of the available models.</li>" \
        "<li>If you want to upload your own database, click on the upload box and upload it.</li>" \
        "<li>Click on the 'Submit' button. <strong>Voila!</strong>. Your SQL query and the result will appear in the left boxes. " \
    
    desc= "Welcome to NEWAI's SQL Query Generator! Feel free to ask questions about the available data named The Chinook, \
                which represents a digital media store, including tables for artists, albums, media tracks, invoices, and customers. \
                For example, you could ask `How many artists are there?`or \
                `What's the average invoice from an American customer whose Fax is missing since 2003 but before 2010?`.\
                You can also upload your data files and ask questions about them. \
                The application matches your question to SQL queries to provide accurate and relevant results. Enjoy exploring the data!"
                   
    demo = gr.Interface(
        fn=handle_question,
        inputs=[
            gr.Textbox(label="Ask a question"),
            gr.Dropdown(['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma2-9b-it', 'llama-3.1-8b-instant'], label="Choose a model"),
            gr.Slider(0, 10, value=5, step=1, label="Max reflections"),
            gr.Files(label="Upload a file", file_count="multiple", type="filepath"),
        ],
        outputs=[
            gr.Textbox(label="SQL Query"),
            gr.Dataframe(label="Query Results"),
            gr.Textbox(label="Summarization")
        ],
        title="NEWAI's Query Generator",
        description= desc,
        article = article,
        theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg)
        )

    demo.launch()
    
if __name__ == "__main__":
    main()