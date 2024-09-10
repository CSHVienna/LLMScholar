"""
LLM Parser Module for the LLMScholar project.

This module provides functionality to parse and correct JSON or CSV data
using a Large Language Model (LLM) via the Groq API. It's designed to handle
cases where initial parsing of LLM outputs fails, providing a fallback
mechanism to reformat and structure the data.

Key functions:
- llm_parser: Main function to parse problematic LLM outputs
- query_llm: Helper function to interact with the Groq API
- prompt_template: Generates prompts for the LLM based on the parsing task

Note: This module requires the GROQ_API_KEY environment variable to be set.
"""
import pandas as pd
from groq import Groq
import os
import json
from utils import get_variable_columns, extract_json, load_json_as_df
from config import VARIABLE_CATEGORIES

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable before running the project.")
client = Groq(
    api_key=groq_api_key,
)

def llm_parser(text, error, variable, format_data: str) -> pd.DataFrame:

    print(f'llm_parser \n {error} \n {text}')

    # Try the first query with the provided text, error, and variable
    llm_response = str(query_llm(text, error, variable, format_data))
    print(f'llm_response: {llm_response}')

    try:
        json_str = extract_json(llm_response)
        data = json.loads(json_str)
        df = load_json_as_df(data, variable)[1]

        if df.empty or df.isnull().any():
                error = "Invalid DataFrame: Contains NaN or is empty."
                print("RESULT: Invalid DataFrame: Contains NaN or is empty.")
                return pd.DataFrame(columns = ['Names']) 
        
    # try one more time
    except Exception as new_error:
        new_error = f"Error during parsing: {str(new_error)}"
        print(f"New error encountered: {new_error}")
        llm_response = str(query_llm(text, new_error, variable, format_data))
        print(f'second llm_response: {llm_response}')
        try:
            json_str = extract_json(llm_response)
            data = json.loads(json_str)
            df = load_json_as_df(data, variable)[1]

            if df.empty or df.isnull().any():
                    error = "Invalid DataFrame: Contains NaN or is empty."
                    print("RESULT: Invalid DataFrame: Contains NaN or is empty.")
                    return pd.DataFrame(columns = ['Names']) 
            
        except Exception as final_error:
            print(f"Final error encountered: {final_error}")
            return pd.DataFrame(columns = ['Names'])


# Helper function to interact with LLM using the prompt template
def query_llm(text, error, variable, format_data):
    instruction = '''You are a helpful bot specialised in organising text data into a structured JSON format.
    You can only output in dictionary format. You are not allowed to change the content of the text, only the format.
    Your output must be in the form of a dictionary, therefore always starting and ending with curly bracets {{}} Never provide any explanations of what you have done.'''
        
    prompt = prompt_template(text, error, variable, format_data)
    
    # This call is a placeholder for how you'd interact with the specific LLM library
    chat_completion = client.chat.completions.create(
        messages=[{
                "role": "system",
                "content": instruction
            }, {
                "role": "user",
                "content": prompt
            }
        ],
        model='llama3-8b-8192',
        temperature=0,
    )
    
    answer = chat_completion.choices[0].message.content
    return answer


def prompt_template(text, error, variable, format_data):

    columns = get_variable_columns(variable)
    col_placeholders = ', '.join([f'"{col}": [all {col.lower()} retrieved OR empty list]' for col in columns])

    if format_data == 'csv':
        prompt = f'''You will be given a response from a Large Language Model. It contains a list of {', '.join(columns)} in csv format.
        While parsing it in a DataFrame, something went wrong: {error}.
        Your task is to identify any given {', '.join(columns).lower()} inside the text. 
        If the model provides general examples or hypothetical data, you have to recognize it and output an empty dictionary.
        You have to output a dictionary of this form: {{{col_placeholders}}}.
        Provide the corrected JSON string as output.
                input text: {text}
        Avoid any explanation.'''


    elif format_data == 'JSON':

        prompt = f'''You will be given a response from a Large Language Model. It contains a list of {', '.join(columns)} in JSON format. 
        While parsing it in a DataFrame, something went wrong: {error}.
        Your task is to identify any given {', '.join(columns).lower()} inside the text. 
        If the model provides general examples or hypothetical data, you have to recognize it and output an empty dictionary.
        Strip away all infos or text that are not inside the JSON format.

        Guidelines to ensure the JSON can be parsed by json.load:
        1. Each key-value pair should be enclosed in double quotes.
        2. Keys must be always same string and repeated.
        3. Ensure proper use of commas between key-value pairs.
        4. Make sure all opening braces/brackets have corresponding closing braces/brackets.

        Provide the corrected JSON string as output.
                input text: {text}
        Avoid any explanation.'''

    return prompt

