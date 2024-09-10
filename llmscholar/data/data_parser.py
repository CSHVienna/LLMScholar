"""
LLM parsing module for the LLMScholar project.

This module handles the parsing of LLM outputs that couldn't be processed by the main parser.
It uses the Groq API to reformat and structure the data into a valid JSON format.

Key functions:
- llm_parser: Main function to parse problematic LLM outputs
- query_llm: Helper function to interact with the Groq API
- prompt_template: Generates prompts for the LLM based on the parsing task

Note: This module requires the GROQ_API_KEY environment variable to be set.
"""

import pandas as pd
from utils import truncate_number, get_variable_columns, extract_json, load_json_as_df
import re
from typing import Tuple, Any
from llm_parser import llm_parser
from tqdm import tqdm
from data_loader import load_llm_data

def extract_usage_info(usage_str):

    pattern = r'(\w+)=([\d\.]+|None)'
    matches = re.findall(pattern, usage_str)
    usage_info = {key: (None if value == 'None' else float(value)) for key, value in matches}
    return usage_info

def parse_data(llm_df: pd.DataFrame, format_data: str) -> pd.DataFrame:
    df = llm_df.copy()
    parsed_dict = {}

    for name, group in tqdm(df.groupby(level=0), desc="Processing Groups"):
        parsed_data = {'Prompt': group['Prompt'].values[0]}
        parsing_success = []

        prompt_tokens = None
        completion_tokens_list = []
        total_tokens_list = []
        total_time_list = []

        for i, response in enumerate(group['Response']):
            answer_col = f"Answer_{i}"
            parsed_data[answer_col] = response

            # Extract and store Usage information
            usage_str_start = response.rfind("Usage(")
            if usage_str_start != -1:
                usage_str = response[usage_str_start:]
                usage_info = extract_usage_info(usage_str)
                
                # Store all usage information in a single column
                parsed_data[f"Usage_{i}"] = usage_info

                # Collect usage metrics for aggregation
                if 'prompt_tokens' in usage_info and prompt_tokens is None:
                    prompt_tokens = usage_info['prompt_tokens']
                if 'completion_tokens' in usage_info:
                    completion_tokens_list.append(usage_info['completion_tokens'])
                if 'total_tokens' in usage_info:
                    total_tokens_list.append(usage_info['total_tokens'])
                if 'total_time' in usage_info:
                    total_time_list.append(usage_info['total_time'])

            parsed_col = f"Parsed_answer_{i}"
            
            # Check if the response is the same as the previous response
            previous_responses = [parsed_data[f"Answer_{x}"] for x in range(i)]
            if response in previous_responses:
                id = previous_responses.index(response)
                parsed_result = parsed_data[f"Parsed_answer_{id}"]
                success = parsing_success[id]
            else:
                if format_data == 'csv':
                    success, parsed_result = parse_csv(response, name)
                elif format_data == 'JSON':
                    success, parsed_result = parse_json(response, name)

            parsed_data[parsed_col] = parsed_result
            parsing_success.append(success)

        parsed_data['Parser Result'] = truncate_number(sum(parsing_success) / len(parsing_success), 2)
        parsed_data['Parser Details'] = parsing_success
        
        # Calculate Usage Result
        if prompt_tokens is not None:
            usage_result = {
                'prompt_tokens': prompt_tokens,
                'average_completion_tokens': sum(completion_tokens_list) / len(completion_tokens_list) if completion_tokens_list else None,
                'average_total_tokens': sum(total_tokens_list) / len(total_tokens_list) if total_tokens_list else None,
                'average_total_time': sum(total_time_list) / len(total_time_list) if total_time_list else None
            }
            parsed_data['Usage Result'] = usage_result

        parsed_dict[name] = parsed_data

    parsed_df = pd.DataFrame.from_dict(parsed_dict, orient='index')
    return parsed_df


#### CSV DATA
# The prompt MUST ask for CSV file and to start and finish with START and END tokens

def parse_csv(text, variable):
    try:
        # Extract data between "START" and "END"
        start_idx = text.index('START') + len('START')
        end_idx = text.index('END')
        data = text[start_idx:end_idx].strip()

        # Split data into rows
        rows = data.splitlines()

        # Handle multi-row or single-row CSVs
        if len(rows) > 2:
            # Multi-row case
            header = rows[0].split(',')
            non_empty_indices = [index for index, col in enumerate(header) if col != ""]
            data_rows = [[row.split(',')[index] for index in non_empty_indices] for row in rows[1:]]
            columns = get_variable_columns(variable)
            df = pd.DataFrame(data_rows, columns=columns)
        else:
            # Single-row case
            header = rows[0].split(',')
            data_values = rows[1].split(',')
            df = pd.DataFrame({'Names': [value.strip() for value in data_values]})

        # Validate the DataFrame
        if df.empty or df.isnull().any().any():
            error = "Invalid DataFrame: Contains NaN or is empty."
            return False, llm_parser(text, error, variable, format_data = 'csv')

    except Exception as e:
        # Error during parsing
        error = f"Error during parsing: {str(e)}"
        return False, llm_parser(text, error, variable, format_data = 'csv')


    return True, df


#### JSON DATA
# The prompt MUST ask for JSON file and start and finish with {} tokens

def parse_json(llm_answer: str, variable: str) -> Tuple[bool, pd.DataFrame]:
    try:
        json_str = extract_json(llm_answer)
        success, df = load_json_as_df(json_str, variable)
        if success and not df.empty:
            return True, df
        else:
            return False, llm_parser(llm_answer, "DataFrame is empty after parsing", variable, format_data='JSON')
    except ValueError as e:
        return False, llm_parser(llm_answer, str(e), variable, format_data='JSON')

if __name__ == "__main__":
    path = '<LLM_DATA_FOLDER>/LLM_data.xlsx'
    df = load_llm_data(path)
    parsed_df = parse_data(df, 'JSON')
    print(parsed_df)