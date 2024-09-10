
"""
Data loading module for the LLMScholar project.

This module provides functionality to load LLM output data from Excel files,
mapping variables to predefined categories and preparing the data for further processing.
"""

import pandas as pd
from config import VARIABLE_CATEGORIES

def load_llm_data(path: str) -> pd.DataFrame:

    df = pd.read_excel(path)

    all_variables = [var for vars_list in VARIABLE_CATEGORIES.values() for var in vars_list]
    dataset_variables = df['Var: variable'].unique()

    if len(dataset_variables) != len(all_variables):
        raise ValueError(f"Mismatch between unique dataset variables ({len(dataset_variables)}) and known categories ({len(all_variables)}).")

    variables_mapping = {dataset_var: all_variables[i] for i, dataset_var in enumerate(dataset_variables)}
    df['Var: variable'] = df['Var: variable'].map(variables_mapping)
    df.set_index('Var: variable', inplace=True)
    df = df[['LLM', 'Prompt', 'Response']]
    return df

if __name__ == "__main__":
    path = '<LLM_DATA_FOLDER>/LLM_data.xlsx'
    df = load_llm_data(path)
    print(df)