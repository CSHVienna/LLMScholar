"""
Utility functions for the LLMScholar project.

This module contains various helper functions used throughout the project for data processing,
manipulation, and analysis. It includes functions for number manipulation, JSON parsing,
dataframe operations, and string preprocessing.

Key functions:
- truncate_number: Rounds a number to a specified number of decimal places
- get_variable_columns: Determines the relevant columns based on the variable category
- extract_json: Extracts JSON data from a string
- load_json_as_df: Converts JSON data to a pandas DataFrame
- parse_year_range: Parses year ranges from string inputs
- preprocess_name: Standardizes name strings for consistency
- ranges_overlap: Checks if two date ranges overlap

Note: This module relies on configurations from the config.py file.
"""

import math
import numpy as np
import json
import re
import pandas as pd
from typing import Tuple, List
from config import VARIABLE_CATEGORIES
from unidecode import unidecode

def truncate_number(x: float, decimals: int = 2) -> float:
    if np.isnan(x):
        return np.nan  # Return NaN if input is NaN
    factor = 10 ** decimals
    return math.floor(x * factor) / factor


def list_to_tuple(list):
    # return (mean of the list, list)
    return (truncate_number(np.mean(list), 2), list)

def get_variable_columns(variable):
    for category, variables in VARIABLE_CATEGORIES.items():
        if variable in variables:
            if category == 'epoch':
                return ['Names', 'Years']
            elif category == 'field':
                return ['Names', 'Papers']
            else:
                return ['Names']
    return ['Names']

def change_column_names(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    new_columns = get_variable_columns(variable)
    df.columns = new_columns
    return df

def extract_json(text: str) -> str:
    try:
        json_str = re.search(r'\{.*\}', text, re.DOTALL).group(0)
        return json_str
    except AttributeError:
        raise ValueError("No JSON object found in the input text")


def load_json_as_df(json_str: str, variable: str) -> Tuple[bool, pd.DataFrame]:
    try:
        data = json.loads(json_str)
        
        if not isinstance(data, dict) or not data:
            return False, pd.DataFrame()
        
        key = list(data.keys())[0]
        entries = data[key]
        required_columns = get_variable_columns(variable)
        
        if not isinstance(entries, list) or not entries:
            return False, pd.DataFrame()
        
        if all(isinstance(entry, dict) for entry in entries):
            df = pd.json_normalize(entries)
            df = change_column_names(df, variable)
        
        elif all(isinstance(entry, str) for entry in entries) and len(required_columns) == 1:
            # Entries are strings and only one column is required
            df = pd.DataFrame(entries, columns=required_columns)
            df = change_column_names(df, variable)
        else:
            return False, pd.DataFrame()
        
        return True, df
    
    except (json.JSONDecodeError, ValueError):
        return False, pd.DataFrame()


def parse_year_range(value_str, default_start=None, default_end=None):
    value_str = value_str.strip().lower()
    present_terms = ['present', 'current', 'ongoing', 'now']

    # Replace any occurrence of 'present' or similar terms with 2020
    for term in present_terms:
        if term in value_str:
            value_str = value_str.replace(term, '2020')

    # Handle cases like 'not applicable', 'n/a', 'unknown'
    if any(keyword in value_str for keyword in ['not applicable', 'n/a', 'unknown']):
        return default_start, default_end

    try:
        if '-' in value_str:
            start_year, end_year = map(int, value_str.split('-'))
            return start_year, end_year
        else:
            year = int(value_str)
            return year, year
    except ValueError:
        return default_start, default_end
    
def preprocess_name(name):
    name = unidecode(name)  # Normalize special characters
    name = re.sub(r'\b(Jr|Dr|Sr)\.?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[.,-]', '', name)
    return name.lower().strip() 

def ranges_overlap(start1, end1, start2, end2, tolerance=3):
    # If any range values are None, consider them invalid for comparison
    if None in [start1, end1, start2, end2]:
        return False

    # Check for overlap with the given tolerance
    if start1 - tolerance <= end2 and end1 + tolerance >= start2:
        return True
    return False