"""
Configuration file for the LLMScholar project.

This file contains global constants and configuration variables used throughout the project,
including variable categories, field mappings, and data folder paths.

Constants:
- VARIABLE_CATEGORIES: Categorization of different types of variables used in the project
- FIELD_MAPPING: Mapping of field abbreviations to their full names
- TWIN_IDS: Mapping of twin categories to their corresponding IDs
- Various file paths for data and output

Note: Ensure that the file paths are correctly set for your environment before running the project.
"""

VARIABLE_CATEGORIES = {
    'top-k': ['top-5', 'top-100'],
    'epoch': ['epoch 1950s', 'epoch 2000s'],
    'field': ['field PER', 'field CM&MP'],
    'twin': ['famous male', 'famous female', 'random male', 'random female']
}

FIELD_MAPPING = {
    'PER' : ['Education'], 
    'CM&MP' : ['Condensed Matter Physics', 'General Materials Science']
}

LLM_DATA_FOLDER = "<LLM_DATA_FOLDER>"
APS_DATA_PATH = "<APS_DATA_PATH>"
GENDER_HANDCODED = "<GENDER_HANDCODED_PATH>"
FEATURE_VECTOR_PATH = "<FEATURE_VECTOR_PATH>"

TWIN_IDS = {
    'famous male': {'APS_ID': 3984855223, 'OA_ID': 'A5038976962'},
    'famous female': {'APS_ID': 25364, 'OA_ID': 'A5012905268'},
    'random female': {'APS_ID': 66635, 'OA_ID': 'A5005478445'},
    'random male': {'APS_ID': 143998, 'OA_ID': 'A5023511813'}
}