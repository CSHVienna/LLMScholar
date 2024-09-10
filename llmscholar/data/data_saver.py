"""
Data Saver Module for the LLMScholar project.

This module provides utility functions for saving processed data to CSV files.
It includes functions to convert DataFrame objects to dictionaries and to save
the criteria DataFrame with special handling for enhanced answer columns.
"""
import pandas as pd

def dataframe_to_dict(df):
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')
    return df

def save_criteria_df(criteria_df, file_path):
    criteria_df_to_save = criteria_df.copy()
    enhanced_answer_cols = [col for col in criteria_df.columns if col.startswith('Enhanced_answer_')]
    
    for col in enhanced_answer_cols:
        criteria_df_to_save[col] = criteria_df_to_save[col].apply(dataframe_to_dict)
    
    criteria_df_to_save.to_csv(file_path)
    print(f"Data saved as {file_path}")