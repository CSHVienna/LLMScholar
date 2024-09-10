"""
Score Computation Module for the LLMScholar project.

This module provides functions to compute various factuality and consistency scores
for the enhanced data produced by the fact checker. It includes functions for calculating
factuality scores for authors, experts, and specific conditions, as well as consistency scores.

Key functions:
- compute_factuality_author: Computes the factuality score for authors
- compute_factuality_experts: Computes the factuality score for experts
- compute_factuality_condition: Computes the factuality score for specific conditions
- compute_scores: Main function that computes all scores for a given variable

The module relies on the consistency_checks module for computing consistency scores.
"""

import numpy as np
from consistency_checks import average_jaccard_index

def compute_factuality_author(enhanced_df):
    return enhanced_df['Status'].value_counts(normalize=True).get('present', 0)

def compute_factuality_experts(enhanced_df, dataset_df, top_percentage=0.1):
    # Get the total number of authors in the dataset
    total_authors = len(dataset_df)
    
    # Calculate the rank threshold based on the top percentage
    rank_threshold = int(total_authors * top_percentage)
    
    # Count how many authors in the enhanced_df are within the top rank threshold
    top_authors = (enhanced_df['Rank'] <= rank_threshold).sum()
    
    # Calculate the proportion of top authors
    return top_authors / len(enhanced_df) if len(enhanced_df) > 0 else 0


def compute_factuality_condition(enhanced_df, variable):
    if variable.startswith('field'):
        return (enhanced_df['Author Field'].notna() & enhanced_df['DOI Field'].notna()).mean()
    elif variable.startswith('epoch'):
        return enhanced_df['Overlap'].mean()
    elif variable.startswith('top-'):
        k = int(variable.split('-')[1])
        return (enhanced_df['Status'] == 'present').sum() / k
    elif variable.startswith('twin'):
        return (enhanced_df['Cosine_Similarity'] >= 0.8).mean()
    else:
        raise ValueError(f"Unknown variable type: {variable}")

def compute_scores(parsed_df, enhanced_results, dataset_df, variable):
    enhanced_columns = list(enhanced_results.keys())
    scores = {}
    
    # Factuality Author
    scores['Factuality Author'] = np.mean([compute_factuality_author(enhanced_results[col]) for col in enhanced_columns])
    
    # Factuality Experts
    scores['Factuality Experts'] = np.mean([compute_factuality_experts(enhanced_results[col], dataset_df) for col in enhanced_columns])
    
    # Factuality Condition
    scores['Factuality Condition'] = np.mean([compute_factuality_condition(enhanced_results[col], variable) for col in enhanced_columns])
    
    # Consistency Authors
    scores['Consistency Authors'] = average_jaccard_index(parsed_df, variable)
    
    # Consistency Format
    scores['Consistency Format'] = parsed_df.loc[variable, 'Parser Result']
    
    return scores
