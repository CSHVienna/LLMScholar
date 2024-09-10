"""
Consistency checks module for the LLMScholar project.

This module provides functions to evaluate the consistency of LLM outputs
using Jaccard similarity for sets of names. It's primarily used to assess
how consistent the LLM is in generating lists of names across multiple runs.

Functions:
- compute_jaccard_index: Calculates the Jaccard similarity between two sets.
- average_jaccard_index: Computes the average Jaccard index across multiple sets of names.
"""

import numpy as np

def compute_jaccard_index(set1, set2):
    """Compute Jaccard similarity index between two sets."""
    if set1 == set2: 
        return 1
    set1, set2 = set(set1), set(set2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def average_jaccard_index(df, variable):
    """
    Compute the average Jaccard index for sets of names across different answers.
    
    This function extracts names from each parsed answer, computes Jaccard similaritys
    between all pairs of name sets, and returns the average similarity.
    """
    parsed_columns = [col for col in df.columns if "Parsed_answer_" in col]

    names_lists = []
    for col in parsed_columns:
        sub_dataframe = df.loc[variable, col]
        try: 
            names = sub_dataframe['Names'].str.lower()
            names_lists.append(set(names))
        except KeyError:
            names_lists.append(set())
            
    n = len(names_lists)

    jaccard_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            jaccard_indices.append(compute_jaccard_index(names_lists[i], names_lists[j]))

    return (np.mean(jaccard_indices) if jaccard_indices else 0, np.std(jaccard_indices) if jaccard_indices else 0)