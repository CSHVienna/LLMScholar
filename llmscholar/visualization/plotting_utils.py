
"""
Plotting Utilities Module for the LLMScholar project.

This module provides various functions for data visualization and analysis of LLM outputs.
It includes functions for plotting factuality scores, gender distributions, and author similarities.

Key components:
- Factuality Results: Functions to compute and plot factuality scores
- Gender Results: Functions to analyze and plot gender distributions
- Twins Similarity: Functions to prepare and plot author similarity data

Note: This module relies heavily on matplotlib and seaborn for plotting.
"""

import pandas as pd
import numpy as np
import ast
import json
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from config import *

### FACTUALITY RESULTS ###

aps_dataset = pd.read_csv(APS_DATA_PATH + "names_factuality.csv")
dois_dataset = pd.read_csv(APS_DATA_PATH + "dois_factuality.csv")

def compute_expert_threshold(dataset_df):
    total_scientists = len(dataset_df)
    top_percentage = 0.001
    top_n = max(1, int(total_scientists * top_percentage))
    sorted_df = dataset_df.sort_values('rank')
    return sorted_df.iloc[top_n - 1]['rank']

expert_threshold = compute_expert_threshold(aps_dataset)

def dict_to_dataframe(s):
    if isinstance(s, str):
        try:
            s = s.replace(': nan,', ': None,').replace(': nan}', ': None}')
            data = ast.literal_eval(s)
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
                return df.replace({None: np.nan})
        except:
            pass
    return s

def load_nested_dataframe(filepath):
    df = pd.read_csv(filepath, index_col=0)
    nested_cols = [col for col in df.columns if col.startswith(('Parsed_answer_', 'Enhanced_answer_'))]
    for col in nested_cols:
        df[col] = df[col].apply(dict_to_dataframe)
    return df

def load_llm_data(filepath = LLM_DATA_FOLDER): # This is to do not display the actual name of scientits, for privacy reasons
    nested_df = load_nested_dataframe(filepath)
    twin_mapping = {
        -4: 'famous male',
        -3: 'random female',
        -2: 'random male',
        -1: 'famous female', 
    }
    twin_entries = nested_df.iloc[-4:]
    non_twin_entries = nested_df.iloc[:-4]
    twin_entries.index = twin_mapping.values()
    reordered_df = pd.concat([non_twin_entries, twin_entries])
    return reordered_df


def set_style(font_scale=2):
    sns.set_style("whitegrid")
    sns.set_context("poster", font_scale=font_scale)
    plt.rcParams['font.family'] = 'serif'

def set_latex():
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def compute_factuality_scores(nested_df, variable, expert_threshold, dois_dataset):
    results = {'scores': [], 'consistency': None, 'consistency_format': None}
    consistency_authors = []

    for i in range(5):
        df = nested_df.loc[variable, f'Enhanced_answer_{i}']
        
        # Factuality Author
        author_score = (df['Status'] == 'present').mean()
        
        # Factuality Experts
        expert_score = (df['Rank'] <= expert_threshold).mean() if 'Rank' in df.columns else np.nan
        
        # Factuality Condition
        if variable.startswith('field'):
            condition_scores = []
            for _, row in df.iterrows():
                if pd.notna(row.get('DOI')) and pd.notna(row.get('APS_ID')):
                    doi_match = dois_dataset[dois_dataset['doi'] == row['DOI']]
                    if not doi_match.empty:
                        authors_list = ast.literal_eval(doi_match['authors_oa_list'].iloc[0])
                        if str(row['APS_ID']) in authors_list:
                            condition_scores.append(1)
                        else:
                            condition_scores.append(0)
                    else:
                        condition_scores.append(0)
                else:
                    condition_scores.append(0)
            condition_score = np.mean(condition_scores) if condition_scores else 0
        elif variable.startswith('epoch'):
            condition_score = df['Overlap'].mean() if 'Overlap' in df.columns else np.nan
        elif variable.startswith('top-'):
            k = int(variable.split('-')[1])
            condition_score = int(len(df) == k)
        else:
            if 'Cosine_Similarity' in df.columns:
                # Calculate the proportion of cosine similarities above 0.8
                condition_score = (df['Cosine_Similarity'] > 0.8).mean()
            else:
                condition_score = np.nan
        
        results['scores'].append([author_score, expert_score, condition_score])
        consistency_authors.append(set(df['Names']))

    results['scores'] = np.array(results['scores'])
    
    # Consistency Authors
    jaccard_scores = [jaccard_similarity(a, b) for a, b in combinations(consistency_authors, 2)]
    results['consistency'] = (np.mean(jaccard_scores), np.std(jaccard_scores))
    
    # Consistency Format
    results['consistency_format'] = nested_df.loc[variable, 'Parser Result']
    
    return results


def plot_factuality_scores(results, category):
    plt.close('all')
    
    criteria = ['Factuality Author', 'Factuality Experts', 'Factuality Condition', 'Consistency Authors', 'Consistency Format']
    if category == 'twin':
        criteria[2] = 'Factuality Condition (Cosine)'
    
    x = np.arange(len(criteria))
    width = 0.35 if category != 'twin' else 0.2
    
    set_style(font_scale=2)
    
    fig, ax = plt.subplots(figsize=(30, 20) if category == 'twin' else (20, 10))
    ax.grid(False)
    
    colors = plt.colormaps['tab20c'](np.linspace(0, 1, 20))
    
    variables = VARIABLE_CATEGORIES[category]
    
    for i, variable in enumerate(variables):
        means = []
        stds = []
        for j, criterion in enumerate(criteria):
            if criterion in ['Factuality Author', 'Factuality Experts', 'Factuality Condition', 'Factuality Condition (Cosine)']:
                mean = results[variable]['scores'][:, j].mean()
                std = results[variable]['scores'][:, j].std()
            elif criterion == 'Consistency Authors':
                mean = results[variable]['consistency'][0]
                std = results[variable]['consistency'][1]
            else:  # Consistency Format
                mean = results[variable]['consistency_format']
                std = 0
            
            means.append(mean)
            stds.append(std)
        
        ax.bar(x + i * width, means, width, yerr=stds, capsize=5, 
               color=colors[i * 4], alpha=0.8,
               label=variable)
    
    ax.set_ylabel('Score', fontsize=24)
    
    sns.despine(ax=ax)
    
    ax.set_xticks(x + width * (len(variables) - 1) / 2)
    ax.set_xticklabels(criteria, rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0, 1)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)
    
    plt.title("")
    plt.tight_layout()
    plt.show()

def plot_results(category, nested_df, expert_threshold, dois_dataset):
    set_style()
    set_latex()

    if category not in VARIABLE_CATEGORIES:
        raise ValueError(f"Invalid category. Choose from: {list(VARIABLE_CATEGORIES.keys())}")

    all_results = {}
    for variable in VARIABLE_CATEGORIES[category]:
        all_results[variable] = compute_factuality_scores(nested_df, variable, expert_threshold, dois_dataset)

    
    if category == 'twin':
        temp = all_results['random male']
        all_results['random male'] = all_results['random female']
        all_results['random female'] = temp

    plot_factuality_scores(all_results, category)

### GENDER RESULTS ###

#load hand coded demographics
class PhysicistDemographics:
    def __init__(self, filename=GENDER_HANDCODED):
        self.filename = filename
        self.data = self.load()

    def load(self):
        with open(self.filename, 'r') as f:
            return json.load(f)

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get(self, name):
        return self.data.get(name, None)

    def add(self, name, info):
        self.data[name] = info
        self.save()

def plot_gender_distribution(results, custom_order):
    plt.rcParams.update({'font.size': 16})

    variables = custom_order
    
    def custom_label(variable):
        if 'famous male' in variable.lower():
            return 'Famous Male'
        elif 'famous female' in variable.lower():
            return 'Famous Female'
        elif 'random male' in variable.lower():
            return 'Random Male'
        elif 'random female' in variable.lower():
            return 'Random Female'
        else:
            return variable

    custom_labels = [custom_label(var) for var in variables]

    x = np.arange(len(variables))
    width = 0.25

    sns.set_palette('pastel')
    
    fig, ax = plt.subplots(figsize=(22, 10))  # Increased width to accommodate legend
    
    colors = {'Male': sns.color_palette()[0], 'Female': sns.color_palette()[2], 
              'Not in APS': sns.color_palette()[3]}
    labels = {'Male': 'Male', 'Female': 'Female', 'Not in APS': 'Not in APS'}

    categories = ['Not in APS', 'Male', 'Female']

    for i, gender in enumerate(categories):
        means = [results[var]['mean'][gender] for var in variables]
        stds = [results[var]['std'][gender] for var in variables]

        ax.bar(x + i*width, means, width, label=labels[gender], color=colors[gender], yerr=stds, capsize=5)

    # Calculate and plot the mean of all answers
    mean_of_means = {gender: np.mean([results[var]['mean'][gender] for var in variables]) for gender in categories}
    std_of_means = {gender: np.std([results[var]['mean'][gender] for var in variables]) for gender in categories}
    
    for i, gender in enumerate(categories):
        ax.bar(len(variables) + i*width, mean_of_means[gender], width, 
               color=colors[gender], alpha=0.5, yerr=std_of_means[gender], capsize=5)

    ax.set_ylabel('Percentage', fontsize=24, fontweight='bold')
    ax.set_xticks(np.concatenate([x + width, [len(variables) + width]]))
    ax.set_xticklabels(custom_labels + ['Mean'], rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Remove grid
    ax.grid(False)
    
    # Move legend outside the plot
    ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

def analyze_variable(nested_df, variable, demographics):
    gender_counts = []
    total_names = []
    
    for i in range(5):  # 5 iterations
        df = nested_df.loc[variable, f'Enhanced_answer_{i}']
        
        gender_count = {'Male': 0, 'Female': 0, 'Not in APS': 0}
        total_count = 0
        
        for _, row in df.iterrows():
            total_count += 1
            
            if row.get('Status') == 'present':
                name = row.get('Names')
                if name:
                    info = demographics.get(name)
                    if info and 'gender' in info:
                        gender = info['gender']
                        gender_count[gender] += 1
                    else:
                        gender_count['Not in APS'] += 1
                else:
                    gender_count['Not in APS'] += 1
            else:
                gender_count['Not in APS'] += 1
        
        # Convert counts to percentages
        gender_percentages = {k: v / total_count for k, v in gender_count.items()}
        gender_counts.append(pd.Series(gender_percentages))
        
        total_names.append(total_count)
    
    # Calculate mean and std of percentages
    mean_percentages = pd.concat(gender_counts, axis=1).mean(axis=1)
    std_percentages = pd.concat(gender_counts, axis=1).std(axis=1)
    
    # Calculate mean and std of total names
    mean_total_names = np.mean(total_names)
    std_total_names = np.std(total_names)
    
    return mean_percentages, std_percentages, mean_total_names, std_total_names


def gender_results(nested_df, demographics):
    results = {}
    
    custom_order = [
        'top-5', 'top-100', 'epoch 1950s', 'epoch 2000s', 'field CM&MP', 'field PER',
        'famous male', 'famous female', 'random male', 'random female'
    ]
    
    for variable in custom_order:
        mean_percentages, std_percentages, mean_total_names, std_total_names = analyze_variable(nested_df, variable, demographics)
        results[variable] = {
            'mean': mean_percentages,
            'std': std_percentages,
            'total_names': mean_total_names,
            'total_names_std': std_total_names
        }
    
    plot_gender_distribution(results, custom_order)


### TWINS SIMILARITY ###

data = pd.read_csv(FEATURE_VECTOR_PATH)

def prepare_analysis_data(llm_df, twin_name):
    TWIN_IDS = {
        'famous male': {'APS_ID': 3984855223, 'OA_ID': 'A5038976962'},
        'famous female': {'APS_ID': 25364, 'OA_ID': 'A5012905268'},
        'random female': {'APS_ID': 66635, 'OA_ID': 'A5005478445'},
        'random male': {'APS_ID': 143998, 'OA_ID': 'A5023511813'}
    }

    all_data = []
    for i in range(5):  
        df = llm_df.loc[twin_name, f'Enhanced_answer_{i}']
        for _, row in df.iterrows():
            if row['Status'] == 'present' and pd.notna(row['OA_ID']):
                all_data.append({
                    'Name': row['Names'],
                    'oa_id': row['OA_ID']
                })
    analysis_data = pd.DataFrame(all_data).drop_duplicates().reset_index(drop=True)

    # Add twin at the top
    twin_data = pd.DataFrame({
        'Name': [twin_name],
        'oa_id': [TWIN_IDS[twin_name]['OA_ID']]
    })
    analysis_data = pd.concat([twin_data, analysis_data]).reset_index(drop=True)

    return analysis_data

def plot_embeddings(data, analysis_data, twin_name, sample_size=10000, random_state=42, title = False):
    set_style(font_scale=1.5)
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['oa_id', 'APS_ID']]

    data_clean = data.dropna(subset=numeric_cols)

    authors_of_interest = analysis_data['oa_id'].tolist()
    sample_size = sample_size - len(authors_of_interest)
    data_sample = pd.concat([
        data_clean[data_clean['oa_id'].isin(authors_of_interest)],
        data_clean[~data_clean['oa_id'].isin(authors_of_interest)].sample(n=min(sample_size, len(data_clean)), random_state=random_state)
    ])

    data_sample = data_sample.merge(analysis_data[['oa_id', 'Name']], on='oa_id', how='left')

    X = data_sample[numeric_cols].values

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    tsne_norm = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne_norm = tsne_norm.fit_transform(X_normalized)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_normalized)

    mask_interest = data_sample['oa_id'].isin(authors_of_interest)
    first_author_oa_id = analysis_data['oa_id'].iloc[0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    for ax, X_plot, title in zip([ax1, ax2, ax3], 
                                 [X_tsne, X_tsne_norm, X_pca], 
                                 ['t-SNE (without normalization)', 't-SNE (with normalization)', 'PCA']):
        ax.scatter(X_plot[~mask_interest, 0], X_plot[~mask_interest, 1], color='grey', alpha=0.5, s=1)
        ax.scatter(X_plot[mask_interest, 0], X_plot[mask_interest, 1], color='blue', s=50)
        
        first_author_index = data_sample[data_sample['oa_id'] == first_author_oa_id].index[0]
        ax.scatter(X_plot[first_author_index, 0], X_plot[first_author_index, 1], color='red', s=100)
        
        ax.annotate('reference', (X_plot[first_author_index, 0], X_plot[first_author_index, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12, color='red', fontweight='bold')
        
        for i, (x, y, oa_id) in enumerate(zip(X_plot[mask_interest, 0], 
                                              X_plot[mask_interest, 1], 
                                              data_sample[mask_interest]['oa_id'])):
            if oa_id != first_author_oa_id:
                ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                            fontsize=10, color='blue', fontweight='bold')
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Dimension 1' if 'SNE' in title else 'Principal Component 1', fontsize=14)
        ax.set_ylabel('Dimension 2' if 'SNE' in title else 'Principal Component 2', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    if title:
        plt.suptitle(f'{twin_name}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{twin_name}_embeddings.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

def plot_authors_similarity(llm_df, twin_name, data, title = False):
    
    analysis_data = prepare_analysis_data(llm_df, twin_name)
    plot_embeddings(data, analysis_data, twin_name, title = title)



