import os
import pandas as pd
from datetime import datetime
from llmscholar.data.loader import load_llm_data
from llmscholar.data.parser import parse_data
from llmscholar.analysis.fact_checker import enhance_data
from llmscholar.config import APS_DATA_PATH, LLM_DATA_FOLDER, VARIABLE_CATEGORIES

def main():
    # Load LLM data
    llm_data_path = os.path.join(LLM_DATA_FOLDER, "LLM_data.xlsx")
    llm_df = load_llm_data(llm_data_path)
    
    # Parse data
    parsed_df = parse_data(llm_df, format_data='JSON')
    
    # Load necessary datasets for enhancement
    dataset_df = pd.read_csv(os.path.join(APS_DATA_PATH, "names_factuality.csv"))
    dois_dataset = pd.read_csv(os.path.join(APS_DATA_PATH, "dois_factuality.csv"))
    authors_twins_metrics = pd.read_csv(os.path.join(APS_DATA_PATH, "authors_twins_metrics.csv"))
    
    # Enhance data for all variable categories
    enhanced_df = pd.DataFrame()
    for category in ['top-k', 'epoch', 'field', 'twin']:
        for variable in VARIABLE_CATEGORIES[category]:
            print(f"Enhancing data for {variable}")
            enhanced_result = enhance_data(parsed_df.loc[variable], variable, dataset_df, dois_dataset, authors_twins_metrics)
            enhanced_df = pd.concat([enhanced_df, enhanced_result])
    
    # Save the enhanced dataframe
    current_date = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", current_date)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"enhanced_df_{current_date}.csv")
    enhanced_df.to_csv(output_file)
    print(f"Enhanced dataframe saved as '{output_file}'")

if __name__ == "__main__":
    main()