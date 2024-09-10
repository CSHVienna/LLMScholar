# LLMScholar: Assessing LLM Outputs in Scholarly Contexts

## Introduction

LLMScholar is a pipeline designed to assess and analyze the outputs of LLaMa 3 70B in scholarly contexts. This project focuses on evaluating the factuality, consistency, and potential biases in LLM-generated responses related to scientific authors and their works.

## Project Structure

The project is organized into several key components:

1. **Data Processing**
   - `data_loader.py`: Loads LLM output data from Excel files.
   - `data_parser.py`: Parses and structures LLM outputs.

2. **Analysis**
   - `fact_checker.py`: Validates LLM outputs against ground truth data.
   - `consistency_checks.py`: Evaluates the consistency of LLM outputs across multiple runs.
   - `score_computation.py`: Computes various factuality and consistency scores.

3. **Visualization**
   - `plotting_utils.py`: Contains functions for data visualization and analysis.
   - `results_visualisation.ipynb`: Jupyter notebook showcasing the main results and graphs.

4. **Configuration and Utilities**
   - `config.py`: Contains project-wide constants and configuration variables.
   - `utils.py`: Provides utility functions used throughout the project.

5. **LLM Interaction**
   - `llm_parser.py`: Handles parsing of problematic LLM outputs using the Groq API.

## Key Features

- Factuality assessment of LLM outputs against a curated dataset of scientific authors and publications.
- Consistency analysis of LLM responses across multiple runs.
- Gender distribution analysis in LLM-generated author lists.
- Visualization of author similarities using dimensionality reduction techniques.

## Results

The main results and visualizations can be found in the `results_visualisation.ipynb` notebook. This includes:
- Factuality scores across different categories of queries
- Gender distribution in LLM-generated author lists
- Author similarity visualizations

## Ongoing Development

Please note that this pipeline is actively being developed, and more experiments are being carried out. The structure and functionality may change in the upcoming days as we refine our methodologies and expand our analyses.
