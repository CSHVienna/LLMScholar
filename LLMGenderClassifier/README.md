# Athors' Gender Inference using LLM with Web Search RAG

## Project Overview

This project aims to infer the gender of scientists using a novel approach that combines two Large Language Model (LLM) technologies, on of which with Web Search Retrieve-Augmented-Generatation (RAG). The process integrates Command-r from Cohere to provide accurate and context-aware scientists's information and the open source model Mixtral8x7b for gender classification. 

## Configuration

Before running this project, you need to set up an environment variable for your API key. The keys are free (with rate-time limits) and essential for accessing and utilizing the open source model on Groq's servers (https://console.groq.com/docs/models) or Cohere's models (https://docs.cohere.com/reference/chat). To configure your environment:

- For Unix/macOS: `export {GROQ/COHERE}_API_KEY='your-api-key-here'`
- For Windows: `setx {GROQ/COHERE}_API_KEY "your-api-key-here"`

Please replace `your-api-key-here` with your actual API key. Choose either `GROQ` or `COHERE` based on which API you are utilizing. Ensure to restart your terminal session after setting the environment variable to apply the changes.

## Development Environment Setup

1. **Install Python & VS Code**: Ensure Python is installed, and install Visual Studio Code along with the Python extension.
2. **Open the Project**: Open the project folder in VS Code.
3. **Environment & Dependencies**: Use the terminal in VS Code to create (`python -m venv venv`) and activate the virtual environment (`source venv/bin/activate` on Unix/macOS, `.\venv\Scripts\activate` on Windows). Then install dependencies (`pip install -r requirements.txt`).
4. **Select Interpreter**: In VS Code, choose the Python interpreter from the virtual environment.

### How It Works

1. **Command-r (Cohere):** This LLM is designed to automatically extract queries from the given prompts, perform web searches, and rank the results according to relevance. It then compiles the retrieved quotes to generate a comprehensive RAG answer. Specifically, for this project, Command-r seeks an overview of the most prominent scientists sharing the queried name, along with the URLs from which the data was extracted. This information forms the basis for subsequent analysis.

2. **Mixtral8x7b:** This secondary LLM steps in to infer the scientists' gender. The model provides an explanation for its gender determinations, enhancing the transparency and interpretability of its conclusions. When there is ambiguity, i.e., when the initial search results in multiple scientists with the same name, Mixtral8x7b focus on aligning the search results with pre-existing data on affiliations and years of activity to accurately select the scientist in question. 

### Data Requirements

To function effectively, the project requires data in a `.csv` format, located in a specified folder, whose filepath must be loaded in main().py. The CSV file must have headers that include at least the following fields:

- `name`: The scientist's name.
- `affiliations`: The scientist's affiliated institutions or organizations.
- `timestamp`: The timestamp of one of the scientist's publications.

### Output

The system processes the input data and enriches it with the following information, stored in a new `.csv` file:

- `prompt_web_search`: The prompt used for initiating the web search.
- `web_search`: The summary of web search results.
- `url`: The URL from which the information was retrieved.
- `prompt_gender`: The prompt used for gender inference.
- `gender`: The inferred gender of the scientist.
- `gender_motivation`: The explanation justifying the inferred gender.

### Objective

This project harnesses the power of advanced LLMs and web search capabilities to address the challenge of identifying scientists' genders from their names and available public data. By providing a structured and automated approach, it aims to facilitate gender-based analyses in scientific research and academia, supporting diversity and inclusion efforts.

For updates and further information, please refer to the project repository.
