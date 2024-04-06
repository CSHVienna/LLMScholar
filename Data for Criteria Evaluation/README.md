# Project Description

## Overview
This project aims to collect essential data for designing criteria to audit Language Model (LLM) responses effectively. By leveraging structured calls to LLMs, we gather targeted insights across four variables crucial for auditing methodologies. Our focus is on understanding the behavior of the suggested authors under various contexts and constraints imposed by the prompts.

## Objectives
### Variable Analysis
Our audit focuses on examining LLM behavior concerning:
- Top-k authors
- Different time frames
- Varied fields of research
- Comparisons with statistical twins

### Data Collection
Utilizing the Groq API and the open-source Mixtral8x7B model, we execute structured prompts to observe and record the LLM's responses. Each variable modifies a single aspect of a consistent prompt template, allowing us to isolate and analyze the impact of that variable.

### Data Extraction and Validation
Post-collection, we extract suggested author names from the LLM's responses. These names are then cross-referenced with the APS dataset to ascertain matches or identify the closest equivalents.

### Semantic Similarity Analysis
We employ a RoBERTa-based Semantic Similarity score to quantify the alignment between responses, providing a metric to gauge consistency and deviation across different prompts.

### Data Aggregation
The final output comprises a comprehensive .csv file encapsulating the prompts, responses, name verifications, and similarity scores. This dataset serves as a foundational element for subsequent analysis and criteria development for LLM auditing.

## Application
The collated data is integral for the notebook environment used in feedback collection, a critical phase in designing nuanced, effective audit criteria. This project lays the groundwork for establishing rigorous, evidence-based standards for auditing LLMs, ensuring their reliability and integrity in diverse research contexts.

---

### Configuration

Before running this project, you need to set up an environment variable for your API key. The keys are free (with rate-time limits) and essential for accessing and utilizing the open source model on Groq's servers ([Groq Models](https://console.groq.com/docs/models)) or Cohere's models ([Cohere Models](https://docs.cohere.com/reference/chat)). To configure your environment:

- For Unix/macOS: `export {GROQ/COHERE}_API_KEY='your-api-key-here'`
- For Windows: `setx {GROQ/COHERE}_API_KEY "your-api-key-here"`

Please replace `your-api-key-here` with your actual API key. Choose either `GROQ` or `COHERE` based on which API you are utilizing. Ensure to restart your terminal session after setting the environment variable to apply the changes.

### Development Environment Setup

1. **Install Python & VS Code**: Ensure Python is installed, and install Visual Studio Code along with the Python extension.
2. **Open the Project**: Open the project folder in VS Code.
3. **Environment & Dependencies**: Use the terminal in VS Code to create (`python -m venv venv`) and activate the virtual environment (`source venv/bin/activate` on Unix/macOS, `.\venv\Scripts\activate` on Windows). Then install dependencies (`pip install -r requirements.txt`).
4. **Select Interpreter**: In VS Code, choose the Python interpreter from the virtual environment.
