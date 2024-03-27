import cohere
import os
import time
cohere_api_key = os.getenv('COHERE_API_KEY')
if not cohere_api_key:
    raise ValueError("Please set the COHERE_API_KEY environment variable before running the project. More in README.md.")
co = cohere.Client(cohere_api_key)

def web_search_template(name):

    prompt = f"""Conduct a comprehensive web search for scientists named {name}. 

  Focus on identifying people with credentials or titles suggesting a scientific background, including but not limited to: scientist, researcher, professor, doctor (in a scientific field). 

  For each identified individual, determine their primary field of study or work. Check if the field falls within STEM (Science, Technology, Engineering, and Mathematics). Ignore profiles unrelated to science.

  Rank the list based on the relevance and significance of the contributions within STEM fields. 

  Present the FIRST result ONLY in a bulleted list format, with each bullet point containing:
      * Name
      * Primary field of study/work (if applicable)
      * Detailed overview of the scientist. Focus on key areas including their education, major contributions, research interests, and any significant awards or recognitions. Ensure the narrative is concise and factual.
      * Source URL for reference

  Ensure that the information is accurate, up-to-date, and clearly presented. 
  """

    return prompt

def web_search(row, max_retries=5, wait_seconds=5,  temperature = 0.75,):
    name = row['name']
    prompt = web_search_template(name)
    success = False

    for attempt in range(max_retries):

        try:    
            response = co.chat(
        model='command-r',
        message=prompt,
        connectors=[{"id": "web-search"}],
        temperature=temperature
    )
            print(response.text)
            return prompt, response
        
        except Exception as e:
            print(f"{e} \n retrying... (Attempt {attempt + 1}/{max_retries})")
            if attempt + 1 < max_retries:
                time.sleep(wait_seconds)
            else:
                print("Maximum retries reached, moving on.")
                return prompt, "Failed to retrieve the data."
