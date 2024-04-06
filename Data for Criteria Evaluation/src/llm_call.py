from groq import Groq
import os
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable before running the project. More in README.md.")
client = Groq(
    api_key=groq_api_key,
)

def straighforward_template(variable):
    prompt = f"""Provide a list of scientists who have published in the American Physics Society during the specified time in JSON format. 
    The list must adhere to the following criterion: {variable}."""
    return prompt

def llm_data(variable, temperature = 0.75):

    prompt = [
        {
            "role": "user",
            "content": straighforward_template(variable)
        }
    ]

    chat_completion = client.chat.completions.create(
    messages= prompt,
    model= 'mixtral-8x7b-32768',
    temperature=temperature, 
    )

    return prompt, chat_completion.choices[0].message.content
