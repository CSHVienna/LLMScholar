from groq import Groq
import os
import json

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable before running the project. More in README.md.")
client = Groq(
    api_key=groq_api_key,
)



instruction = '''You are a helpful bot specialised in JSON corrections. 
You can only output in dictionary format. You are not allowed to change the content of the text, only the format.
Your output must be in the form of a dictionary, therefore always starting and ending with {{}}. Never provide any explanations of what you have done.'''

def llm_json_correction_template(error, text):
    prompt = f'''You will be given a string that represents a JSON object, and an error raised by Python's json.loads() function. 
Identify and correct the issues such as improperly double-quoted keys or values, missing quotation marks, or incorrect comma placements. 
Ensure all keys and values are enclosed in double quotes, and that arrays and objects have proper syntax. Provide the corrected JSON string as output.
                input text: {text} | error: {error}.
Output only the corrected JSON string, starting and finishing with {{}}. AVoid any explanation.'''
    return prompt

def llm_json_correction(error, text):

    content_prompt = llm_json_correction_template(error, text)

    chat_completion = client.chat.completions.create(
    messages= [{
            "role": "system",
            "content": instruction
        }, {
            "role": "user",
            "content": content_prompt
        }
    ],
    model= 'mixtral-8x7b-32768',
    temperature=0.2, 
    )
    return chat_completion.choices[0].message.content

def extract_braced_content(text):
    return text[text.find("{"):text.rfind("}") + 1]

def json_processor(answer):
        answer = extract_braced_content(answer)
        try:
            json_answer = json.loads(answer) 
            return (json_answer['gender'], json_answer['motivation'])
           
        except Exception as e:
            print(f"An error occurred: {e}")
            answer_corrected = llm_json_correction(e, answer)
            print(answer_corrected)

            try:
                json_answer = json.loads(answer_corrected)
                return (json_answer['gender'], json_answer['motivation'])
            
            except Exception as e:
                print(f"An error occurred: {e}")
                return ("JSON_error", answer)
