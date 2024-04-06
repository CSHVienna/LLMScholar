from groq import Groq
import os
import json
import Levenshtein

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable before running the project. More in README.md.")
client = Groq(
    api_key=groq_api_key,
)

instruction = '''You are a helpful bot specialised in JSON formats. 
You can only output in dictionary format. You are not allowed to change the content of the text, only the format.
Your output must be in the form of a dictionary, therefore always starting and ending with {{}}. Never provide any explanations of what you have done.'''

def name_retrieval_template(text):
    prompt = f'''You will be given a response from a Large Language Model. 
    Your task is to identify if there are any given names inside the text. 
    If the model provides general examples or hypotethical names, which are not actual names, you have to recognise it and output an empty dictionary. 
    You have to output a dictionary of this form: {{"names": [all name retrieved OR empty list]}}. 
    Provide the corrected JSON string as output.
                input text: {text} 
    Avoid any explanation.'''
    return prompt

def llm_name_retrieval(text):

    content_prompt = name_retrieval_template(text)

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
    print(chat_completion.choices[0].message.content)
    answer = chat_completion.choices[0].message.content
    json_answer = json_loader(answer)
    print(json_answer)
    return json_answer


def json_loader(text):
    max_attempts = 4  # Prevents infinite loops in case of unexpected input
    attempt = 0
    json_str = text[text.find("{"):text.rfind("}") + 1]

    while attempt < max_attempts:
        try: 
            json_answer = json.loads(json_str)
            return json_answer
        except json.JSONDecodeError:
            json_str = json_str[1:-1]
            print(json_str)
        attempt += 1
    
    return "Unable to fix the JSON data. Please check its format."

def lev_distance(name, df):
    names = df['name']

    max_lev = 0
    most_similar_name = None
    for n in names:
        lev_distance = Levenshtein.ratio(name, n)
        if lev_distance > max_lev:
            max_lev = lev_distance
            most_similar_name = n
    return most_similar_name

def names_in_aps(author_names, retrieved_names_dict):
    
    if retrieved_names_dict == "Unable to fix the JSON data. Please check its format.":
        return "Unable to fix the JSON data. Please check its format."
    
    names_to_check = retrieved_names_dict['names']

    if len(names_to_check) == 0: 
        return "No names found"

    else: 
        result_dict = {}
        for name in names_to_check:
            print(name)
            if name in author_names['name'].values:
                result_dict[name] = "full name PRESENT"
            else:
                most_similar_name = lev_distance(name, author_names)
                result_dict[name] = f"most similar name in APS: {most_similar_name}"

        return result_dict

