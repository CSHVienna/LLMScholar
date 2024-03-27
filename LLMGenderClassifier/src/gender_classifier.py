from groq import Groq
import os
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable before running the project. More in README.md.")
client = Groq(
    api_key=groq_api_key,
)


instruction = """I am a helpful bot specialised in gender classification. 
Given a text about someone, I always infer if the subject of the text is male or female from the pronouns and possessive determiners used.
I speak only dictionary language, therefore everyoutput I give no matter what would be of the form: 
{{\"gender\": \"[one word among 'female', 'male', or 'unkown']\", \"motivation\": \"[your motivation]\"}} 
I cannot generate any introduction or additional text beside the dictionary as output."""

def gender_classifier_template(web_search, name, affiliations, timestamp):
    prompt = f"""Given information retrieved from the web about the scientist "{name}", your task is infer their gender from the text.

    When analyzing web search results, if multiple individuals are identified, determine the gender specifically for the individual who most closely aligns with the provided details:  
    name: {name}, affiliations: {affiliations}, year of activity: {timestamp[:4]} 
     
    Verify that the individual for whom you are inferring gender corresponds with the details supplied. 
    Upon confirming the match, structure the relevant information as follows, utilizing a dictionary format:
    {{\"gender\": \"[one word among 'female', 'male', or 'unkown']\", \"motivation\": \"[your motivation]\"}} 

    web search result: {web_search}
    """
    return prompt

def gender_classifier(row, web_search_result, temperature = 0.25):

    content_prompt = gender_classifier_template(web_search_result, row['name'], row['affiliations'], row['timestamp'])

    prompt = [
                {
            "role": "system",
            "content": instruction
        }, 
        {
            "role": "user",
            "content": content_prompt
        }
    ]


    chat_completion = client.chat.completions.create(
    messages= prompt,
    model= 'mixtral-8x7b-32768',
    temperature=0.75, 
    )

    print(chat_completion.choices[0].message.content)

    return prompt, chat_completion.choices[0].message.content