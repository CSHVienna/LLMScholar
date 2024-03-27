# main.py
from web_search import web_search
from gender_classifier import gender_classifier
from url_extraction import url_extraction
from json_corrector import json_processor
import pandas as pd
import json
import time

def main():

    start_time = time.time() 

    data_path = "/Users/danielebarolo/LLMScholar/LLMGenderClassifier"

    df = pd.read_csv(data_path + "/data/test_df.csv")

    df[['prompt_web_search', 'web_search', 'url', 'prompt_gender', 'gender', 'gender_motivation']] = ''
    
    for index, row in df.iterrows():
        
        #Conduct a web search on the scientist's name
        prompt_coral, response_coral = web_search(row)
        web_search_result = response_coral.text

        df.at[index,'prompt_web_search'] = prompt_coral
        df.at[index,'web_search'] = web_search_result
        df.at[index,'url'] = url_extraction(web_search_result)

        #Infer the gender according to the web search result
        prompt_mixtral, gender_answer = gender_classifier(row, web_search_result, temperature = 0.25)
        df.at[index,'prompt_gender'] = prompt_mixtral

        extract_info = json_processor(gender_answer)
        df.at[index, 'gender'] = extract_info[0]
        df.at[index, 'gender_motivation'] = extract_info[1]

    
    df.to_csv(data_path + "/data/enriched_test_df.csv", index=False)

    end_time = time.time()  

    print ("#################### FINISHED ####################")    
    print(f"Execution TIME: {((end_time - start_time)/60):.2f} minutes") 
    print ("##################################################")

if __name__ == "__main__":
    main()
