from semantic_similarity import semantic_similarity_score
from llm_call import llm_data
from author_check import llm_name_retrieval, names_in_aps
import pandas as pd


def main():


    my_folder_path = '/Users/danielebarolo/Desktop/Unipd/4th Semester (Intership)/Code/APS/Data' #insert own path
    author_names_path = my_folder_path + "/author_names.csv"
    author_names_df = pd.read_csv(author_names_path)

    variables = { "top-25" : "Top 25 scientists", 
                 "timeframe" : "Scientists that have published between 1990 and 2000", 
                 "field" : "Scientists that have published in the field of Network Science", 
                 "stat_twin" : "Possible statistical twins of Albert-László Barabási"
    }

    df = pd.DataFrame(columns=['prompt', 'answer_1', 'names_in_aps_1', 'answer_2', 'names_in_aps_2', 'answer_3', 'names_in_aps_3', 
                               'answer_4', 'names_in_aps_4', 'semantic_similarity_score'])
    
    df.to_csv("./data/try_llm_responses.csv", index=False)

    for key, value in variables.items():

        times = 4 # Number of times the model will be called
        print(f"#######################{key}#########################")

        for i in range(times):
            print (f"#######################answer n: {i+1}#########################")
            answer = llm_data(value)
            print(answer)
            df.at[key,f'answer_{i+1}'] = answer[1]
            prompt = answer[0]
            names_suggested = llm_name_retrieval(answer[1])
            names_check = names_in_aps(author_names_df, names_suggested)
            print(names_check)
            df.at[key,f'names_in_aps_{i+1}'] = names_check

        df.at[key,"semantic_similarity_score"] = semantic_similarity_score(df.loc[key, ['answer_1', 'answer_2', 'answer_3', 'answer_4']])
        df.at[key,'prompt'] = prompt


    df.index = variables.keys()
    df.to_csv("./data/llm_responses.csv", index=False)


if __name__ == "__main__":
    main()