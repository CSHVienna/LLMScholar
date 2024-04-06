# Save this as interactive_widget.py
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import ast
import re

class AdherenceEvaluation:
    def __init__(self, df, index, save_path=None, id=None):
        self.df = df
        self.index = index
        self.texts = [answer for answer in self.df.loc[self.index, [f"answer_{i}" for i in range(1, 5)]]]
        self.current_index = 0
        self.results = {'Adherence': []} # Store the adherence scores
        self.save_path = save_path # Path to save the results
        self.id = id
        self.setup_widgets()
        self.update_text_display()

    def setup_widgets(self):

        # Extract the prompt text using ast.literal_eval for safety
        prompt_text = ast.literal_eval(self.df.at[self.index, 'prompt'])[0]['content']
        # Creating the HTML display for the prompt
        prompt_display_html = f"""
        <div style="margin: 20px; padding: 10px; background-color: #f9f9f9; border-left: 6px solid #ccc;">
            <h2 style="color: #333;">Prompt:</h2>
            <p style="font-size: 18px; color: #555;">{prompt_text}</p>
        </div>
        """
        
        self.instruction_text = """<div style="margin: 20px; padding: 10px; background-color: #eef9f9; border-left: 6px solid #77cccc; overflow-wrap: break-word; word-wrap: break-word; overflow: hidden;">
    <h2 style="color: #333;">Task Instructions:</h2>
    <p>This criterion is essential for assessing the prompt's quality in terms of clarity. It determines whether the model accurately adheres to the given instructions, executing precisely what is requested by the prompt. </p>
    <p>For every answer:</p>
    <ul>
        <li>Rate the adherence to the instruction format, with 1 being not at all and 10 being fully compliant.</li>
        <li>After rating an answer, click the 'Confirm' button. You will then automatically proceed to evaluate the next answer.</li>
        <li>Continue this process until you have evaluated all four responses.</li>
    </ul>
    <h3 style="color: #333;">Output Expectations:</h3>
    <p>For sake of clarity, we give an example of what we mean by adherence. Since the prompt is designed to be straightforward, the expected ideal output should be: </p>
    <ul>
        <li>A simple JSON file format.</li>
        <li>Including the exact number of requested (possibly full) names.</li>
        <li>Excluding any unrelated content such as introductions or explanations.</li>
    </ul>
    <p>Here is an example of an ideal answer if 'top-10' was requested:</p>
    <pre style="white-space: pre-wrap; word-wrap: break-word;">[{"name": "Andrea McDowell"}, {"name": "Mark Levoy"}, {"name": "Kristen Grauman"}, {"name": "Gerald Jay Sussman"}, {"name": "Scott Aaronson"}, {"name": "Yuri Trevkunov"}, {"name": "Alexander M. Goncharov"}, {"name": "Oleg V. Prezhdo"}, {"name": "Efthimios Kaxiras"}, {"name": "Michael E. Flatté"}]</pre>
    <p><strong>Note:</strong> The relevance of the names or their recognition as 'top' scientists is not the primary focus here. Our aim is to assess the consistency of the output with the provided format and instruction adherence, not the accuracy or relevance of the names themselves.</p>
</div>"""

        self.instruction_text = widgets.HTML(value=self.instruction_text)
        self.prompt_display = widgets.HTML(value=prompt_display_html)
        
        # Initialize texts and other interactive components
        self.texts = [answer for answer in self.df.loc[self.index, [f"answer_{i}" for i in range(1, 5)]]]
        self.text_display = widgets.HTML(value="")
        self.title = widgets.HTML(value=f"<b>Answer 1:</b>")
        
        # Initialize slider and button with respective functionalities
        self.rating_instruction = widgets.HTML(value="<b>Does the output adhere to the instructions? Please provide a adherence score: </b>")
        adherence_label = widgets.HTML(value="<b>ADHERENCE:</b>")
        self.adherence_rating = widgets.IntSlider(value=5, min=1, max=10, step=1, layout={'width': '60%'})
        
        self.confirm_button = widgets.Button(description='Confirm and move to next answer', button_style='success', layout={'width': '20%'})
        self.confirm_button.on_click(self.on_confirm_clicked)

        # Horizontal box to contain the rating slider and button
        self.rating_line = widgets.HBox([adherence_label, self.adherence_rating, self.confirm_button])

    def update_text_display(self):
        if self.current_index < len(self.texts):
            self.text_display.value = f"<div style='height: 300px; overflow-y: auto;'>{self.texts[self.current_index]}</div>"
            self.title.value = f"<b>Answer {self.current_index + 1}:</b>"

    def on_confirm_clicked(self, b):
        # Handle confirm button click: update display or end evaluation
        if self.current_index < len(self.texts) - 2:
            self.results['Adherence'].append((f"Answer {self.current_index + 1}", self.adherence_rating.value))
            self.current_index += 1
            self.update_text_display()
        elif self.current_index == len(self.texts) - 2:
            self.results['Adherence'].append((f"Answer {self.current_index + 1}", self.adherence_rating.value))
            self.confirm_button.description = 'Confirm'
            self.current_index += 1
            self.update_text_display()
        else:
            self.results['Adherence'].append((f"Answer {self.current_index + 1}", self.adherence_rating.value))
            self.confirm_button.description = 'Completed ✓'
            self.confirm_button.disabled = True
            
            #store results in a csv file
            if self.save_path and self.id:
                results_df = pd.DataFrame(self.results)
                results_df.to_csv(self.save_path + f"/adherence/{self.id}_adherence_{self.index}.csv", index=False)
                print("Thanks; your evaluation has been saved!")

    def display(self):
        # Combine all widgets into the layout and display it
        layout = widgets.VBox([self.prompt_display, self.instruction_text, self.title, self.text_display, self.rating_instruction, self.rating_line])
        display(layout)
        self.update_text_display()

##############################################################################################################

class ConsistencyEvaluation:
    def __init__(self, df, index, save_path=None, id=None):
        self.df = df
        self.index = index
        self.texts = [answer for answer in self.df.loc[self.index, [f"answer_{i}" for i in range(1, 5)]]]
        self.titles = ["Answer 1", "Answer 2", "Answer 3", "Answer 4"]
        self.semscore = self.df.loc[self.index, 'semantic_similarity_score']
        self.current_index = 0
        self.results = {'Consistency': []} # Store the consistency scores
        self.save_path = save_path # Path to save the results
        self.id = id
        self.setup_widgets()
        self.update_text_display()

    def setup_widgets(self):
        self.instruction_text = widgets.HTML(f"""
        <div style="margin: 20px; padding: 10px; background-color: #eef9f9; border-left: 6px solid #77cccc;">
            <h2 style="color: #333;">Task Instructions:</h2>
            <p>Your task is to evaluate the similarity of the four provided answers. This criterion aims at assessing the clarity of the prompt, which shall yield consistent results across multiple executions.</p>
            <ul>
                <li>Review all four answers, focusing on their similarity to each other.</li>
                <li>Rate the consistency from one (not similar at all) to ten (extremely similar).</li>
                <li>Once you have decided on the score, click the <strong>'Confirm'</strong> button to lock in your rating.</li>
            </ul>
            <p>Note: We have also calculated a Semantic Similarity score between the answers using the RoBERTa model: <strong>Score: {float(self.semscore) * 10:.2f}</strong></p>
        </div>
        """)

        self.text_display = widgets.HTML(value="")
        self.title = widgets.HTML(value=f"<b>{self.titles[self.current_index]}:</b>")
        self.consistency_rating = widgets.IntSlider(value=5, min=1, max=10, step=1, description='CONSISTENCY:', style={'description_width': 'initial'}, layout={'width': '50%'})
        self.rating_instruction = widgets.HTML(value="<b>Are the outputs very similar to each other? Please provide a (unique) consistency score:</b>")
        self.confirm_button = widgets.Button(description='Confirm', button_style='success', layout={'width': '20%'})
        self.confirm_button.on_click(self.lock_slider)

        self.left_button = widgets.Button(description="--", disabled=True)
        self.right_button = widgets.Button(description="Answer 2")
        self.left_button.on_click(self.on_left_clicked)
        self.right_button.on_click(self.on_right_clicked)

        self.rating_line = widgets.HBox([self.consistency_rating, self.confirm_button])
        self.navigation_box = widgets.HBox([self.left_button, self.right_button])
        self.layout = widgets.VBox([self.instruction_text, self.title, self.text_display, self.navigation_box, self.rating_instruction, self.rating_line])
    
    def update_text_display(self):
        self.text_display.value = f"<div style='font-weight: normal; height: 400px; overflow-y: auto;'>{self.texts[self.current_index]}</div>"
        self.left_button.description = "<- " + self.titles[self.current_index - 1] if self.current_index > 0 else "--"
        self.right_button.description = self.titles[self.current_index + 1] + " ->" if self.current_index < len(self.texts) - 1 else "--"
        self.title.value = f"<b>{self.titles[self.current_index]}:</b>"
        self.left_button.disabled = self.current_index == 0
        self.right_button.disabled = self.current_index == len(self.texts) - 1

    def on_left_clicked(self, b):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_text_display()

    def on_right_clicked(self, b):
        if self.current_index < len(self.texts) - 1:
            self.current_index += 1
            self.update_text_display()

    def lock_slider(self, b):
        self.consistency_rating.disabled = True
        self.confirm_button.disabled = True
        self.confirm_button.description = 'Completed ✓'

        if self.save_path and self.id:
            self.results['Consistency'].append(self.consistency_rating.value)
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(self.save_path + f"/consistency/{self.id}_consistency_{self.index}.csv", index=False)
            print("Thanks; your evaluation has been saved!")
        # Logic to store the consistency score could be added here

    def display(self):
        display(self.layout)
        self.update_text_display()

##############################################################################################################

class FactualityEvaluation:
    def __init__(self, df, index, save_path=None, id=None):
        self.df = df
        self.index = index
        self.texts = [answer for answer in self.df.loc[self.index, [f"names_in_aps_{i}" for i in range(1, 5)]]]
        self.current_index = 0
        self.results = {'Factuality': []}
        self.save_path = save_path
        self.id = id
        self.graph_output = widgets.Output()
        self.df_output = widgets.Output()
        self.instruction_text = self.get_instruction_text()
        self.setup_widgets()
        self.update_visualization(self.current_index)

    def get_instruction_text(self):
        instruction_html = """
        <div style="margin: 20px; padding: 10px; background-color: #eef9f9; border-left: 6px solid #77cccc; overflow-wrap: break-word; word-wrap: break-word; overflow: hidden;">
            <h2 style="color: #333;">Task Instructions:</h2>
            <p>This criterion is to assess to what extent the model produces factually accurate names or engages in 'hallucination'—generating non-existent or incorrect data, particularly in the context of name generation. 
            This evaluation addresses the 'tail-end knowledge' problem, highlighting scenarios where the model is asked to provide specific data it may not have encountered frequently.</p>
            <ol>
                <li>Review the graph and the list of most similar names.</li>
                <li>Assign a factuality score from 1 (all names hallucinated) to 10 (all names are accurate).</li>
                <li>After scoring, click the 'Confirm and move to next answer' button to proceed to subsequent responses.</li>
            </ol>
            <p>Notes:</p>
            <ul>
                <li><strong>Handling Non-Name Outputs:</strong> If the model outputs general instructions or unrelated information instead of names, or simply says 'No names found', we appreciate your understanding and ask you to move to the next answer.</li>
                <li><strong>Fact-Checking Pipeline:</strong> A supporting system extracts the provided names, formatting them into JSON, and verifies their presence in the APS dataset. Names directly found are labeled as 'full name PRESENT'. If not directly found, a 'most similar name' is determined via Levenshtein distance for your evaluation.
                For names matched via similarity, you need to determine if they are sufficiently close to consider the factual integrity upheld. This judgment helps identify if minor spelling variations or abbreviations are present, rather than outright inaccuracies.</li>
            </ul>
        </div>
        """
        return widgets.HTML(value=instruction_html)

    def setup_widgets(self):
        self.title = widgets.HTML(value=f"<b>Answer {self.current_index + 1}:</b>")
        self.factuality_rating = widgets.IntSlider(value=5, min=1, max=10, step=1, description='FACTUALITY:',
                                                   style={'description_width': 'initial'}, layout={'width': '50%'})
        rating_instruction = widgets.HTML(value="<b>Does the output contain factual names? Please provide a factuality score:</b>")
        self.confirm_button = widgets.Button(description='Confirm and move to next answer', button_style='success', layout={'width': '20%'})
        self.confirm_button.on_click(self.on_confirm_clicked)
        self.layout = widgets.VBox([self.instruction_text, self.title, widgets.HBox([self.graph_output, self.df_output]), rating_instruction, self.factuality_rating, self.confirm_button])
        display(self.layout)

    def update_visualization(self, index):
        self.current_index = index
        answer_text = self.texts[self.current_index]

        # Define the instruction outside the condition to use it in both scenarios.
        rating_instruction = widgets.HTML(value="<b>Does the output contain factual names? Please provide a factuality score:</b>")
        
        if answer_text == "No names found":
            with self.graph_output:
                clear_output(wait=True)
                friendly_message = widgets.HTML(value="<div style='color: #31708f;'><strong>Note:</strong> No names were generated in this response. Please skip to the next answer.</div>")
                display(friendly_message)
            with self.df_output:
                clear_output(wait=True)

            # Adjust the layout to exclude the rating instruction when no names are found.
            self.confirm_button.description = 'Skip to next answer -->'
            self.layout.children = [self.instruction_text, self.title, self.graph_output, self.confirm_button]
        else:
            # Include the rating instruction when names are present.
            self.layout.children = [self.instruction_text, self.title, widgets.HBox([self.graph_output, self.df_output]), rating_instruction, self.factuality_rating, self.confirm_button]

            self.confirm_button.description = 'Confirm and move to next answer'
            self.factuality_rating.disabled = False
            self.factuality_rating.value = 5  # Reset to default value

            name_status = ast.literal_eval(answer_text)
            total_names = len(name_status)
            full_name_present_count = sum(1 for status in name_status.values() if 'full name PRESENT' in status)
            unsure_count = total_names - full_name_present_count

            with self.graph_output:
                clear_output(wait=True)
                labels = ['Full Name Present', 'Unsure']
                sizes = [full_name_present_count, unsure_count]
                colors = ['lightgreen', 'lightcoral']
                explode = (0.1, 0)
                                # Function to format the label as 'absolute value (relative value%)'
                def func(pct, allvals):
                    absolute = int(round(pct/100.*sum(allvals)))
                    return f"{absolute} ({pct:.1f}%)"
                plt.figure(figsize=(5, 5))
                plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=lambda pct: func(pct, sizes), shadow=True, startangle=140)
                plt.axis('equal')
                plt.show()

            with self.df_output:
                clear_output(wait=True)
                regex = r"most similar name in APS: (.+)"
                not_fully_present_names = {name: re.search(regex, similar).group(1) for name, similar in name_status.items() if 'most similar' in similar}
                not_fully_present_df = pd.DataFrame(list(not_fully_present_names.items()), columns=['Name', 'Most Similar Name'])
                display(not_fully_present_df.style.applymap(lambda x: 'background-color: #ffcccc'))

    def on_confirm_clicked(self, b):
        new_index = self.current_index + 1
        if self.texts[self.current_index] != "No names found":
            self.results['Factuality'].append((f"Answer {self.current_index + 1}", self.factuality_rating.value))
        else:
            self.results['Factuality'].append((f"Answer {self.current_index + 1}", "No names found"))
        if new_index < len(self.texts):
            self.update_visualization(new_index)
            self.title.value = f"<b>Answer {new_index + 1}:</b>"
        else:
            b.description = 'Completed ✓'
            b.disabled = True
            if self.save_path and self.id:
                results_df = pd.DataFrame(self.results)
                results_df.to_csv(f"{self.save_path}/factuality/{self.id}_factuality_{self.index}.csv", index=False)
                print("Thanks; your evaluation has been saved!")      

    def display(self):
        self.update_visualization(self.current_index)


##############################################################################################################

class FlexibilityEvaluation:
    def __init__(self, save_path=None, id=None):
        self.save_path = save_path
        self.id = id
        self.results = {'Flexibility': []}  # Initialize an empty list to store results

        self.flexibility_instruction = widgets.HTML(value="""
        <div style="margin: 20px; padding: 10px; background-color: #eef9f9; border-left: 6px solid #77cccc; overflow-wrap: break-word; word-wrap: break-word; overflow: hidden;">
            <h2 style="color: #333;">Task Instructions:</h2>
            <p>Flexibility criterion is defined by how well the template adapts to different variables while still maintaining high performance on the three criteria: Coherence, Factuality, and Factual Consistency.</p>
            <p>Consider the following when scoring:</p>
            <ul>
                <li>Rate the Flexibility from 1 (works very bad for all variables) to 10 (works smoothly for every variable).</li>
                <li>After assigning your score, please click the 'Confirm' button to finalize your evaluation.</li>
            </ul>
        </div>
        """)

        self.flexibility_rating = widgets.IntSlider(
            value=5,
            min=1,
            max=10,
            step=1,
            description='Flexibility Score:',
            style={'description_width': 'initial'},
            layout={'width': '50%'}
        )

        self.confirm_flexibility_button = widgets.Button(
            description='Confirm',
            button_style='success',
            layout={'width': '20%'}
        )
        self.confirm_flexibility_button.on_click(self.on_confirm_flexibility_clicked)

        self.rating_instruction = widgets.HTML(value="<b>Is the prompt template able to adapt well on different variables? Please provide a flexibility score:</b>")

        self.flexibility_layout = widgets.VBox([self.flexibility_instruction, self.rating_instruction, self.flexibility_rating, self.confirm_flexibility_button])

    def on_confirm_flexibility_clicked(self, b):
        self.flexibility_rating.disabled = True
        self.confirm_flexibility_button.description = 'Completed ✓'
        self.confirm_flexibility_button.disabled = True

        # Save the assigned score
        self.results['Flexibility'].append(self.flexibility_rating.value)
        if self.save_path and self.id:
            # Create a DataFrame and save the results
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(f"{self.save_path}/{self.id}_flexibility_evaluation.csv", index=False)
            print("Your evaluation has been saved! You are done. Thank you for your time! :D")

    def display(self):
        # Method to display the widget layout
        display(self.flexibility_layout)

