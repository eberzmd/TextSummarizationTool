import numpy as np
import pandas as pd 
from re import sub
from os import path
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from warnings import filterwarnings
from sklearn.cluster import KMeans
filterwarnings("ignore")
from transformers import pipeline
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter
import customtkinter
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.tag import pos_tag
from random import seed
from fpdf import FPDF

seed(11)

if __name__ == "__main__":
    print("****************************** start tkinter ******************************")
    print(datetime.now())

    customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

    self = customtkinter.CTk()

    WIDTH = 650
    HEIGHT = 350

    self.geometry(f"{WIDTH}x{HEIGHT}")
    self.title("Narrative Trend Identification Tool")

    ######### Commands for all user inputs #########

    # Create string variable to display the name of the file that was selected 
    filename_var = tkinter.StringVar()
    filename_var.set("<None Selected>")


    # Select a File Button
    def UploadAction():
        global filepath
        filepath = tkinter.filedialog.askopenfilename()
        print('Selected:', filepath)
        filename = path.basename(filepath)
        self.file_selected.configure(text = filename)

    global header_ind_var
    header_ind_var = tkinter.StringVar()
    header_ind_var.set("0")
    def checkbox_event():
        print("checkbox toggled, current value:", header_ind_var.get())
        
    global model_type_var

    # Radio buttons for model type (fast v slow)
    model_type_var = tkinter.IntVar(0)

    def radiobutton_event():
        print("radiobutton toggled, current value:", model_type_var.get())        

    # Variables that will be saved when Run button is clicked
    num_clusters_var = tkinter.IntVar()
    #num_clusters_var.set(3)

    def outputfolder_event():
        global filepathout
        filepathout = tkinter.filedialog.askdirectory()


    def run_button_event():
        global unsupervised_model_ind
        global num_clusters_var
        num_clusters_var = int(self.cluster_input.get())

    ################################################

    # ========= create four frames (rows) =========
    self.grid_columnconfigure(0, weight=1)
    self.grid_rowconfigure(0, weight=1)

    self.frame_one = customtkinter.CTkFrame(master=self, corner_radius=0)


    # ========= frame_one =========

    # where does frame_one go in the overarching GUI?
    self.frame_one.grid(row=0, column=0, sticky="nswe")

    # set frame_one rows and columns
    self.frame_one.grid_rowconfigure(0, minsize=50) # padding
    self.frame_one.grid_rowconfigure((0, 1, 2, 3), minsize=20) # padding
    #self.frame_one.grid_rowconfigure(11, minsize=10) # padding
    self.frame_one.grid_columnconfigure((0,1), weight = 1, minsize=20)

    # create a text description for app in frame_one
    self.label_info_1 = customtkinter.CTkLabel(master=self.frame_one, text="The Narrative Trend Identifcation Tool ingests xlsx or csv " + 
                                                "text data and classifies it into n groups, creates sentiment scores for each entry, " +
                                                "and produces both a detailed and simplified summary.",
                                                wraplength=WIDTH + 20,
                                                justify=tkinter.LEFT)

    self.label_info_1.grid(column=0, row=0, columnspan=3, pady=20)

    # create a button to select file in frame_one
    self.upload_link = customtkinter.CTkButton(master=self.frame_one, 
                                                text="Select a File", command=UploadAction)
    self.upload_link.grid(column=0,row=1)

    # create a checkbox for headers/no headers in frame_one
    self.headers_switch = customtkinter.CTkCheckBox(master=self.frame_one,
                                                    text="Has header?", command=checkbox_event,
                                                    variable=header_ind_var, onvalue="1", offvalue="0")
    self.headers_switch.grid(column=0,row=2, pady=10)

    # placeholder for filename - will figure out how to update with selected file
    self.file_selected = customtkinter.CTkLabel(master=self.frame_one,
                                                text="[No File Selected]")
    self.file_selected.grid(column=2,row=1)


    # ========= model_options_frame =========

    # where does model_options_frame go in the overarching GUI?
    self.model_options_frame = customtkinter.CTkFrame(master=self.frame_one)
    self.model_options_frame.grid(column=0, row=8, sticky="ew", padx=(10,10))

    # set model_options_frame rows and columns
    self.model_options_frame.grid_rowconfigure((0,2,4,6), minsize=10) # padding

    # create a label for number of clusters/groups for classification model
    self.label_info_4 = customtkinter.CTkLabel(master=self.model_options_frame, text="Number of clusters/groups for " +
                                                "classification model:",
                                                wraplength=WIDTH/2,
                                                justify=tkinter.LEFT)
    self.label_info_4.grid(column=0, row=3, padx=(10,10), pady=(10,10))

    # create a text entry box for summary size input
    self.cluster_input = customtkinter.CTkEntry(master=self.model_options_frame, placeholder_text="Please enter a number")
    self.cluster_input.grid(column=0, row=5)

    # ========= bottom_section =========


    # run/exit button frame
    self.run_exit_frame = customtkinter.CTkFrame(master=self.frame_one)
    self.run_exit_frame.grid(column=2, row=8, sticky="ew", padx=(10,10))

    # set output_options_frame rows and columns
    self.run_exit_frame.grid_rowconfigure((0,2,4,6), minsize=10) # padding
    
    # label for saving results to a folder
    self.results_folder_label = customtkinter.CTkLabel(master=self.run_exit_frame, text="Save results to this folder:",
                                                        wraplength=WIDTH/2,
                                                        justify=tkinter.LEFT)
    self.results_folder_label.grid(column=0, row=0, padx=(5,5), pady=(10,10))
    
    # button to select a folder to save rsults
    self.results_folder_button = customtkinter.CTkButton(master=self.run_exit_frame, text="Select Folder", command=outputfolder_event)
    self.results_folder_button.grid(column=2, row=0, padx=(5,5), pady=(5,5))

    
    # run button
    self.run_button = customtkinter.CTkButton(master=self.run_exit_frame, text="Run", command=run_button_event, width=100)
    self.run_button.grid(column=0, row=2, padx=(0,10), pady=(20,0))

    # exit button
    self.exit_button = customtkinter.CTkButton(master=self.run_exit_frame, text="Exit", width=100, command=self.destroy)
    self.exit_button.grid(column=2, row=2, padx=(10,0), pady=(20,0))

    self.mainloop()

    print("****************************** end tkinter ******************************")
    print(datetime.now())
    
    ###########################################################################################################################
    ############## Defining functions to be used throughout notebook ##############

    # Define summarize function and load two different nlp models

    simple_summarize = pipeline("summarization", "google/pegasus-xsum", truncation=True)
    complex_summarize = pipeline("summarization", "facebook/bart-large-cnn", truncation=True) # default vals: max_length=142, min_length=56        

    # Call this function in another function so we can create summaries for multiple narratives in one line
    def multi_summarize(df):
        simple_summary_list = [simple_summarize(text) for text in df]
        #complex_summary_list = []
        #for text in df:
        #    complex_summary_list.append(complex_summarize(text))
        #    collect()
        complex_summary_list = [complex_summarize(text) for text in df]
        return simple_summary_list, complex_summary_list

    # Fix contractions so our model can understand the text
    def decontracted(phrase):
        # specific
        phrase = sub(r"won\'t", "will not", phrase)
        phrase = sub(r"can\'t", "can not", phrase)

        # general
        phrase = sub(r"n\'t", " not", phrase)
        phrase = sub(r"\'re", " are", phrase)
        #phrase = sub(r"\'s", " is", phrase)
        phrase = sub(r"\'d", " would", phrase)
        phrase = sub(r"\'ll", " will", phrase)
        phrase = sub(r"\'t", " not", phrase)
        phrase = sub(r"\'ve", " have", phrase)
        phrase = sub(r"\'m", " am", phrase)
        
        phrase = phrase.replace("'", "")
        return phrase

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    # Tokenize and lemmatize
    def preprocess(text):
        result=[]
        for token in simple_preprocess(text) :
            if token not in STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def text_cleaner(text,num):
        newString = text.lower()
        newString = BeautifulSoup(newString, "lxml").text
        newString = sub(r'\([^)]*\)', '', newString)
        newString = sub('"','', newString)
        newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
        newString = sub(r"'s\b","",newString)
        newString = sub("[^a-zA-Z]", " ", newString) 
        newString = sub('[m]{2,}', 'mm', newString)
        if(num==0):
            tokens = [w for w in newString.split() if not w in stop_words]
        else:
            tokens=newString.split()
        long_words=[]
        for i in tokens:
            if len(i)>1:                                                 #removing short word
                long_words.append(i)   
        return (" ".join(long_words)).strip()

    ###########################################################################################################################

    ### Reading in movie plot summaries
    # redundant. fix later so tkinter values so this isn't needed
    if header_ind_var=="0":
        header_val = 0
    else:
        header_val = None
    header_val = 0

    if str(filepath).endswith("csv")==True:
        df_narratives = pd.read_csv(str(filepath), header=header_val)
    else:
        df_narratives = pd.read_excel(str(filepath), header=header_val)

    df_narratives.columns = ["narratives"]
        
    # Create Summary_ID so we can connect this info to our other dataframes
    df_narratives["Narrative_ID"] = df_narratives.index+1

    id_list = []
    for row in df_narratives.index:
        if df_narratives['Narrative_ID'].iloc[row] < 10:
            id_list.append('Narrative #0' + df_narratives['Narrative_ID'].iloc[row].astype(str))
        else: id_list.append('Narrative #' + df_narratives['Narrative_ID'].iloc[row].astype(str))
    df_narratives["Narrative_ID"] = id_list


    # This particular data had \r\n show up a lot so removed in favor of space. My guess is these are page breaks.
    df_narratives['narratives'] = df_narratives['narratives'].apply(lambda x: x.replace("\r\n", " ")).apply(lambda x: x.replace("_x000D_\n", " ")).apply(lambda x: decontracted(x))
    
    # chose this vectorizer since it keeps the text data in a legible format. Other stopwords methods didn't work as well
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=0.11)
    X = vectorizer.fit_transform(df_narratives['narratives'])

    print("******************************start k means******************************")
    print(datetime.now())

    # 3 representing the 3 genres that we are hoping the model detects based on the summaries
    true_k = num_clusters_var

    if true_k > 1:
        # I forget why I needed to take off the hyperparameters. they may have broke the code/model but would be good to test
        kmeans = KMeans(n_clusters=true_k, random_state=11, init='k-means++', n_init=300, max_iter=500, tol=0.0001)
        kmeans = kmeans.fit(X)
        label = kmeans.fit_predict(X)
        # Create Groups labels based on kmeans output
        df_narratives['Groups'] = kmeans.labels_
    else: df_narratives['Groups'] = 1

    print("******************************end k means******************************")
    print(datetime.now())

    # Create df to be used to determine top topics for each group
    grouped_narratives = df_narratives.copy()
    grouped_narratives = grouped_narratives[["narratives", "Groups"]]
    grouped_narratives['narratives'] = grouped_narratives[['Groups', 'narratives']].groupby(['Groups'])['narratives'].transform(lambda x: ' '.join(x))
    grouped_narratives = grouped_narratives.drop_duplicates()
    grouped_narratives = grouped_narratives.sort_values('Groups')

    stemmer = SnowballStemmer("english")

    print("****************************** Start Topics LDA ******************************")
    print(datetime.now())

    lda_models_per_group = []
    for k in range(0, len(grouped_narratives.index)):
        tagged_sentence = [pos_tag(str(grouped_narratives.iloc[k, 0]).split())]
        edited_sentence = [[word for word,tag in tagged_sentence[i] if tag != 'NNP' and tag != 'NNPS'] for i in range(0, len(tagged_sentence))]
        rejoined_docs = [' '.join(edited_sentence[j]) for j in range(0, len(edited_sentence))]
        #processed_docs = []
        processed_docs = [preprocess(doc) for doc in rejoined_docs]
        dictionary = Dictionary(processed_docs)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        lda_models_per_group.append(LdaMulticore(bow_corpus, 
                                        num_topics = 1, 
                                        id2word = dictionary, 
                                        passes = 2,
                                        iterations = 50,
                                        decay=0.5,
                                        workers = 3))

    print("****************************** End Topics LDA ******************************")
    print(datetime.now())
    
    # Saves series for LDA Topics per Group to be used in topics_df
    groups_series = []
    topics_series = []
    for n in range(0, len(lda_models_per_group)):
        groups_series.append(f"{n}")
        for j in lda_models_per_group[n].show_topic(0, topn=10):
            topics_series.append(j[0])


    topics_df = pd.DataFrame(np.array((list(chunks(topics_series, 10)))))
    topics_df['Groups'] = groups_series
    #topics_df = topics_df.set_index('Groups').reset_index()

    print("******************************start summarization******************************")
    print(datetime.now())

    # This creates summaries for all of our narratives/text entries and saves them in summary_list
    df_narratives['SimpleSummaries'], df_narratives['ComplexSummaries'] = multi_summarize(df_narratives['narratives'])
        
    print("******************************end summarization******************************")
    print(datetime.now())

    sentiment_pipeline = pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english") # default model: distilbert-base-uncased-finetuned-sst-2-english
    # tried siebert/sentiment-roberta-large-english but it took way too long
    stop_words = set(stopwords.words('english')) 

    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                            "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                            "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                            "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                            "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                            "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                            "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                            "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                            "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                            "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                            "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                            "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                            "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                            "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                            "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                            "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                            "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                            "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                            "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                            "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                            "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                            "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                            "you're": "you are", "you've": "you have"}

    df_narratives['ComplexSummaries'] = [summary_txt[0]["summary_text"] for summary_txt in df_narratives["ComplexSummaries"]] 
    df_narratives['SimpleSummaries'] = [summary_txt[0]["summary_text"] for summary_txt in df_narratives["SimpleSummaries"]]

    cleaned_text = [text_cleaner(t,0) for t in df_narratives['ComplexSummaries']]
    #df_narratives['ComplexSummaries'] = cleaned_text

    print(datetime.now())
    # Create lists to be used in DataFrame showing the Top 5 most Positive and most Negative words in our summaries
    n=0
    summary_num = []
    sentiment_list = []
    sent_words = []
    sent_labels = []
    sent_scores = []
    for sum_item in cleaned_text:
        n+=1
        for i in range(0, len(sum_item.split())):            
            sentiment_list.append(sentiment_pipeline(sum_item.split()[i]))
            if n < 10:
                summary_num.append("Narrative #0"+str(n))
            else: summary_num.append("Narrative #"+str(n))

    for j in range(0, len(sentiment_list)):
        sent_labels.append(sentiment_list[j][0]['label'])
        sent_scores.append(sentiment_list[j][0]['score'])
    print(datetime.now())

    sentiment_df = pd.DataFrame([summary_num, sent_labels, sent_scores]).T
    sentiment_df.columns = ['summaries', 'label', 'score']
    #sentiment_df = sentiment_df.drop_duplicates()
    
    # // this line is to set the score of negative words to negative numerical value
    sentiment_df.loc[sentiment_df['label'] == "NEGATIVE", 'score'] *= -1

    summary_sentiments = sentiment_df.groupby('summaries')['score'].sum()#.sort_values()
    summary_sentiments = pd.DataFrame(summary_sentiments).reset_index()
    summary_sentiments.columns = ["Narrative_ID", "Sentiment_score"]
    
    df_narratives['Groups'] = df_narratives['Groups'].astype(str)
    master_df = df_narratives.merge(summary_sentiments, how='left', on='Narrative_ID').merge(topics_df, how='left', on='Groups')
    master_df = master_df[["Narrative_ID", "SimpleSummaries", "ComplexSummaries", "narratives", "Groups", "Sentiment_score", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    topics_df = topics_df[["Groups", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    topics_df.columns = ["Groups", "Topic_01", "Topic_02", "Topic_03", "Topic_04", "Topic_05", "Topic_06", "Topic_07", "Topic_08", "Topic_09", "Topic_10"]
    sentiment_table_df = master_df[["Groups", "Narrative_ID", "Sentiment_score"]]

    # Save to excel so that is available at least
    currentDay = datetime.now().day
    currentMonth = datetime.now().month
    currentYear = datetime.now().year

    writer = pd.ExcelWriter(str(filepathout)+'/NarrativeToolReport_'+str(currentYear)+'_'+str(currentMonth)+'_'+str(currentDay)+'.xlsx', engine='xlsxwriter')

    topics_df.to_excel(writer, sheet_name='Topics')
    summary_sentiments.to_excel(writer, sheet_name='Sentiments')
    df_narratives.to_excel(writer, sheet_name='Narratives')

    writer.save()

    df_narratives['narratives'] = df_narratives['narratives'].replace("–", "-")
    df_narratives = df_narratives.merge(sentiment_table_df, how='left', on=['Narrative_ID', 'Groups'])
    df_narratives = df_narratives.sort_values(['Groups', 'Sentiment_score']).reset_index(drop=True)

    sentiment_table_df.to_excel("C:/Users/WA683XR/Documents/Unsupervised_Text_Summarization/Final/Demo/sentiment_table_df.xlsx")
    topics_df.to_excel("C:/Users/WA683XR/Documents/Unsupervised_Text_Summarization/Final/Demo/topics_df.xlsx")
    df_narratives.to_excel("C:/Users/WA683XR/Documents/Unsupervised_Text_Summarization/Final/Demo/df_narratives.xlsx")

    pdf = FPDF()
    pdf.set_font("arial", size=24)
    
    label_data = ['Simple Summary', 'Detailed Summary', 'Full-length Narrative']*len(df_narratives)
    item_data = [[df_narratives['SimpleSummaries'][j], df_narratives['ComplexSummaries'][j], df_narratives['narratives'][j]] for j in range(0, len(df_narratives))]
    group_data = df_narratives['Groups'].reset_index(drop=True)
    topics_data = [', '.join(list(topics_df.iloc[group, 1:11])) for group in range(0, len(topics_df))]
    
    ################################ Title Page ################################
    pdf.add_page()
    # Displaying a full-width cell with centered text:
    pdf.cell(w=0,h=10, txt="Narrative Trend Identification Tool Report", align="C")
    pdf.set_font("arial", size=18)
    pdf.cell(w=-195,h=50, txt=f"Produced on {currentMonth}/{currentDay}/{currentYear}", new_x="LMARGIN", new_y="NEXT", align="C")
    
    home_link = pdf.add_link()
    pdf.set_link(home_link, y=0.0, page=1)
    #pdf.cell(w=0,h=10, txt="Table of Contents", align="C")
    pdf.multi_cell(0, 2, '\n') # line breaks
    
    pdf.set_font("arial", style="U",size=18)
    pdf.cell(w=0, h=10, txt="Group 0", new_x="LMARGIN", new_y="NEXT", align="L")  
    pdf.set_font("arial", size=14)
    pdf.cell(w=0, h=10, txt=f"Topics: "+ f"{topics_data[0]}", new_x="LMARGIN", new_y="NEXT", align="L")   
    pdf.multi_cell(0, 5, '\n') # line breaks

    pdf.set_font("arial", style="U",size=18)
    pdf.cell(w=0, h=10, txt="Group 1", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_font("arial", size=14)
    pdf.cell(w=0, h=10, txt=f"Topics: "+ f"{topics_data[1]}", new_x="LMARGIN", new_y="NEXT", align="L")    
    pdf.multi_cell(0, 5, '\n') # line breaks

    pdf.set_font("arial", style="U",size=18)
    pdf.cell(w=0, h=10, txt="Group 2", new_x="LMARGIN", new_y="NEXT", align="L")    
    pdf.set_font("arial", size=14)
    pdf.cell(w=0, h=10, txt=f"Topics: "+ f"{topics_data[2]}", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.multi_cell(0, 5, '\n') # line breaks

    pdf.set_font("arial", style="U",size=18)
    pdf.cell(w=0, h=10, txt="Group 3", new_x="LMARGIN", new_y="NEXT", align="L")    
    pdf.set_font("arial", size=14)
    pdf.cell(w=0, h=10, txt=f"Topics: "+ f"{topics_data[3]}", new_x="LMARGIN", new_y="NEXT", align="L")    
    pdf.multi_cell(0, 5, '\n') # line breaks

    pdf.set_font("arial", style="U",size=18)
    pdf.cell(w=0, h=10, txt="Group 4", new_x="LMARGIN", new_y="NEXT", align="L")    
    pdf.set_font("arial", size=14)
    pdf.cell(w=0, h=10, txt=f"Topics: "+ f"{topics_data[4]}", new_x="LMARGIN", new_y="NEXT", align="L")    
    pdf.multi_cell(0, 5, '\n') # line breaks

    #pdf.set_font("arial", style="U",size=18)
    #pdf.cell(w=0, h=10, txt="Group 5", new_x="LMARGIN", new_y="NEXT", align="L")    
    #pdf.set_font("arial", size=14)
    #pdf.cell(w=0, h=10, txt=f"Topics: "+ f"{topics_data[5]}", new_x="LMARGIN", new_y="NEXT", align="L")    
    #pdf.multi_cell(0, 5, '\n') # line breaks

    pdf.add_page()
    ################### Code for making pages for each group ###################
    col_width = pdf.w * 0.9
    spacing = 1.25
    row_height = pdf.font_size     
    
    pdf.set_font("arial", size=18)
    pdf.cell(w=0,h=10, txt=f"Group 0", align="C", link=home_link)
    pdf.multi_cell(0, 5, '\n') # line breaks
    lastgroup = '0'
    pdf.set_font("arial", size=14)
    pdf.cell(w=0,h=15, txt=f"Topics: "+ f"{topics_data[0]}", align="C")
    pdf.multi_cell(0, 10, '\n') # line breaks
    group_counter = 0
    
    for i in range(0, len(item_data)):
        while group_data[i] != lastgroup:
            group_counter += 1
            pdf.set_font("arial", size=18)
            pdf.add_page()
            pdf.cell(w=0,h=10, txt=f"Group {group_data[i]}", align="C", link=home_link)
            pdf.multi_cell(0, 5, '\n')
            pdf.set_font("arial", size=14)
            pdf.cell(w=0,h=15, txt=f"Topics: "+ f"{topics_data[group_counter]}", align="C")
            pdf.multi_cell(0, 10, '\n') # line breaks
            lastgroup = group_data[i]
        pdf.set_font("arial", style='U', size=14)
        pdf.multi_cell(0, 5, '\n')
        pdf.cell(w=0,h=15, txt=f"{df_narratives['Narrative_ID'][i]}", align="L")
        pdf.cell(w=0,h=15, txt=f"Sentiment Score: {round(df_narratives['Sentiment_score'][i], 2)}", align="R")
        pdf.multi_cell(0, 10, '\n')
        for h in range(0, 3):
            pdf.set_font("arial", size=10)
            pdf.multi_cell(col_width, row_height * spacing, txt=label_data[h], border=1, ln=1, max_line_height=pdf.font_size+20)
            pdf.multi_cell(col_width, row_height * spacing, txt=item_data[i][h], border=1, ln=1, max_line_height=pdf.font_size+20)
            pdf.ln(row_height * spacing)
        pdf.add_page()
    
    ################################# Save PDF #################################

    pdf.output('NTIT_Report_'+str(currentYear)+'_'+str(currentMonth)+'_'+str(currentDay)+'.pdf')

if __name__ == "__main__":
    import dash
    import pandas as pd
    from dash import html, dash_table
    import dash_bootstrap_components as dbc
    from dash.dependencies import Input, Output
    import webbrowser

    sentiment_table_df['id'] = sentiment_table_df.index
    
    app = dash.Dash(__name__)
    
    app.title = "Narrative Trend ID Tool"
    
    #app._favicon = ("favicon.ico")

    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll',
            'font-family':'Arial, sans-serif'
        }
    }

    style_data_conditional = [
        {
            "if": {"state": "active"},
            "backgroundColor": "#D6DBDF",
            "border": "1px solid yellow",
            'font-family':'Arial, sans-serif'
        },
        {
            "if": {"state": "selected"},
            "backgroundColor": "#D6DBDF",
            "border": "1px solid yellow",
            'font-family':'Arial, sans-serif'
        },
    ]

    app.layout = html.Div([dbc.Container([
                            html.Div([html.H1("Narrative Trend Identification Tool", 
                                              style={'position': 'relative', 
                                                     'font-weight':'bold', 
                                                     'color': 'white', 
                                                     'font-family': 'Arial, sans-serif', 
                                                     'font-size': '24px', 
                                                     'text-align': 'left',
                                                     'top':'50px'}),
                                html.Img(src=app.get_asset_url("logo.png"), 
                                         style={'position': 'absolute', 
                                                'max-height': '200px', 
                                                'left':'105px', 'top':'20px', 'bottom': '0px'})
                                    ], style=
                                     {#'border-radius': '25px',
                                      'background-image': 'url("/assets/bg2.png")', 
                                      'padding':'150px'
                                      }
                                    ),
        
        
                            html.Div([
                            html.Div("LDA Modeling - Relevant Topics by Group ", style={'font-weight':'bold'}),
                            html.Br(),
                            dash_table.DataTable(
                                data=topics_df.to_dict('records'),
                                columns=[{"name": i, "id": i} for i in topics_df.columns if i != 'id'], 
                                id='topics_tbl', 
                                style_cell={'font-family':'Arial, sans-serif', 'font-size': '12px'},
                                style_data_conditional=style_data_conditional,
                                selected_cells=[])
                            ], style={ 'padding': '25px 50px 75px 50px',
                                       'border-radius': '25px',
                                       'margin-right': '100px', 
                                       'margin-left': '100px',
                                       'background-color': '#FFFFFF',
                                       'position':'relative', 
                                       'bottom':'75px',
                                       'border-style':'solid',
                                       'border-color':'#d3d3d3',
                                       'border-width':'1px'}),
        
        
        
                            html.Br(),
                    
                            html.Div([
                            html.Div([
                                html.Div('Sentiment Scores per Narrative', style={'font-weight':'bold'}),
                                html.Br(),
                                    dash_table.DataTable(
                                        sentiment_table_df.to_dict('records'),
                                        [{"name": i, "id": i} for i in sentiment_table_df.columns if i != 'id'],
                                        id='sentiments_tbl', 
                                        page_current=0,
                                        page_size=10, 
                                        filter_action='native', 
                                        sort_action='native',
                                        sort_by=[{'column_id': 'Sentiment_score', 'direction': 'asc'}],
                                        #row_selectable='single',
                                        style_cell={'font-family':'Arial, sans-serif', 'font-size': '12px'},
                                        style_data_conditional=style_data_conditional)
                                ]),
                            html.Div('Simple Summary: ', style={'font-weight':'bold'}),
                            html.Div(id='simple-summary-output', style={'whiteSpace': 'pre-line'}),
                            html.Br(),
                            html.Div('Detailed Summary: ', style={'font-weight':'bold'}),
                            html.Div(id='detailed-summary-output', style={'whiteSpace': 'pre-line'}),
                            html.Br(),
                            html.Div('Full-length Narrative: ', style={'font-weight':'bold'}),
                            html.Div(id='full-narrative-output', style={'whiteSpace': 'pre-line'})],
                            style={ 'padding': '25px 50px 75px 50px',
                                       'border-radius': '25px',
                                       'margin-right': '100px', 
                                       'margin-left': '100px',
                                       'background-color': '#FFFFFF',
                                       'position':'relative',
                                       'bottom':'60px',
                                       'border-style':'solid',
                                       'border-color':'#d3d3d3',
                                       'border-width':'1px'}),  
        
                            ])
                        ], style={
                                  'background-color': '#2d2d30',  
                                  'background-repeat': 'no-repeat', 
                                  'background-attachment': 'fixed', 
                                  'background-position': 'center', 
                                  'marginBottom': "-10px", 'marginTop': "-10px", 
                                  'marginLeft': '-10px', 'marginRight': "-10px", 
                                  'color':'#2e2e38',
                                  'font-family': 'Arial, sans-serif'})
                    #])

    # callback and function so cells can be selected in topics table
    @app.callback(
        Output("sentiments_tbl", "style_data_conditional"),
        [Input("sentiments_tbl", "active_cell")]
        )
    def update_selected_row_color(active):
        style = style_data_conditional.copy()
        if active:
            style.append(
                {
                    "if": {"row_index": active["row"]},
                    "backgroundColor": "#D6DBDF",
                    "border": "1px solid yellow",
                    'font-family':'Arial, sans-serif'
                },
            )
        return style

    # callback and function allowing Simple Summary to be updated when a row is selected in sentiment table
    @app.callback(
        Output('simple-summary-output', 'children'),
        Input('sentiments_tbl', 'active_cell')
    )
    def simple_update_output(active_cell):
        print(active_cell)
        return '{}'.format(df_narratives["SimpleSummaries"][active_cell.get("row_id")])

    # callback and function allowing Detailed Summary to be updated when a row is selected in sentiment table
    @app.callback(
        Output('detailed-summary-output', 'children'),
        Input('sentiments_tbl', 'active_cell')
    )
    def detailed_update_output(active_cell):
        print(active_cell)
        return '{}'.format(df_narratives["ComplexSummaries"][active_cell.get("row_id")])

    # callback and function allowing Full-Length Narrative to be updated when a row is selected in sentiment table
    @app.callback(
        Output('full-narrative-output', 'children'),
        Input('sentiments_tbl', 'active_cell')
    )
    def update_output(active_cell):
        print(active_cell)
        return '{}'.format(df_narratives["narratives"][active_cell.get("row_id")])

    # testing
#    @app.callback(
#        Output('full-narrative-output', 'children'),
#        Input('sentiments_tbl', 'active_cell')
#    )
#    def update_output(active_cell):
#        print(active_cell)
#        return '{}'.format(active_cell)

    # This automatically opens the dashboard so user doesn't have to manually navigate to their browser and the correct address    
    port = 8050 # or simply open on the default `8050` port

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:{}".format(port))

    open_browser()
    app.run_server(debug=False, use_reloader=False, port=port)
