# NLTK Imports
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sumy Imports
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from transformers import pipeline
import time

def sumy_lsa_summarize(text_content):
    # Latent Semantic Analysis is a unsupervised learning algorithm that can be used for extractive text summarization.
    # Initializing the parser
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    start_time=time.time()
    # Initialize the stemmer
    #stemmer = Stemmer('english')
    # Initializing the summarizer
    summarizer = LsaSummarizer()
    #summarizer.stop_words = get_stop_words('english')

    # Finding number of sentences and applying percentage on it: since sumy requires number of lines
    #sentence_token = sent_tokenize(text_content)
    #select_length = int(len(sentence_token) * (1/2))

    # Evaluating and saving the Summary
    summary = ""
    for sentence in summarizer(parser.document,1):
        summary += str(sentence)
    # Returning NLTK Summarization Output
    end_time=time.time()
    execution_time=end_time-start_time
    return {'Summary':summary,'Execution_Time in Seconds':execution_time}

def sumy_text_rank_summarize(text_content):
    # TextRank is an unsupervised text summarization technique that uses the intuition behind the PageRank algorithm.
    # Initializing the parser
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    start_time=time.time()
    # Initialize the stemmer
    #stemmer = Stemmer('english')
    # Initializing the summarizer
    summarizer = TextRankSummarizer()
    #summarizer.stop_words = get_stop_words('english')

    #sentence_token = sent_tokenize(text_content)
    #select_length = int(len(sentence_token)* (int(percent) / 100))

    # Evaluating and saving the Summary
    summary = ""
    for sentence in summarizer(parser.document,1):
        summary += str(sentence)
    # Returning NLTK Summarization Output
    end_time=time.time()
    execution_time=end_time-start_time
    return {'Summary':summary,'Execution_Time in Seconds':execution_time}

def transformers_summarize(text_content):
    #parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    start_time=time.time()
    summarization = pipeline("summarization")
    
    summary = ''
    for i in range(0, (len(text_content)//1000)+1):
        summary_text = summarization(text_content[i*1000:(i+1)*1000])[0]['summary_text']
        summary = summary + summary_text + ' '
    
    #summary = summarization(text_content)[0]['text']
    #for sentence in summarization(parser.document, sentences_count=select_length):
        #summary += str(sentence)
        
    end_time=time.time()
    execution_time=end_time-start_time
    return {'Summary':summary,'Execution_Time in Seconds':execution_time}

def cosine_similarity(original_text,summarized_text):
    X_list = word_tokenize(original_text) 
    Y_list = word_tokenize(summarized_text)
  
    # sw contains the list of stopwords
    sw = stopwords.words('english') 
    l1 =[];l2 =[]
  
    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}
  
    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0
  
    # cosine formula 
    for i in range(len(rvector)):
        c+= l1[i]*l2[i]
        cosine = c / float((sum(l1)*sum(l2))**0.5)
    
    return cosine



