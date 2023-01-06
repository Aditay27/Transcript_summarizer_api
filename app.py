from youtube_transcript_api import YouTubeTranscriptApi, VideoUnavailable, TooManyRequests
from youtube_transcript_api.formatters import TextFormatter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from flask import Flask,jsonify, request
from pytube import YouTube, extract
import os
import sys
import nltk
import speech_recognition as sr
r = sr.Recognizer()
from moviepy.editor import *

from summarize import sumy_lsa_summarize, sumy_text_rank_summarize, transformers_summarize,cosine_similarity
from dotenv import load_dotenv
# Creating Flask Object and returning it.
app = Flask(__name__)

# "Punkt" download before nltk tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print('Downloading punkt')
    nltk.download('punkt', quiet=True)

# "Wordnet" download before nltk tokenization
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print('Downloading wordnet')
    nltk.download('wordnet')

# "Stopwords" download before nltk tokenization
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print('Downloading Stopwords')
    nltk.download("stopwords", quiet=True)

#Summarization for Videos without Subtitles
@app.route('/transcription/<path:link>',methods=['GET'])
def transcription(link):
    #link=link.replace('/watch?v=','/')
    video = YouTube(link)
    yt = video.streams.get_lowest_resolution()
    yt.download(filename = 'audio.mp4')
    
    video_file=VideoFileClip(r"audio.mp4")
    video_file.audio.write_audiofile(r"audio.wav")
    
    with sr.AudioFile('audio.wav') as source:
        audio = r.listen(source)
    
    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:
        # using google speech recognition
        formatted_text = r.recognize_google(audio)
        
        summary_lsa= sumy_lsa_summarize(formatted_text)  # Sumy for extractive summary using LSA.
        summary_textrank= sumy_text_rank_summarize(formatted_text)  # Sumy for Text Rank Based Summary.
        summary_huggingface=transformers_summarize(formatted_text) #Hugging Face summarization using Transformers.

        # Returning Result
        response_list1 = {'processed_summary': summary_lsa['Summary'],'Execution_time in seconds':summary_lsa['Execution_Time in Seconds'],
                             'similarity_score':cosine_similarity(formatted_text,summary_lsa['Summary'])}
        response_list2 = {'processed_summary': summary_textrank['Summary'],'Execution_time in seconds':summary_textrank['Execution_Time in Seconds'],
                             'similarity_score':cosine_similarity(formatted_text,summary_textrank['Summary'])}
        response_list3 = {'processed_summary': summary_huggingface['Summary'],'Execution_time in seconds':summary_huggingface['Execution_Time in Seconds'],
                             'similarity_score':cosine_similarity(formatted_text,summary_huggingface['Summary'])}


        return jsonify(response_lsa=response_list1, response_textrank=response_list2,
                        response_huggingface=response_list3, original_text=formatted_text), 200

        
    except:
        return 'Sorry Run again'
    
    
#Summarization for Videos with Subtitles
@app.route('/summarize/<path:url>', methods=['GET'])
def transcript_fetched_query(url):
    #url=url.replace('/watch?v=','/')
    id=extract.video_id(url)
        
    if id:
        try:
            # Using Formatter to store and format received subtitles properly.
            formatter = TextFormatter()
            transcript = YouTubeTranscriptApi.get_transcript(id)
            text = ' '.join([d['text'] for d in transcript])
            formatted_text = formatter.format_transcript(transcript).replace("\n", ".")

            summary_lsa= sumy_lsa_summarize(formatted_text)  # Sumy for extractive summary using LSA.
            summary_textrank= sumy_text_rank_summarize(formatted_text)  # Sumy for Text Rank Based Summary.
            summary_huggingface=transformers_summarize(text) #Hugging Face summarization using Transformers.

            # Returning Result
            response_list1 = {'processed_summary': summary_lsa['Summary'],'Execution_time':summary_lsa['Execution_Time in Seconds'],
                             'similarity_score':cosine_similarity(formatted_text,summary_lsa['Summary'])}
            response_list2 = {'processed_summary': summary_textrank['Summary'],'Execution_time':summary_textrank['Execution_Time in Seconds'],
                             'similarity_score':cosine_similarity(formatted_text,summary_textrank['Summary'])}
            response_list3 = {'processed_summary': summary_huggingface['Summary'],'Execution_time':summary_huggingface['Execution_Time in Seconds'],
                             'similarity_score':cosine_similarity(text,summary_huggingface['Summary'])}


            return jsonify(response_lsa=response_list1, response_textrank=response_list2,
                           response_huggingface=response_list3, original_text=formatted_text), 200

        # Catching Exceptions
        except VideoUnavailable:
            return jsonify(success=False, message="VideoUnavailable: The video is no longer available.",
                                response=None), 400
        except TooManyRequests:
            return jsonify(success=False,
                                message="TooManyRequests: YouTube is receiving too many requests from this IP."
                                        " Wait until the ban on server has been lifted.",
                                response=None), 500

            
    elif id is None or len(id) <= 0:
        # video_id parameter doesn't exist in the request.
        return jsonify(success=False,
                        message="Video ID is not present in the request. "
                                "Please check that you have added id in your request correctly.",
                        response=None), 400
        
    else:
        # Some another edge case happened. Return this message for preventing exception throw.
        return jsonify(success=False,
                        message="Please request the server with your arguments correctly.",
                        response=None), 400


if __name__=="__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=80)
