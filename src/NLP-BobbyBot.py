#!/usr/bin/env python
# coding: utf-8

# In[64]:


import nltk
import string
import random
import pandas as pd
from gtts import gTTS
import os
import pyaudio
from nltk.stem.porter import *
from nltk.stem import * 
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import speech_recognition as sr
import sklearn
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer


# Module #1: Load and Preprocess text data.
# In the first module, go to https://en.wikipedia.org/wiki/Coronavirus_disease_2019 or to https://en.wikipedia.org/wiki/Tennis or pick any other Wikipedia text about a topic of your choice:
# 1. Copy some text from the page, 
# 2. Paste it in a word document, 
# 3. Save it as .txt file.
# a. Save the file in the same directory where you saved your jupyter notebook for homework 3.
# 4. Preprocess and normalize your text.

# In[65]:


## link used: https://en.wikipedia.org/wiki/Music

# 3)

File_object = open("C:\\Users\\MEPI\\Desktop\\NLP\\Music.txt","r")

file = File_object.read()

File_object.close() 


# In[66]:


#4) Lower case - no punkt - remove stop words- NB: we are not gonna feed a file that has no punctuations and stop words removed
# to the bot in order to have an output that is readable

sent_tokens = sent_tokenize(file)

word_tokens = word_tokenize(file) 

stop_words = set(stopwords.words('english')) 

# remove stop words

new_file = []  

for w in word_tokens: 
    if w not in stop_words: 
        new_file.append(w) 


new_file1=" ".join(new_file)

## remove punctuation

no_punctuations="".join(s for s in new_file1 if s not in string.punctuation)

## make it lower case

final_file= no_punctuations.lower()


# Module #2: Greeting the bot.
# In the second module you will create two lists of strings, one is called: my_greetings and the other is called bot_response:
# 1. List: my_greetings consists of a selection of greeting words you may say to the bot.
# 2. List: bot_response consists of a selection of replies returned by the bot, randomly.
# a. Upon your input, the bot checks if your input exists in my_greetings, if it does, then it picks a random reply from bot_response.
# 3. At any time, if you input an erroneous text, the bot should reply: “I am sorry, I do not understand what you are asking me.”

# In[67]:


my_greetings= ['hi','hello','good afternoon','hola','bonjour','bonsoir','salut']

bot_response= ['Ahla w sahla','Hello there human! :)','Bonjour tout les jours!', 'Bonsoir tout les soirs!','Saba7o']

def greeting(user_input):
    boolean = 1
    
    for k in my_greetings:
        if k==user_input:
            index=random.randint(0, len(bot_response)-1)
            return bot_response[index]
            boolean = 0
    
    if boolean == 1:
        return "I am sorry, I do not understand what you are asking me."


# Module #3: Text Vectorization and Similarity Measure.
#     
# In this third module, you will ask the bot some questions about the topic you chose in Module 1. For example, if you chose the tennis page on Wikipedia, then suitable questions can be:
# • Tell me about Roger Federer
# • What is a grand slam tournament?
# In this case, the bot will search the text for an appropriate answer to your question. To that end, you should:
# 1. Vectorize your question and text using TF-IDF technique.
# 2. Using Cosine similarity measure, return from the text the part of text that has
# the highest similarity measure with your question.
# 3. At any time, if you input an erroneous text, the bot should reply: “I am
# sorry, I do not understand what you are asking me.”

# ### Please refer to the last cell to run all methods

# In[68]:



# ===============THE BELOW DIDNT WORK===== TRIAL 1
# def ask_question():
#     question=input("What would you like to know?\n\n")
#    
#     feature_names = vectorizer.get_feature_names()
#     dense = vectors.todense()
#     denselist = dense.tolist()
#     df = pd.DataFrame(denselist, columns=feature_names)
    
#     cos_vec0=vectors[0].reshape(1,-1)
#     cos_vec1=vectors[1].reshape(1,-1)

#     cos_sim=cosine_similarity(vectors[0].reshape(1,-1),vectors[1].reshape(1,-1)) 
    
#     if cos_sim==0:
#         print("I am sorry, I do not understand what you are asking me.”)
#     else:
#         print (f"Cosine Similarity between the question and the text:{cos_sim}") 
#         print (f"Cosine Distance between the question and the text:{1-cos_sim}")


## ============ FINAL CODE ==============

vectorizer = TfidfVectorizer()

def ask_question(user_response):
    
    robo_response=''
    
    sent_tokens1=sent_tokenize(file)  #we are only doing sentence tokenization in order to keep the output in a readable format
    
    sent_tokens1.append(user_response)
    
    tfidf = vectorizer.fit_transform(sent_tokens1)
    
    vals = cosine_similarity(tfidf[-1], tfidf)
    
    idx=vals.argsort()[0][-2]
    
    flat = vals.flatten()
    
    flat.sort()
    
    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        robo_response=robo_response+" I am sorry! I don't understand you"
        return robo_response
    
    else:
        robo_response = robo_response+sent_tokens1[idx]
        return robo_response
    
def user_and_bot_text():  
    
    bye = False
    
    print()
    
    print("Bobby Bot: My name is Bobby Bot, today I will answer all your questions on MUSIC! :)\nIf you want to end the conversation, type Bye or Bye Bobby!")
    
    print()
    
    user_response = input()
    
    user_response=user_response.lower()  # to make sure it reads messages such as BONJOUR

    if(user_response!='bye'and user_response!='bye bobby'):
        print()
        bot_answer=greeting(user_response)  
        print("Bobby Bot: "+bot_answer)
        
    while(bye==False):
        
        question=''
        print()
        question=input("Bobby Bot: Ask me something...if no questions, end the conversation! \n\n")
        
        if(question!='bye'and question!='bye bobby'):
            print()
            print("Bobby Bot: "+ ask_question(question))

        else:
            bye=True
            print()
            print("Bobby Bot: Bye Human! Take care :)")


# Module #4: Speech Recognition.
# In module 4, you will incorporate a speech recognition task.
# Part I: To that end, you need to install on your system the SpeechRecognition package. Then:
# 1. Record your voice uttering a line of task such as a greeting: “Hello Bobby bot” or “What is Covid”, or “tell me about the tennis grand slam” (depending on your text).
# 2. Load the mp3 voice into your jupyter notebook
# 3. Input it to the speech recognizer
# 4. Print the text transcription of the line you recorded.
# 5. Feed this line to your bot, i.e., call the method with the transcribed text,
# instead of user input (that you did in modules 2 and 3).
# Part II: Then, use your microphone to talk directly and have the speech recognizer recognize your utterance. To that end you need to install:
# 1. portaudio
# 2. pyaudio
# 3. Installation instructions for Windows and Mac are found here:
# https://people.csail.mit.edu/hubert/pyaudio/
# 4. The text you speak through the microphone, should be captured and sent to the bot, and the bot should reply (Module 3).
# 

# In[69]:


# Part 1

#NB: This method will no be as interactive as the other methods, since we do not really know which recording the user wants to
# play first.



def user_2():
    
    r = sr.Recognizer()
    
    sound=sr.AudioFile("C:\\Users\\MEPI\\Desktop\\NLP\\what is music.wav") #make sure to adjust this directory when you are running it on your machine

    with sound as source:
        
        r.adjust_for_ambient_noise(source)
        audio=r.listen(source)
        
    try:
        result=r.recognize_google(audio,language='en')
        print('Converting audio transcripts into text ...')
        print()
        print(result)
        print()
        print("Bobby Bot: "+ ask_question(result))

    except:
         print('Sorry.. run again...')
    
user_2()
    


# In[70]:


# Part 2
# in this part the method will be interactive and the user can ask any question from the pool of questions in any order
# he/she wants

def user():
    
    with sr.Microphone() as source:
        
        r=sr.Recognizer()
        r.adjust_for_ambient_noise(source)
        data=r.record(source,duration=10)
        print()
        print("User's voice......")

        try:
            text=r.recognize_google(data,language='en')
            return text
        except:
             print("Sorry, I did not get that")
                
                
def user_more_duration():
    
    with sr.Microphone() as source:
        
        r=sr.Recognizer()
        r.adjust_for_ambient_noise(source)
        data=r.record(source,duration=20)
        print()
        print("User's voice......")

        try:
            text=r.recognize_google(data,language='en')
            return text
        except:
             print("Sorry, I did not get that")

                
def user_speaks():
    
    bye2=False
    
    print()
    
    print("Bobby Bot: My name is Bobby Bot, today I will answer all your questions on MUSIC! :)\nIf you want to end the conversation, type Bye or Bye Bobby!")
    
    print()
    
    print()
    
    print("Start talking now")
    
    user_response = user()
    
    user_response=user_response.lower()  # to make sure it reads messages such as BONJOUR

    if(user_response!='bye'and user_response!='bye bobby'):
        bot_answer=greeting(user_response)
        print()
        print("Bobby Bot: "+bot_answer)
        

    while(bye2==False):
        
        print()
        print("Bobby Bot: Ask me something!\n\n")
        print()
        print("Start talking now")
        question=user()
        
        if(question!='bye'and question!='bye bobby'):
            print()
            print("Bobby Bot: "+ ask_question(question))
            
        else:
            bye2=True
            print()
            print("Bobby Bot: Bye Human! Take care :)")

        


# Module #5:
# In this fifth module, you need to implement a text-to-speech task, where you output the response of the bot through your system’s speakers. To this end you need to install:
# 1. gTTs package of pypi
# 2. Installation instructions for Windows and Mac are found here:
# https://pypi.org/project/gTTS/
# 
# Module #6:
# Finally, at any moment you type and/or speak “Bye” or “Bye Bobby”, the bot should reply with “Bye human” or “Take care” and close the input (you can no longer input text).
# 
# ➔In fine, when I run your code, I should be able to greet your Bobby bot or ask him a question, by inputting a text and/or by speaking through the microphone. Bobby bot should reply to me by text and/or through the speakers.
# 
# ➔All the best! Enjoy your first Chatbot!

# In[71]:


def bot_speaks():
    
    bye5=False
    
    audio1 = gTTS(text="My name is Bobby Bot, today I will answer all your questions on MUSIC! :)\nIf you want to end the conversation, type Bye or Bye Bobby!", lang='en')
    
    audio1.save("message1.mp3") #the parameters are machine and OS specific...e.g.: using macOS probably should be afplay message1.mp3
    
    os.system("message1.mp3")
    
    print()
    
    user_response = input()
    
    user_response=user_response.lower()  # to make sure it reads messages such as BONJOUR

    if(user_response!='bye'and user_response!='bye bobby'):
        bot_answer=greeting(user_response)
        audio2 = gTTS(text= bot_answer, lang='en')
        audio2.save("message2.mp3")
        os.system("message2.mp3")
        
        

    while(bye5==False):
        
            
        print()
            
        question=input("Bobby Bot: Ask me something!\n\n")
            
        if(question!='bye'and question!='bye bobby'):
            answer= ask_question(question)
            audio4 = gTTS(text=answer , lang='en')
            audio4.save("message4.mp3")
            os.system("message4.mp3")
                
                
            
        else:
            bye5=True
            audio5 = gTTS(text="Bye Human! Take care! :)", lang='en')
            audio5.save("message5.mp3")
            os.system("message5.mp3")

        


# In[72]:


# additional method. In this method both the user and the bot speak.
# NB: when using this method the bot will replt with an answer 20 seconds after the bot starts talking

def user_and_bot_speak():
    
    bye6=False
    
    audio1 = gTTS(text="My name is Bobby Bot, today I will answer all your questions on MUSIC! :)\nIf you want to end the conversation, type Bye or Bye Bobby!", lang='en')
    
    audio1.save("message1.mp3")
    
    os.system("message1.mp3")
    
    print()
    
    print("Start talking when bobby finishes talking")
    
    user_response = user_more_duration()
    
    user_response=user_response.lower()  # to make sure it reads messages such as BONJOUR

    if(user_response!='bye'and user_response!='bye bobby'):
        bot_answer=greeting(user_response)
        audio2 = gTTS(text= bot_answer, lang='en')
        audio2.save("message2.mp3")
        os.system("message2.mp3")

    while(bye6==False):
        
        print()
        print("Bobby Bot: Ask me something!\n\n")
        print()
        print("Start talking when bobby finishes talking")
        question=user_more_duration()

        if(question!='bye'and question!='bye bobby'):
            answer=ask_question(question)
            audio2 = gTTS(text=answer, lang='en')
            audio2.save("message3.mp3")
            os.system("message3.mp3")

        else:
            bye6=True
            audio5 = gTTS(text="Bye Human! Take care! :)", lang='en')
            audio5.save("message5.mp3")
            os.system("message5.mp3")


# ### What is music?
# ### What are some general definitions of music?
# ### Where does the word music derive from?
# ### Where does music play a key role?
# ### Give me some examples of music genres

# In[73]:


# =========== Please select any of the questions above =========

speaks=input("Would you like to speak or not? Enter yes or no\n\n")
speaks=speaks.lower()
print()
robot=input("Would you like to hear Bobby speak or not? Enter yes or no\n\n")
robot=robot.lower()

if speaks=='yes':
    if robot=='yes':
        user_and_bot_speak()
        
    else: user_speaks()
    
elif robot=='yes':
    bot_speaks()
else: user_and_bot_text()

## NB: to start the conversation again, make sure to end the previous one by typing bye or bye bobby
## NB: be patient to get an answer to a question when either you are bobby are speaking...bobby is still a baby
## NB: You can only greet the bot once
## NB: If you are speaking through microphone, do not take all time in the world, so bobby can help you...and we do not end up with an empty object :p
## NB: when you are speaking through microphone and you want to end the conversation...say bye only
## NB: Make sure to greet the bot before asking any questions...be polite ^_^
## ---------------------------------------------------------------------------------------------------------------


# In[ ]:





# In[ ]:





# In[ ]:




