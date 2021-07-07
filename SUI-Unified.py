#!/bin/python
import base64
import streamlit as st
import speech_recognition as sr
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import subprocess,time

# st.title("Welcome to EmowavE!")

stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
punkts='''"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~'''


def CorFilt(i):
    ps = PorterStemmer()

    buff = word_tokenize(i.lower().replace("\n", "").replace("  ", " ").replace("n't", " not"))
    buff2 = ""
    for j in pos_tag(buff):
        if j[-1] == 'RB' and j[0] != "not":
            pass
        else:
            buff2 += j[0] + " "
    buff2 = buff2.replace("not ", "NOT")
    buff = word_tokenize(buff2.strip())
    ans = ""
    for j in buff:
        if (j not in punkts) and (j not in stopwords):
            if j == "!":
                ans += " XXEXCLMARK"
            elif j == "?":
                ans += " XXQUESMARK"
            else:
                if j != "'s" and j != "``":
                    ans += " " + ps.stem(j)
    return ans.strip()

import pickle

f=open("vectorizer","rb")
vectorizer=pickle.load(f)
f.close()

st.sidebar.markdown("## Select Backend")
my_button = st.sidebar.radio("", ('VSM', 'LSTM'))

if my_button == 'VSM':
    f = open("EmoVec", "rb")
    EmoVec = pickle.load(f)
    f.close()


    def EmowavE(sent, vectorizer=vectorizer, EmoVec=EmoVec, trans=True):
        transDict = {'gu': 'Gujarati',
                     'hi': 'Hindi'}
        # Translate from any language to english
        if trans:
            analysis = TextBlob(sent)
            if analysis.detect_language() != 'en':
                try:
                    print(f"\nInput text was in {transDict[analysis.detect_language()]}")
                except:
                    print(f"\nInput text was not in English")
                print("\nTranslating...")
                output = subprocess.check_output(['trans', '-b', sent])
                sent = output.decode('utf-8').strip()
                print(f"\nTranslation in English: {sent}")

        EmoBuff = vectorizer.transform([CorFilt(sent)])
        EmoDict = {0: 'anger',
                   1: 'disgust',
                   2: 'fear',
                   3: 'joy',
                   4: 'sadness'}
        return EmoDict[
            np.argmax([float(cosine_similarity(EmoBuff.reshape(-1, 1).T, EmoVec[i].reshape(-1, 1).T)) for i in
                       range(EmoVec.shape[0])])]


    # file_ = open("images/listening_text.GIF", "rb")
    # contents = file_.read()
    # data_url = base64.b64encode(contents).decode("utf-8")
    # file_.close()

    r = sr.Recognizer()
    m = sr.Microphone()
    counter = 0
    # if st.button('Start Analyzing'):
    placeholder0 = st.empty()
    placeholder1 = st.empty()
    placeholder2 = st.empty()

    try:
        print("A moment of silence, please...")
        with m as source:
            r.adjust_for_ambient_noise(source, duration=5)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
        while True:
            print("Say something!")
            # placeholder0.write("Say something!")
            # placeholder0.markdown(
            #     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            #     unsafe_allow_html=True,
            # )
            placeholder0.image("images/listening_text.GIF")
            with m as source:
                audio = r.listen(source)
            print("Got it! Now to recognize it...")
            placeholder2.write("")
            placeholder0.image("images/analyzing_text.GIF")
            # placeholder0.write("Got it! Trying to recognize it...")
            try:
                # recognize speech using Google Speech Recognition
                value = r.recognize_google(audio)

                # we need some special handling here to correctly print unicode characters to standard output
                if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                    print(u"You said {}".format(value).encode("utf-8"))
                else:  # this version of Python uses unicode for strings (Python 3+)
                    # placeholder0.write("")
                    print("You said {}".format(value))
                    placeholder1.write("You said : {}".format(value))
                    print(f"Emotion detected: {EmowavE(value)}")
                    placeholder2.image(f"images/{EmowavE(value)}.png")
            except sr.UnknownValueError:
                print("Oops! Didn't catch that")
                placeholder0.image("images/retry_text.GIF")
                placeholder1.write("EmowavE could not understand that!")
                time.sleep(3)
                placeholder1.write("")
                # placeholder0.write("Oops! Didn't catch that")
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
                placeholder0.image("images/retry_text.GIF")
                time.sleep(3)
                # placeholder0.write("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    except KeyboardInterrupt:
        pass

############################################################################################################

elif my_button == 'LSTM':
    import tensorflow as tf
    model=tf.keras.models.load_model("models/")


    def EmopreD(sent, model=model, vectorizer=vectorizer):
        EmoDict = {0: 'anger',
                   1: 'disgust',
                   2: 'fear',
                   3: 'joy',
                   4: 'sadness'}

        buff = vectorizer.transform([CorFilt(sent)]).toarray()
        return EmoDict[np.argmax(model.predict(buff.reshape(1, 1, buff.shape[1])))]


    # file_ = open("images/listening_text.GIF", "rb")
    # contents = file_.read()
    # data_url = base64.b64encode(contents).decode("utf-8")
    # file_.close()

    r = sr.Recognizer()
    m = sr.Microphone()
    counter = 0
    # if st.button('Start Analyzing'):
    placeholder0 = st.empty()
    placeholder1 = st.empty()
    placeholder2 = st.empty()

    try:
        print("A moment of silence, please...")
        with m as source:
            r.adjust_for_ambient_noise(source, duration=5)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
        while True:
            print("Say something!")
            # placeholder0.write("Say something!")
            # placeholder0.markdown(
            #     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            #     unsafe_allow_html=True,
            # )
            placeholder0.image("images/listening_text.GIF")
            with m as source:
                audio = r.listen(source)
            print("Got it! Now to recognize it...")
            placeholder2.write("")
            placeholder0.image("images/analyzing_text.GIF")
            # placeholder0.write("Got it! Trying to recognize it...")
            try:
                # recognize speech using Google Speech Recognition
                value = r.recognize_google(audio)

                # we need some special handling here to correctly print unicode characters to standard output
                if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                    print(u"You said {}".format(value).encode("utf-8"))
                else:  # this version of Python uses unicode for strings (Python 3+)
                    # placeholder0.write("")
                    print("You said {}".format(value))
                    placeholder1.write("You said : {}".format(value))
                    print(f"Emotion detected: {EmopreD(value)}")
                    placeholder2.image(f"images/{EmopreD(value)}.png")
            except sr.UnknownValueError:
                print("Oops! Didn't catch that")
                placeholder0.image("images/retry_text.GIF")
                placeholder1.write("EmowavE could not understand that!")
                time.sleep(3)
                placeholder1.write("")
                # placeholder0.write("Oops! Didn't catch that")
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
                placeholder0.image("images/retry_text.GIF")
                time.sleep(3)
                # placeholder0.write("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    except KeyboardInterrupt:
        pass




