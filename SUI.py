#!/bin/python

import streamlit as st
import pandas as pd
import numpy as np
import time, os
import matplotlib.pyplot as plt
import speech_recognition as sr
import sys
import os
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess


URL = st.text_input("Enter youtube URL", "")
os.system("rm sentData.txt")
if st.button('Start Mining'):
    #os.system("chmod +x start-vsm.sh")
    os.system(f"youtube-dl '{URL}' -o - | ffmpeg -i - -f wav - | pv | python3 sentiment-vsm.py&")


    chart = st.empty()

    counter = 0
    chart_data = pd.DataFrame(np.array([[0, 0, 0, 0, 0]]), columns=["anger", "disgust", "fear", "joy", "sadness"])
    while (1):
        try:
            f = open("sentData.txt")
            break
        except:
            pass
    while (1):
        EmocounT = {"anger": 0,
                    "disgust": 0,
                    "fear": 0,
                    "joy": 0,
                    "sadness": 0}
        buff = f.readline()
        buff = buff.strip()
        if buff != "":
            EmocounT[buff] += 1
            chart_data.loc[counter] = pd.DataFrame(np.array(
                [[EmocounT["anger"], EmocounT["disgust"], EmocounT["fear"], EmocounT["joy"], EmocounT["sadness"]]]), \
                                                   columns=["anger", "disgust", "fear", "joy", "sadness"]).loc[0]
            plot_data = pd.DataFrame(chart_data.sum())
            # chart.bar_chart(chart_data,orientation='vertical')
            groups = plot_data.groupby(plot_data.index)
            fig, ax = plt.subplots()
            for name, group in groups:
                ax.bar(name, group[0], label=name, align='center')
            chart.pyplot(fig)
            time.sleep(0.5)
            counter += 1
