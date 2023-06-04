import streamlit as st
import pickle
import numpy as np

lr = pickle.load(open('lr.pkl','rb'))

st.title("Sentiment Analysis")

    # Text input
text = st.text_area("Enter text:", "")

if st.button("Classify"):
    # Perform sentiment analysis
    sentiment = str(lr.predict(text)[0])

    
    # Display sentiment result
    st.write("Sentiment:", sentiment)   

    
