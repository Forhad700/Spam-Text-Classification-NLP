import streamlit as st
import pickle
import numpy as np

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('svm_spam_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

st.title('Spam Text Classification')

user_input = st.text_area("Enter Your Text/Message")

if st.button('Predict Text'):
    if user_input: 
        user_input_tfidf = vectorizer.transform([user_input])  

        prediction = svm_model.predict(user_input_tfidf)

        if prediction == 'spam':
            st.write("Alert! **Spam** Detected !!!")
        else:
            st.write("No **Spam** Detected")
    else:
        st.write("Please Input Some Text to Classify")