import streamlit as st
import joblib

model=joblib.load("spam_classifier_by_svm_model.pkl")
vec=joblib.load("TFIDF_VECTORIZATION.pkl")

st.title("Spam Message Classifier")
st.write("A simple ML app to detect whether a message is spam or not spam")

user_input = st.text_area("Enter your message here:")

if st.button("Classify"):
    if user_input.strip()=="":
        st.warning("Please Enter a message first!")
    else:
        vect_input=vec.transform([user_input])
        prediction=model.predict(vect_input)[0]

        if prediction == 1:
            st.error("This message is SPAM!")
        else:
            st.success("This message is Not Spam!")