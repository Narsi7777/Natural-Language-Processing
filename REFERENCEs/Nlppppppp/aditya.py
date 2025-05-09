import streamlit as st
from transformers import pipeline

# Load model once
@st.cache_resource
def load_bot():
    return pipeline("text-generation", model="microsoft/DialoGPT-small")

bot = load_bot()

st.title("🤖 Offline AI Chatbot (No API Key)")
st.write("Talk to a local AI without internet or API key!")

if "history" not in st.session_state:
    st.session_state.history = []

# Show chat history
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input
user_input = st.chat_input("Say something to the chatbot...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        result = bot(user_input, max_length=1000, num_return_sequences=1)
        response = result[0]['generated_text'][len(user_input):].strip()

    st.session_state.history.append({"role": "assistant", "content": response})

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)
