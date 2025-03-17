import streamlit as st
import os

# App title
st.set_page_config(page_title="💬 MindEase Chatbot")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "text": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

with st.sidebar:
    st.title('🧠 MindEase Chatbot')
    st.write("Your Safe Space to Talk & Heal. 💙")

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Model-A', 'Model-B'], key='selected_model')

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=20, max_value=80, value=50, step=5)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "text": "How may I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# Function to generate dynamic responses
def chatbot_response(user_message):
    if "stress" in user_message.lower():
        return "I'm here for you. Do you want to talk about what's causing stress?"
    elif "anxiety" in user_message.lower():
        return "Anxiety can be tough. Have you tried deep breathing exercises?"
    else:
        return "Hello! I'm here to chat. Tell me more about what's on your mind."

user_input = st.chat_input("Type a message...")

if user_input:
    # Display User Message
    st.session_state.messages.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_response = chatbot_response(user_input)
    st.session_state.messages.append({"role": "assistant", "text": bot_response})

    # Display Chatbot Response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
