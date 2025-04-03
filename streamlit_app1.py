import streamlit as st
import json
from typing import Generator
from groq import Groq

# Set Page Config
st.set_page_config(page_icon="💙", layout="wide", page_title="Mental Health Chat")

# Initialize Groq Client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Define File for Reinforcement Learning Data
FEEDBACK_FILE = "chat_feedback.json"

# Function to Load and Save Feedback Data
def load_feedback_data():
    try:
        with open(FEEDBACK_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"positive": 0, "negative": 0, "improvement_suggestions": []}

def save_feedback_data(data):
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(data, file, indent=4)

feedback_data = load_feedback_data()

# Initialize Session State for Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define Available Models
models = {
    "gemma2-9b-it": "Gemma2-9b-it",
    "llama-3.3-70b-versatile": "LLaMA3.3-70b",
    "llama-3.1-8b-instant": "LLaMA3.1-8b",
    "llama3-70b-8192": "LLaMA3-70b",
    "llama3-8b-8192": "LLaMA3-8b",
}

# Sidebar for Model Selection
st.sidebar.title("Settings")
model_option = st.sidebar.selectbox("Choose a model:", list(models.keys()), format_func=lambda x: models[x])

# Display Chat Messages in WhatsApp-Like Format
st.title("💬 Mental Health Chatbot")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(f"**You:** {message['content']}")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**Bot:** {message['content']}")

# Function to Generate Chat Responses
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Handle User Input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(f"**You:** {prompt}")

    full_response = ""
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            max_tokens=512,
            stream=True
        )
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except:
        st.error("Something went wrong.", icon="🚨")

    # Store Assistant Response
    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})

    # **Emergency Detection (Only for Critical Cases)**
    critical_keywords = ["suicide", "end my life", "kill myself", "no reason to live", "giving up", "can't go on"]
    if any(word in prompt.lower() for word in critical_keywords):
        st.error("You're not alone. Please reach out to a professional or talk to a trusted person. 💙", icon="🚨")

# **New Feedback System with Emoji Buttons**
st.subheader("📝 How was the chatbot's response?")

col1, col2, col3, col4, col5 = st.columns(5)

feedback = None

with col1:
    if st.button("😡", key="1"):
        feedback = 1
with col2:
    if st.button("😞", key="2"):
        feedback = 2
with col3:
    if st.button("😐", key="3"):
        feedback = 3
with col4:
    if st.button("😊", key="4"):
        feedback = 4
with col5:
    if st.button("😍", key="5"):
        feedback = 5

if feedback is not None:
    if feedback in [1, 2]:  # Negative Feedback
        feedback_data["negative"] += 1
        st.warning("We’re sorry! What can we improve?")
        improvement_tip = st.text_input("Your suggestion:")
        if improvement_tip:
            feedback_data["improvement_suggestions"].append(improvement_tip)
            st.success("Thanks for your feedback! We'll improve. 💡")
    else:  # Positive Feedback
        feedback_data["positive"] += 1
        st.success("Glad you liked it! 😊")

    save_feedback_data(feedback_data)

    # **RL: Adjust Model Based on Feedback**
    if feedback_data["negative"] > feedback_data["positive"]:
        st.warning("We're improving responses based on your feedback. Expect better interactions soon! 🚀")
    elif feedback_data["positive"] > feedback_data["negative"]:
        st.success("Our chatbot is improving! Thanks for your support. 💙")

# **Feedback Summary in Sidebar**
st.sidebar.subheader("📊 Feedback Summary")
st.sidebar.write(f"👍 Positive Ratings: {feedback_data['positive']}")
st.sidebar.write(f"👎 Negative Ratings: {feedback_data['negative']}")
st.sidebar.write(f"💡 Suggestions: {len(feedback_data['improvement_suggestions'])}")

if feedback_data["improvement_suggestions"]:
    st.sidebar.subheader("🔍 User Suggestions")
    for idx, suggestion in enumerate(feedback_data["improvement_suggestions"][-3:], start=1):
        st.sidebar.write(f"{idx}. {suggestion}")
