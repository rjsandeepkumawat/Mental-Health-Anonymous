import streamlit as st
import json
from typing import Generator
from groq import Groq

# Set Page Config
st.set_page_config(page_title="💬 MindEase Chatbot", page_icon="🧠", layout="centered")

# Initialize Groq Client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Define Feedback File
FEEDBACK_FILE = "chat_feedback.json"

# Load & Save Feedback Data
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

# Session State for Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Model Selection
models = {
    "gemma2-9b-it": "Gemma2-9b-it",
    "llama-3.3-70b-versatile": "LLaMA3.3-70b",
    "llama-3.1-8b-instant": "LLaMA3.1-8b",
    "llama3-70b-8192": "LLaMA3-70b",
    "llama3-8b-8192": "LLaMA3-8b",
}
st.sidebar.title("Settings")
model_option = st.sidebar.selectbox("Choose a model:", list(models.keys()), format_func=lambda x: models[x])

# Clear Chat Button
def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button("🗑 Clear Chat", on_click=clear_chat)

# **Chat UI**
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💬 MindEase Chatbot</h1>", unsafe_allow_html=True)

# **Display Chat Messages**
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar="🧑‍⚕️" if msg["role"] == "user" else "🤖"):
        st.markdown(f"**{'You' if msg['role'] == 'user' else 'Bot'}:** {msg['content']}")

# **Generate Chat Response**
def chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# **Handle User Input**
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
            response_gen = chat_responses(chat_completion)
            full_response = st.write_stream(response_gen)
    except:
        st.error("Something went wrong.", icon="🚨")

    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})

    # **Emergency Detection (Only in Critical Cases)**
    emergency_keywords = ["suicide", "end my life", "kill myself", "no reason to live", "giving up", "can't go on"]
    if any(word in prompt.lower() for word in emergency_keywords):
        st.error("You're not alone. Please talk to a professional or someone you trust. 💙", icon="🚨")

# **Sidebar Feedback System (Bottom Left)**
st.sidebar.subheader("💬 Chat Feedback")

feedback = st.sidebar.radio(
    "Rate your experience:",
    ["😡 Very Bad", "😞 Bad", "😐 Neutral", "😊 Good", "😍 Excellent"],
    index=None
)

if feedback:
    score = {"😡 Very Bad": 1, "😞 Bad": 2, "😐 Neutral": 3, "😊 Good": 4, "😍 Excellent": 5}[feedback]
    feedback_data["positive" if score > 3 else "negative"] += 1

    if score <= 2:
        st.sidebar.warning("What can we improve?")
        tip = st.sidebar.text_input("Your suggestion:")
        if tip:
            feedback_data["improvement_suggestions"].append(tip)

    save_feedback_data(feedback_data)

# **Feedback Summary**
st.sidebar.subheader("📊 Feedback Summary")
st.sidebar.write(f"👍 Positive: {feedback_data['positive']} | 👎 Negative: {feedback_data['negative']}")
if feedback_data["improvement_suggestions"]:
    st.sidebar.subheader("🔍 Recent Suggestions")
    for s in feedback_data["improvement_suggestions"][-3:]:
        st.sidebar.write(f"- {s}")
