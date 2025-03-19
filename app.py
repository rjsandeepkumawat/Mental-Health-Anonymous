import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# App Config
st.set_page_config(page_title="💬 MindEase Chatbot", page_icon="🧠", layout="centered")

# Function to clear chat
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "text": "How may I assist you today?"}]
    st.session_state.alert_flag = False
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2E86C1;'>💬 MindEase Chatbot</h1>
        <p>Your friendly virtual mental health assistant</p>
        <hr style='border: 1px solid #AED6F1;'>
    </div>
""", unsafe_allow_html=True)
# Sidebar
with st.sidebar:
    st.write("Your Safe Space to Talk & Heal. 💙")
    st.button('🗑️ Clear Chat History', on_click=clear_chat_history)

    st.subheader('Models and Parameters')
    selected_model = st.selectbox('Choose a model', ['model1','model2','model3'], key='selected_model')

    temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    max_tokens = st.slider('Max Tokens', min_value=50, max_value=500, value=150, step=10)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "text": "How may I assist you today?"}]
if "alert_flag" not in st.session_state:
    st.session_state.alert_flag = False

# Load Model
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Distress keywords
distress_keywords = ["suicide", "kill myself", "die", "worthless", "hopeless", "end life", "depressed", "can't go on", "hurt myself"]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# Function to generate AI response
def generate_ai_response(user_msg):
    # Distress keyword check
    if any(keyword in user_msg.lower() for keyword in distress_keywords):
        st.session_state.alert_flag = True

    # Generate response using model
    inputs = tokenizer([user_msg], return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_tokens)
    reply = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return reply

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # User message display
    st.session_state.messages.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI Response
    bot_response = generate_ai_response(user_input)
    st.session_state.messages.append({"role": "assistant", "text": bot_response})

    with st.chat_message("assistant"):
        st.markdown(bot_response)

# Show distress alert at bottom
if st.session_state.alert_flag:
    st.markdown("---")
    st.warning("⚠️ **It seems you're facing something tough. Please don't hesitate to reach out to professionals.**")
    st.info("💡 **Suggestion:** Talk to a mental health expert or use trusted apps like Nirveon X")
    st.markdown("---")

# Footer
st.markdown("""
    <hr style='border: 1px solid #AED6F1;'>
    <p style='text-align:center; color: gray;'>Powered by MindEase | Your well-being matters 💙</p>
""", unsafe_allow_html=True)
