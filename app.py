import streamlit as st


def response_generator(user_input):
    None

st.title("💬 MindEase")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What's in your mind!"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    
    with st.chat_message("user"):
        st.markdown(prompt)      # Display user message

    
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))  # Display AI response with streaming effect

    
    st.session_state.messages.append({"role": "assistant", "content": response})  # Save response to history
