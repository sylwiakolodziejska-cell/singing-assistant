# Import libraries
import streamlit as st

# Import helper functions from utils.py
from utils import (
    encode_image_file, 
    build_user_input, 
    print_user_input,
    get_api_response, 
    )

# App configurations
st.set_page_config(
    page_title="Bitte RAG ChatBot", 
    page_icon=':material/chat_bubble:', # speech bubble icon 
    layout='centered')

# Add a title to the app
st.title("🤖 Bitte RAG ChatBot") # include a bot emoji

# Add a description to the app
st.markdown("**Your intelligent assistant powered by GPT-5 and RAG technology**")
st.divider()

# add a collapsible section
with st.expander("ℹ️ About this webapp", expanded = False):
    st.markdown(
        """
            **Bitte RAG Chatbot**
            - **Model:** `gpt-5` via OpenAI Responses API
            - **RAG:** File Search tool using your pre-built Vector Store
            - **Features:** multi-turn chat, image inputs, clear conversation
            - **Secrets:** reads `OPENAI_API_KEY` and `VECTOR_STORE_ID` from Streamlit secrets or environment variables
            
            **How it works**
            Your message and (optional) images go to the Responses API along with a system resp.
            The File Search tool retrieves relevant passages from your vector store to ground the answer.
        """
    )

# Store the previous response id
if 'previous_response_id' not in st.session_state:
    st.session_state.previous_response_id = None

# Initialize the chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Create a sidebar with user controls 
with st.sidebar:
    st.header("User sidebar")
    st.divider()
    # Clear the conversation history - reset chat history and context
    if st.button("Clear Chat History", use_container_width = True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        # Reset the page
        st.rerun()

# Render all previous messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == 'user':
            # Handle structured messages (user input)
            print_user_input(message['content'])
        elif message['role'] == 'assistant':
            # Handle simple text messages (assistant responses)
            st.markdown(message['content'])
        else:
            st.error(f"Unknown message role: {message['role']}")

# User interface - upload images
uploaded_files = st.file_uploader(
    "Upload images", 
    type=['jpg', 'jpeg', 'png', 'webp'], 
    accept_multiple_files=True,
    key=f"file_uploader_{len(st.session_state.messages)}",
    )

# User interface - chat input
user_message = st.chat_input("Type your message here...")

if user_message is not None:

    # Process only the currently uploaded images into an API-compatible format
    images = [encode_image_file(image_file) for image_file in uploaded_files] # if uploaded_files else []

    # Build the user input list for the responses API (user message + uploaded images)
    user_input = build_user_input(user_message, images)

    # Display the user's message
    with st.chat_message('user'):
        print_user_input(user_input)

    # Store the user message
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    # Generate the AI response
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            try:
                response = get_api_response(
                    input=user_input, # user message + uploaded images
                    previous_response_id=st.session_state.previous_response_id,
                    )

                # Display the AI's response
                st.markdown(response['output_text'])

                # Store the assistant's response 
                st.session_state.messages.append({'role': 'assistant', 'content': response['output_text']})
                
                # Update previous response id
                st.session_state.previous_response_id = response['output_id']

            except Exception as e:
                st.error(f"Error generating response: {e}")