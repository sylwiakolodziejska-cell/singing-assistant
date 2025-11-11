import streamlit as st

from dotenv import load_dotenv
from os import getenv
from json import loads
from base64 import b64encode
from typing import List, Dict, Any
from openai import OpenAI


def encode_image_file(image_file: st.runtime.uploaded_file_manager.UploadedFile) -> Dict[str, Any]:

    mime_type = f"image/{image_file.type.split('/')[-1]}" if image_file.type else 'image/png'
    image_url = f"data:{image_file.type};base64,{b64encode(image_file.read()).decode('utf-8')}"

    return {'type': mime_type, 'image_url': image_url}


def build_user_input(
        user_prompt: str, 
        images: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
    """
    Build the user input list array for the OpenAI from user prompt and images.

    Args:
        user_prompt: The user prompt to be sent to the OpenAI
        images: The images to be sent to the OpenAI

    Returns:
        A list of input list compatible with the openAI responses API
    """
    content = []
    if user_prompt and user_prompt.strip(): # user prompt
        content.append({
            'type': 'input_text',
            'text': user_prompt.strip(),
        })
    for img in images:
        content.append({
            'type': 'input_image',
            'image_url': img['image_url'],
        })
    return [{'type': 'message', 'role': 'user', 'content': content}] # if content else []


def print_user_input(message_content: List[Dict[str, Any]]) -> None:
    for item in message_content:
        if item['type'] == 'message':
            for content_item in item.get('content', []): # get item['contents'], otherwise empty list
                if content_item['type'] == 'input_text':
                    st.markdown(content_item['text'])
                elif content_item['type'] == 'input_image':
                    st.image(content_item['image_url'], width=250)
                else:
                    st.error(f"Unknown content type: {content_item['type']}")


def get_variables() -> Dict[str, Any]:

    # Load environment variables
    load_dotenv()

    # Retrieve the credentials and environment variables
    openai_api_key = getenv('OPENAI_API_KEY') or st.secrets['OPENAI_API_KEY']
    vector_store_ids = loads(getenv('VECTOR_STORE_IDS') or st.secrets['VECTOR_STORE_IDS'])
    model_name = getenv('MODEL_NAME') or st.secrets['MODEL_NAME']
    system_prompt = getenv('SYSTEM_PROMPT') or st.secrets['SYSTEM_PROMPT']

    print(vector_store_ids)

    # Warn if OpenAI API key, the vector store id, model name or system prompt are not set:
    if not openai_api_key:
        st.warning("OpenAI API key is not set. Please set the OpenAI API key in the environment variables or Streamlit secrets.")
    if not vector_store_ids:
        st.warning("Vector store ID is not set. Please set the Vector store ID in the environment variables or Streamlit secrets.")
    if not model_name:
        st.warning("Model name is not set. Please set the model name in the environment variables or Streamlit secrets.")
    if not system_prompt:
        st.warning("System prompt is not set. Please set the system prompt in the environment variables or Streamlit secrets.")

    return {'openai_api_key': openai_api_key, 'vector_store_ids': vector_store_ids, 'model_name': model_name, 'system_prompt': system_prompt}


# Function to generate a response from the OpenAI responses API
def get_api_response(
        input: List[Dict[str, Any]], 
        type: str = 'file_search',
        max_num_results: int = 20,
        previous_response_id: str = None,
        ) -> Any:
    """
    Call the OpenAI responses API with the input list.

    Args:
        user_input: The user input list to be sent to the OpenAI
        previous_response_id: The previous response id to be sent to the OpenAI
    """

    # Load environment variables
    env_variables = get_variables()

    # Initialize the OpenAI client
    client = OpenAI(api_key=env_variables['openai_api_key'])

    # Get the response from the OpenAI responses API
    response = client.responses.create(
        model=env_variables['model_name'],
        instructions=env_variables['system_prompt'], # system prompt for the AI assistant
        input=input, # user message + uploaded images
        tools = [{
            'vector_store_ids': env_variables['vector_store_ids'],
            'type': type, 
            'max_num_results': max_num_results,
            }],
        previous_response_id=previous_response_id,
    )

    # Retrive the ID if available and get the text output
    return {'output_id': getattr(response, 'id', None), 'output_text': response.output_text}