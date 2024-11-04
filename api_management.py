import streamlit as st
import os
import logging

# Configure logging for API key retrieval
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_management.log"),
        logging.StreamHandler()
    ]
)


def get_api_key(api_key_name: str, session_state_key: str) -> str:
    """
    Retrieve the API key from Streamlit's session state or environment variables.

    Args:
        api_key_name (str): The name of the environment variable (e.g., 'OPENAI_API_KEY').
        session_state_key (str): The key used in Streamlit's session state (e.g., 'openai_api_key').

    Returns:
        str: The API key if found, otherwise an empty string.
    """
    # Attempt to retrieve the API key from session state
    api_key = st.session_state.get(session_state_key, "").strip()
    if not api_key:
        # Fallback to environment variable if not found in session state
        api_key = os.getenv(api_key_name, "").strip()
        if api_key:
            logging.info(f"Retrieved {api_key_name} from environment variables.")
        else:
            logging.warning(f"{api_key_name} not found in session state or environment variables.")
    else:
        logging.info(f"Retrieved {api_key_name} from session state.")

    return api_key