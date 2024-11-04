import os
import json
import logging
from typing import List, Dict, Tuple, Union, Optional

from pydantic import BaseModel, Field

import tiktoken
from dotenv import load_dotenv

from openai import OpenAI
import google.generativeai as genai

from api_management import get_api_key
from assets import PROMPT_PAGINATION, PRICING, LLAMA_MODEL_FULLNAME

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pagination_detector.log"),
        logging.StreamHandler()
    ]
)


# Define the PaginationData model
class PaginationData(BaseModel):
    page_urls: List[str] = Field(
        default_factory=list,
        description="List of pagination URLs, including 'Next' button URL if present"
    )


def calculate_pagination_price(token_counts: Dict[str, int], model: str) -> float:
    """
    Calculate the price for pagination based on token counts and the selected model.

    Args:
        token_counts (Dict[str, int]): A dictionary containing 'input_tokens' and 'output_tokens'.
        model (str): The name of the selected model.

    Returns:
        float: The total price for the pagination operation.
    """
    input_tokens = token_counts.get('input_tokens', 0)
    output_tokens = token_counts.get('output_tokens', 0)

    try:
        input_price = input_tokens * PRICING[model]['input']
        output_price = output_tokens * PRICING[model]['output']
        total_price = input_price + output_price
        logging.info(
            f"Calculated pagination cost: Input Tokens={input_tokens}, Output Tokens={output_tokens}, Total Cost=${total_price:.6f}")
        return total_price
    except KeyError as e:
        logging.error(f"Pricing information missing for model {model}: {e}")
        raise


def detect_pagination_elements(
        url: str,
        indications: str,
        selected_model: str,
        markdown_content: str
) -> Tuple[Union[PaginationData, Dict, str], Dict[str, int], float]:
    """
    Uses AI models to analyze markdown content and extract pagination elements.

    Args:
        url (str): The URL of the page to extract pagination from.
        indications (str): User-provided indications for pagination detection.
        selected_model (str): The name of the selected AI model.
        markdown_content (str): The markdown content of the webpage.

    Returns:
        Tuple[Union[PaginationData, Dict, str], Dict[str, int], float]:
            - Parsed pagination data.
            - Token counts.
            - Pagination price.
    """
    try:
        # Construct the prompt for pagination detection
        prompt_pagination = (
            f"{PROMPT_PAGINATION}\n"
            f"The URL of the page to extract pagination from: {url}. "
            "If the URLs that you find are not complete, combine them intelligently in a way that fits the pattern. "
            "**ALWAYS GIVE A FULL URL**."
        )

        if indications:
            prompt_pagination += (
                f"\n\nThese are the user's indications that you should pay special attention to: {indications}\n\n"
                "Below is the markdown content of the website:\n\n"
            )
        else:
            prompt_pagination += (
                "\n\nThere are no user indications in this case. Just apply the logic described above.\n\n"
                "Below is the markdown content of the website:\n\n"
            )

        prompt_pagination += markdown_content

        logging.info(f"Constructed prompt for pagination detection using model {selected_model}.")

        # Initialize AI client based on the selected model
        if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
            # Use OpenAI API
            openai_client = OpenAI(api_key=get_api_key('OPENAI_API_KEY'))
            messages = [
                {"role": "system", "content": PROMPT_PAGINATION},
                {"role": "user", "content": markdown_content},
            ]
            response = openai_client.ChatCompletion.create(
                model=selected_model,
                messages=messages,
                temperature=0.7,
            )
            content = response.choices[0].message.content
            parsed_data = json.loads(content)
            token_counts = {
                "input_tokens": response['usage']['prompt_tokens'],
                "output_tokens": response['usage']['completion_tokens']
            }
            pagination_price = calculate_pagination_price(token_counts, selected_model)
            pagination_data = PaginationData(**parsed_data)
            logging.info(f"Pagination detected using OpenAI model {selected_model}: {pagination_data.page_urls}")
            return pagination_data, token_counts, pagination_price

        elif selected_model == "gemini-1.5-flash":
            # Use Google Gemini API
            genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": PaginationData
                }
            )
            prompt = prompt_pagination
            input_tokens = model.count_tokens(prompt)
            completion = model.generate_content(prompt)
            usage_metadata = completion.usage_metadata
            token_counts = {
                "input_tokens": usage_metadata.prompt_token_count,
                "output_tokens": usage_metadata.candidates_token_count
            }
            try:
                parsed_data = json.loads(completion.text)
                pagination_data = PaginationData(**parsed_data)
                logging.info(f"Pagination detected using Google Gemini model: {pagination_data.page_urls}")
            except json.JSONDecodeError:
                logging.error("Failed to parse Gemini Flash response as JSON.")
                pagination_data = PaginationData(page_urls=[])

            pagination_price = calculate_pagination_price(token_counts, selected_model)
            return pagination_data, token_counts, pagination_price

        elif selected_model == "Llama3.1 8B":
            # Use Local Llama Model via OpenAI API pointing to a local server
            openai.api_key = "lm-studio"
            openai.api_base = "http://localhost:1234/v1"
            messages = [
                {"role": "system", "content": PROMPT_PAGINATION},
                {"role": "user", "content": markdown_content},
            ]
            response = openai.ChatCompletion.create(
                model=LLAMA_MODEL_FULLNAME,
                messages=messages,
                temperature=0.7,
            )
            response_content = response['choices'][0]['message']['content'].strip()
            try:
                parsed_data = json.loads(response_content)
                pagination_data = PaginationData(**parsed_data)
                logging.info(f"Pagination detected using Local Llama model: {pagination_data.page_urls}")
            except json.JSONDecodeError:
                logging.error("Failed to parse Llama3.1 8B response as JSON.")
                pagination_data = PaginationData(page_urls=[])

            token_counts = {
                "input_tokens": response['usage']['prompt_tokens'],
                "output_tokens": response['usage']['completion_tokens']
            }
            pagination_price = calculate_pagination_price(token_counts, selected_model)
            return pagination_data, token_counts, pagination_price

        else:
            raise ValueError(f"Unsupported model: {selected_model}")

    except Exception as e:
        logging.error(f"An error occurred in detect_pagination_elements: {e}")
        # Return default values if an error occurs
        return PaginationData(page_urls=[]), {"input_tokens": 0, "output_tokens": 0}, 0.0