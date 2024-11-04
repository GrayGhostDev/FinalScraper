import os
import random
import time
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Type, Tuple, Optional, Union

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text
import tiktoken
import streamlit as st

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from openai import OpenAI
import google.generativeai as genai

from api_management import get_api_key
from assets import (
    PROMPT_PAGINATION,
    PRICING,
    LLAMA_MODEL_FULLNAME,
    USER_AGENTS,
    HEADLESS_OPTIONS,
    HEADLESS_OPTIONS_DOCKER,
    TIMEOUT_SETTINGS,
    NUMBER_SCROLL,
    SYSTEM_MESSAGE,
    USER_MESSAGE
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)


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
        markdown_content: str,
        openai_api_key: str,
        google_api_key: str
) -> Tuple[Union[PaginationData, Dict, str], Dict[str, int], float]:
    """
    Uses AI models to analyze markdown content and extract pagination elements.

    Args:
        url (str): The URL of the page to extract pagination from.
        indications (str): User-provided indications for pagination detection.
        selected_model (str): The name of the selected AI model.
        markdown_content (str): The markdown content of the webpage.
        openai_api_key (str): OpenAI API key.
        google_api_key (str): Google Gemini API key.

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
            openai_client = OpenAI(api_key=openai_api_key)
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
            genai.configure(api_key=google_api_key)
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
            openai_client = OpenAI(api_key="lm-studio", api_base="http://localhost:1234/v1")
            messages = [
                {"role": "system", "content": PROMPT_PAGINATION},
                {"role": "user", "content": markdown_content},
            ]
            response = openai_client.ChatCompletion.create(
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


def is_running_in_docker() -> bool:
    """
    Detect if the app is running inside a Docker container.
    This checks if the '/proc/1/cgroup' file contains 'docker'.
    """
    try:
        with open("/proc/1/cgroup", "rt") as file:
            return "docker" in file.read()
    except Exception as e:
        logging.warning(f"Unable to determine Docker environment: {e}")
        return False


def setup_selenium(attended_mode: bool = False) -> webdriver.Chrome:
    """
    Sets up the Selenium WebDriver with Chrome options and manages ChromeDriver using webdriver-manager.

    Args:
        attended_mode (bool): If True, runs the browser in non-headless mode for debugging purposes.

    Returns:
        webdriver.Chrome: Configured Selenium WebDriver instance.
    """
    options = Options()
    service = Service(ChromeDriverManager().install())

    # Apply headless options based on whether the code is running in Docker
    if is_running_in_docker():
        # Running inside Docker, use Docker-specific headless options
        for option in HEADLESS_OPTIONS_DOCKER:
            options.add_argument(option)
    else:
        # Not running inside Docker, use the normal headless options
        for option in HEADLESS_OPTIONS:
            options.add_argument(option)

    if attended_mode:
        options.headless = False  # Run in non-headless mode for debugging

    # Initialize the WebDriver
    try:
        driver = webdriver.Chrome(service=service, options=options)
        logging.info("Selenium WebDriver initialized successfully.")
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize Selenium WebDriver: {e}")
        raise


def fetch_html_selenium(url: str, attended_mode: bool = False, driver: Optional[webdriver.Chrome] = None) -> str:
    """
    Fetches the HTML content of the given URL using Selenium.

    Args:
        url (str): The URL to scrape.
        attended_mode (bool): If True, keeps the browser open for debugging.
        driver (webdriver.Chrome, optional): Pre-initialized WebDriver instance.

    Returns:
        str: The page source HTML.
    """
    if driver is None:
        driver = setup_selenium(attended_mode)
        should_quit = True
    else:
        should_quit = False
        if not attended_mode:
            driver.get(url)

    try:
        driver.get(url)
        logging.info(f"Navigated to URL: {url}")

        # Wait until the page is fully loaded
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        logging.info("Page loaded successfully.")

        if not attended_mode:
            # Add realistic actions like scrolling to mimic human behavior
            actions = ActionChains(driver)
            for scroll_height in [0.5, 1.2, 1]:
                actions.move_to_element(driver.find_element(By.TAG_NAME, "body"))
                driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {scroll_height});")
                logging.debug(f"Scrolled to {scroll_height * 100}% of the page.")
                time.sleep(random.uniform(1.1, 1.8))

        html = driver.page_source
        logging.info("Fetched page source successfully.")
        return html
    except Exception as e:
        logging.error(f"Error fetching HTML from {url}: {e}")
        raise
    finally:
        if should_quit:
            driver.quit()
            logging.info("Selenium WebDriver closed.")


def clean_html(html_content: str) -> str:
    """
    Cleans the HTML content by removing header and footer elements.

    Args:
        html_content (str): Raw HTML content.

    Returns:
        str: Cleaned HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove headers and footers based on common HTML tags or classes
    for element in soup.find_all(['header', 'footer']):
        element.decompose()  # Remove these tags and their content
        logging.debug(f"Removed element: {element.name}")

    logging.info("Cleaned HTML content by removing headers and footers.")
    return str(soup)


def html_to_markdown_with_readability(html_content: str) -> str:
    """
    Converts cleaned HTML content to Markdown format.

    Args:
        html_content (str): Cleaned HTML content.

    Returns:
        str: Converted Markdown content.
    """
    cleaned_html = clean_html(html_content)

    # Convert to markdown
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    markdown_content = markdown_converter.handle(cleaned_html)

    logging.info("Converted HTML content to Markdown.")
    return markdown_content


def save_raw_data(raw_data: str, output_folder: str, file_name: str) -> str:
    """
    Save raw markdown data to the specified output folder.

    Args:
        raw_data (str): Raw markdown content.
        output_folder (str): Directory to save the file.
        file_name (str): Name of the file.

    Returns:
        str: Path to the saved raw data file.
    """
    os.makedirs(output_folder, exist_ok=True)
    raw_output_path = os.path.join(output_folder, file_name)
    try:
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            f.write(raw_data)
        logging.info(f"Raw data saved to {raw_output_path}")
        return raw_output_path
    except Exception as e:
        logging.error(f"Failed to save raw data to {raw_output_path}: {e}")
        raise


def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model based on provided fields.

    Args:
        field_names (List[str]): List of field names to extract.

    Returns:
        Type[BaseModel]: Dynamically created Pydantic model.
    """
    field_definitions = {field: (str, ...) for field in field_names}
    DynamicListingModel = create_model('DynamicListingModel', **field_definitions)
    logging.info(f"Created dynamic listing model with fields: {field_names}")
    return DynamicListingModel


def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a container model that holds a list of the given listing model.

    Args:
        listing_model (Type[BaseModel]): Pydantic model for individual listings.

    Returns:
        Type[BaseModel]: Container Pydantic model.
    """
    DynamicListingsContainer = create_model('DynamicListingsContainer', listings=(List[listing_model], ...))
    logging.info("Created dynamic listings container model.")
    return DynamicListingsContainer


def trim_to_token_limit(text: str, model: str, max_tokens: int = 120000) -> str:
    """
    Trims the input text to a maximum number of tokens to prevent exceeding model limits.

    Args:
        text (str): The text to trim.
        model (str): The model name to determine encoding.
        max_tokens (int, optional): Maximum number of tokens. Defaults to 120000.

    Returns:
        str: Trimmed text if necessary.
    """
    try:
        encoder = tiktoken.encoding_for_model(model)
        tokens = encoder.encode(text)
        if len(tokens) > max_tokens:
            trimmed_text = encoder.decode(tokens[:max_tokens])
            logging.warning(f"Trimmed text to {max_tokens} tokens.")
            return trimmed_text
        return text
    except Exception as e:
        logging.error(f"Error trimming to token limit: {e}")
        return text


def generate_system_message(listing_model: BaseModel) -> str:
    """
    Dynamically generate a system message based on the fields in the provided listing model.

    Args:
        listing_model (BaseModel): Pydantic model for listings.

    Returns:
        str: Generated system message.
    """
    schema_info = listing_model.model_json_schema()

    field_descriptions = []
    for field_name, field_info in schema_info["properties"].items():
        field_type = field_info["type"]
        field_descriptions.append(f'"{field_name}": "{field_type}"')

    schema_structure = ",\n                ".join(field_descriptions)

    system_message = f"""
    You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
    from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
    with no additional commentary, explanations, or extraneous information. 
    You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
    Please process the following text and provide the output in pure JSON format with no words before or after the JSON:
    Please ensure the output strictly follows this schema:

    {{
        "listings": [
            {{
                {schema_structure}
            }}
        ]
    }} """

    logging.info("Generated system message for AI model.")
    return system_message


def format_data(
        data: str,
        DynamicListingsContainer: Type[BaseModel],
        DynamicListingModel: Type[BaseModel],
        selected_model: str,
        openai_api_key: str,
        google_api_key: str
) -> Tuple[Optional[Dict], Dict[str, int]]:
    """
    Formats the scraped Markdown data into structured JSON using the selected model's API.

    Args:
        data (str): Markdown content.
        DynamicListingsContainer (Type[BaseModel]): Pydantic container model.
        DynamicListingModel (Type[BaseModel]): Pydantic listing model.
        selected_model (str): Selected AI model.
        openai_api_key (str): OpenAI API key.
        google_api_key (str): Google Gemini API key.

    Returns:
        Tuple[Optional[Dict], Dict[str, int]]: Parsed data and token counts.
    """
    token_counts = {}

    try:
        if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
            # Use OpenAI API
            openai_client = OpenAI(api_key=openai_api_key)
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE + data},
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
            logging.info(f"Formatted data using OpenAI model: {selected_model}")
            return parsed_data, token_counts

        elif selected_model == "gemini-1.5-flash":
            # Use Google Gemini API
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": DynamicListingsContainer
                }
            )
            prompt = SYSTEM_MESSAGE + "\n" + USER_MESSAGE + data
            input_tokens = model.count_tokens(prompt)
            completion = model.generate_content(prompt)
            usage_metadata = completion.usage_metadata
            token_counts = {
                "input_tokens": usage_metadata.prompt_token_count,
                "output_tokens": usage_metadata.candidates_token_count
            }
            parsed_data = json.loads(completion.text)
            logging.info("Formatted data using Google Gemini model.")
            return parsed_data, token_counts

        elif selected_model == "Llama3.1 8B":
            # Use Local Llama Model via OpenAI API pointing to a local server
            sys_message = generate_system_message(DynamicListingModel)
            openai_client = OpenAI(api_key=openai_api_key, api_base="http://localhost:1234/v1")

            response = openai_client.ChatCompletion.create(
                model=LLAMA_MODEL_FULLNAME,
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": USER_MESSAGE + data}
                ],
                temperature=0.7,
            )

            response_content = response.choices[0].message.content
            parsed_response = json.loads(response_content)

            token_counts = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }

            logging.info("Formatted data using Local Llama model.")
            return parsed_response, token_counts

        else:
            raise ValueError(f"Unsupported model: {selected_model}")

    except Exception as e:
        logging.error(f"Error in format_data: {e}")
        raise


def save_formatted_data(
        formatted_data: Optional[Dict],
        output_folder: str,
        json_file_name: str,
        excel_file_name: str
) -> Optional[pd.DataFrame]:
    """
    Saves the formatted JSON data and converts it to Excel format.

    Args:
        formatted_data (Optional[Dict]): Parsed JSON data.
        output_folder (str): Directory to save the files.
        json_file_name (str): Name of the JSON file.
        excel_file_name (str): Name of the Excel file.

    Returns:
        Optional[pd.DataFrame]: DataFrame if successful, else None.
    """
    if not formatted_data:
        logging.warning("No formatted data to save.")
        return None

    os.makedirs(output_folder, exist_ok=True)

    try:
        # Save the formatted data as JSON
        json_output_path = os.path.join(output_folder, json_file_name)
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=4)
        logging.info(f"Formatted data saved to JSON at {json_output_path}")
    except Exception as e:
        logging.error(f"Failed to save formatted data to JSON: {e}")
        raise

    # Prepare data for DataFrame
    try:
        if isinstance(formatted_data, dict):
            # If the data is a dictionary containing lists, assume these lists are records
            data_for_df = next(iter(formatted_data.values())) if len(formatted_data) == 1 else formatted_data
        elif isinstance(formatted_data, list):
            data_for_df = formatted_data
        else:
            raise ValueError("Formatted data is neither a dictionary nor a list, cannot convert to DataFrame")

        df = pd.DataFrame(data_for_df)
        logging.info("DataFrame created successfully.")
    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        return None

    try:
        # Save the DataFrame to an Excel file
        excel_output_path = os.path.join(output_folder, excel_file_name)
        df.to_excel(excel_output_path, index=False)
        logging.info(f"Formatted data saved to Excel at {excel_output_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to save DataFrame to Excel: {e}")
        return None


def calculate_price(token_counts: Dict[str, int], model: str) -> Tuple[int, int, float]:
    """
    Calculates the token usage and estimated cost based on input and output tokens.

    Args:
        token_counts (Dict[str, int]): Dictionary containing 'input_tokens' and 'output_tokens'.
        model (str): Selected AI model.

    Returns:
        Tuple[int, int, float]: Input tokens, output tokens, and total cost.
    """
    input_token_count = token_counts.get("input_tokens", 0)
    output_token_count = token_counts.get("output_tokens", 0)

    try:
        input_cost = input_token_count * PRICING[model]["input"]
        output_cost = output_token_count * PRICING[model]["output"]
        total_cost = input_cost + output_cost
        logging.info(
            f"Calculated cost for model {model}: Input Tokens={input_token_count}, Output Tokens={output_token_count}, Total Cost=${total_cost:.6f}")
        return input_token_count, output_token_count, total_cost
    except KeyError as e:
        logging.error(f"Pricing information missing for model {model}: {e}")
        raise


def generate_unique_folder_name(url: str) -> str:
    """
    Generates a unique folder name based on the URL and current timestamp.

    Args:
        url (str): The URL being scraped.

    Returns:
        str: Unique folder name.
    """
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    try:
        url_name = re.sub(r'\W+', '_', urlparse(url).netloc)
    except Exception as e:
        logging.warning(f"Error parsing URL for folder name: {e}")
        url_name = "scrape"
    unique_folder = f"{url_name}_{timestamp}"
    logging.info(f"Generated unique folder name: {unique_folder}")
    return unique_folder


def scrape_url(
        url: str,
        fields: List[str],
        selected_model: str,
        output_folder: str,
        file_number: int,
        markdown: str,
        openai_api_key: str,
        google_api_key: str
) -> Tuple[int, int, float, Optional[Dict]]:
    """
    Scrapes a single URL and saves the results.

    Args:
        url (str): The URL to scrape.
        fields (List[str]): Fields to extract.
        selected_model (str): Selected AI model.
        output_folder (str): Directory to save the results.
        file_number (int): Identifier for the file.
        markdown (str): Markdown content of the page.
        openai_api_key (str): OpenAI API key.
        google_api_key (str): Google Gemini API key.

    Returns:
        Tuple[int, int, float, Optional[Dict]]: Input tokens, output tokens, total cost, and formatted data.
    """
    try:
        # Save raw data
        raw_file_name = f'rawData_{file_number}.md'
        save_raw_data(markdown, output_folder, raw_file_name)

        # Create the dynamic listing model
        DynamicListingModel = create_dynamic_listing_model(fields)

        # Create the container model that holds a list of the dynamic listing models
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)

        # Format data
        formatted_data, token_counts = format_data(
            markdown,
            DynamicListingsContainer,
            DynamicListingModel,
            selected_model,
            openai_api_key,
            google_api_key
        )

        # Save formatted data
        json_file_name = f'sorted_data_{file_number}.json'
        excel_file_name = f'sorted_data_{file_number}.xlsx'
        save_formatted_data(
            formatted_data,
            output_folder,
            json_file_name,
            excel_file_name
        )

        # Calculate and return token usage and cost
        input_tokens, output_tokens, total_cost = calculate_price(token_counts, selected_model)
        return input_tokens, output_tokens, total_cost, formatted_data

    except Exception as e:
        logging.error(f"An error occurred while processing {url}: {e}")
        return 0, 0, 0.0, None


def scrape_multiple_urls(
        urls: List[str],
        fields: List[str],
        selected_model: str,
        openai_api_key: str,
        google_api_key: str
) -> Tuple[str, int, int, float, List[Dict], Optional[str]]:
    """
    Scrapes multiple URLs and aggregates the results.

    Args:
        urls (List[str]): List of URLs to scrape.
        fields (List[str]): Fields to extract.
        selected_model (str): Selected AI model.
        openai_api_key (str): OpenAI API key.
        google_api_key (str): Google Gemini API key.

    Returns:
        Tuple[str, int, int, float, List[Dict], Optional[str]]:
            - output_folder (str)
            - total_input_tokens (int)
            - total_output_tokens (int)
            - total_cost (float)
            - all_data (List[Dict])
            - markdown (Optional[str])
    """
    if not urls:
        logging.warning("No URLs provided for scraping.")
        raise ValueError("No URLs provided for scraping.")

    output_folder = os.path.join('output', generate_unique_folder_name(urls[0]))
    os.makedirs(output_folder, exist_ok=True)

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    all_data = []
    markdown = None  # Store markdown for the first (or only) URL

    driver = setup_selenium()

    for i, url in enumerate(urls, start=1):
        try:
            logging.info(f"Starting scrape for URL {i}: {url}")
            raw_html = fetch_html_selenium(url, driver=driver)
            current_markdown = html_to_markdown_with_readability(raw_html)
            if i == 1:
                markdown = current_markdown  # Store markdown for the first URL

            input_tokens, output_tokens, cost, formatted_data = scrape_url(
                url,
                fields,
                selected_model,
                output_folder,
                i,
                current_markdown,
                openai_api_key,
                google_api_key
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost += cost
            if formatted_data:
                all_data.append(formatted_data)
            logging.info(f"Completed scrape for URL {i}: {url}")
        except Exception as e:
            logging.error(f"Failed to scrape URL {i}: {url} - {e}")
            continue

    if driver:
        driver.quit()
        logging.info("Selenium WebDriver closed after scraping multiple URLs.")

    return output_folder, total_input_tokens, total_output_tokens, total_cost, all_data, markdown