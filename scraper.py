import os
import time
import re
import json
from datetime import datetime
from typing import List

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

import html2text
import tiktoken
import openai

load_dotenv()


def setup_selenium():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
    service = Service(r"./chromedriver-win64/chromedriver.exe")

    driver = webdriver.Chrome(service=service, options=options)
    return driver


def fetch_html_selenium(url):
    driver = setup_selenium()
    try:
        driver.get(url)
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        html = driver.page_source
        return html
    finally:
        driver.quit()


def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup.find_all(['header', 'footer']):
        element.decompose()
    return str(soup)


def html_to_markdown_with_readability(html_content):
    cleaned_html = clean_html(html_content)
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    markdown_content = markdown_converter.handle(cleaned_html)
    return markdown_content


# Define the pricing for the models
pricing = {
    "gpt-3.5-turbo-0613": {
        "input": 0.0015 / 1000,
        "output": 0.002 / 1000,
    },
    "gpt-4-0613": {
        "input": 0.03 / 1000,
        "output": 0.06 / 1000,
    },
}

model_used = "gpt-3.5-turbo-0613"


def save_raw_data(raw_data, timestamp, output_folder='output'):
    os.makedirs(output_folder, exist_ok=True)
    raw_output_path = os.path.join(output_folder, f'rawData_{timestamp}.md')
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)
    print(f"Raw data saved to {raw_output_path}")
    return raw_output_path


def trim_to_token_limit(text, model, max_tokens=200000):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        trimmed_text = encoder.decode(tokens[:max_tokens])
        return trimmed_text
    return text


def format_data(data):
    openai.api_key = os.getenv('OPENAI_API_KEY')

    system_message = (
        "You are an intelligent text extraction and conversion assistant. "
        "Your task is to extract structured information from the given text and "
        "convert it into a JSON format matching the specified schema. The JSON should "
        "contain only the structured data extracted from the text, with no additional "
        "commentary, explanations, or extraneous information."
    )

    user_message = f"Extract the following information from the provided text:\n\n{data}"

    functions = [
        {
            "name": "extract_listings",
            "description": "Extracts listings from the page content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Title": {"type": "string"},
                                "Number of Points": {"type": "string"},
                                "Creator": {"type": "string"},
                                "Time Posted": {"type": "string"},
                                "Number of Comments": {"type": "string"},
                            },
                            "required": [
                                "Title",
                                "Number of Points",
                                "Creator",
                                "Time Posted",
                                "Number of Comments",
                            ],
                        },
                    }
                },
                "required": ["listings"],
            },
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model_used,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            functions=functions,
            function_call={"name": "extract_listings"},
        )

        message = response["choices"][0]["message"]

        if 'function_call' in message:
            function_call = message['function_call']
            function_args = function_call.get('arguments')
            extracted_data = json.loads(function_args)
            return extracted_data
        else:
            raise Exception("No function call in response")
    except Exception as e:
        print(f"Error in format_data: {e}")
        raise


def save_formatted_data(formatted_data, timestamp, output_folder='output'):
    os.makedirs(output_folder, exist_ok=True)
    json_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4)
    print(f"Formatted data saved to JSON at {json_output_path}")

    if 'listings' in formatted_data:
        data_for_df = formatted_data['listings']
    else:
        raise ValueError("Formatted data does not contain 'listings' key.")

    try:
        df = pd.DataFrame(data_for_df)
        print("DataFrame created successfully.")
        excel_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.xlsx')
        df.to_excel(excel_output_path, index=False)
        print(f"Formatted data saved to Excel at {excel_output_path}")
        return df
    except Exception as e:
        print(f"Error creating DataFrame or saving Excel: {str(e)}")
        return None


def calculate_price(input_text, output_text, model=model_used):
    encoder = tiktoken.encoding_for_model(model)
    input_token_count = len(encoder.encode(input_text))
    output_token_count = len(encoder.encode(output_text))
    input_cost = input_token_count * pricing[model]["input"]
    output_cost = output_token_count * pricing[model]["output"]
    total_cost = input_cost + output_cost
    return input_token_count, output_token_count, total_cost


if __name__ == "__main__":
    url = 'https://news.ycombinator.com/'

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_html = fetch_html_selenium(url)
        markdown = html_to_markdown_with_readability(raw_html)
        raw_data_path = save_raw_data(markdown, timestamp)

        # Optionally remove URLs from the markdown content
        # cleaned_content = remove_urls_from_file(raw_data_path)
        # For now, we'll use the original markdown content

        formatted_data = format_data(markdown)
        df = save_formatted_data(formatted_data, timestamp)

        formatted_data_text = json.dumps(formatted_data)

        input_tokens, output_tokens, total_cost = calculate_price(
            markdown, formatted_data_text, model=model_used
        )
        print(f"Input token count: {input_tokens}")
        print(f"Output token count: {output_tokens}")
        print(f"Estimated total cost: ${total_cost:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")