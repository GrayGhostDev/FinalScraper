import streamlit as st
import pandas as pd
import json
from datetime import datetime
from scraper import (
    fetch_html_selenium,
    save_raw_data,
    format_data,
    save_formatted_data,
    calculate_price,
    html_to_markdown_with_readability,
)
import sys
st.write(f"Python executable: {sys.executable}")
st.write(f"Python version: {sys.version}")

# Initialize Streamlit app
st.set_page_config(page_title="Universal Web Scraper")
st.title("Universal Web Scraper 🦑")

# Sidebar components
st.sidebar.title("Web Scraper Settings")

# Model selection
model_options = ["gpt-3.5-turbo-0613", "gpt-4-0613"]
model_selection = st.sidebar.selectbox("Select Model", options=model_options, index=0)

# URL input
url_input = st.sidebar.text_input("Enter URL", value="https://news.ycombinator.com/")

# Fields input
fields_input = st.sidebar.text_area(
    "Enter Fields to Extract (one per line):",
    value="Title\nNumber of Points\nCreator\nTime Posted\nNumber of Comments",
    height=150,
)

st.sidebar.markdown("---")

# Process fields into a list
fields = [field.strip() for field in fields_input.split("\n") if field.strip()]

# Initialize variables to store token and cost information
input_tokens = output_tokens = total_cost = 0

# Define the scraping function
def perform_scrape(url, fields, model_used):
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_html = fetch_html_selenium(url)
        markdown = html_to_markdown_with_readability(raw_html)
        save_raw_data(markdown, timestamp)
        formatted_data = format_data(markdown, fields, model_used)
        formatted_data_text = json.dumps(formatted_data)
        input_tokens, output_tokens, total_cost = calculate_price(
            markdown, formatted_data_text, model=model_used
        )
        df = save_formatted_data(formatted_data, timestamp)
        return df, formatted_data, markdown, input_tokens, output_tokens, total_cost, timestamp
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None, None, 0, 0, 0, None

# Handling button press for scraping
if 'perform_scrape' not in st.session_state:
    st.session_state['perform_scrape'] = False

if st.sidebar.button("Scrape"):
    if not url_input.strip():
        st.sidebar.error("Please enter a valid URL.")
    elif not fields:
        st.sidebar.error("Please enter at least one field to extract.")
    else:
        with st.spinner('Please wait... Data is being scraped.'):
            results = perform_scrape(url_input, fields, model_selection)
            if results[0] is not None:
                st.session_state['results'] = results
                st.session_state['perform_scrape'] = True
            else:
                st.session_state['perform_scrape'] = False

if st.session_state.get('perform_scrape'):
    (
        df,
        formatted_data,
        markdown,
        input_tokens,
        output_tokens,
        total_cost,
        timestamp,
    ) = st.session_state['results']
    if df is not None:
        # Display the DataFrame and other data
        st.subheader("Scraped Data")
        st.dataframe(df)

        st.sidebar.markdown("## Token Usage")
        st.sidebar.markdown(f"**Input Tokens:** {input_tokens}")
        st.sidebar.markdown(f"**Output Tokens:** {output_tokens}")
        st.sidebar.markdown(f"**Estimated Total Cost:** :green[$${total_cost:.4f}$]")

        # Create columns for download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download JSON",
                data=json.dumps(formatted_data, indent=4),
                file_name=f"{timestamp}_data.json",
            )
        with col2:
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False),
                file_name=f"{timestamp}_data.csv",
            )
        with col3:
            st.download_button(
                "Download Markdown",
                data=markdown,
                file_name=f"{timestamp}_data.md",
            )
