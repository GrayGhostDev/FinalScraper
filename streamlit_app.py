import streamlit as st
from streamlit_tags import st_tags_sidebar
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
    create_dynamic_listing_model,
    create_listings_container_model,
    scrape_url,
    scrape_multiple_urls,
    setup_selenium,
    generate_unique_folder_name,
    detect_pagination_elements
)
from pagination_detector import PaginationData
import re
from urllib.parse import urlparse
from assets import PRICING
import os
import logging

from api_management import get_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit_app.log"),
        logging.StreamHandler()
    ]
)

# Initialize Streamlit app
st.set_page_config(page_title="Universal Web Scraper", page_icon="🦑", layout="wide")
st.title("Universal Web Scraper 🦑")


# Initialize session state variables
def initialize_session_state():
    """
    Initialize all necessary session state variables with default values.
    """
    default_keys = {
        'scraping_state': 'idle',  # Possible states: 'idle', 'waiting', 'scraping', 'completed'
        'results': None,
        'driver': None,
        'openai_api_key': '',
        'gemini_api_key': '',
        'fields_input': '',
        # 'groq_api_key': '',            # Remove if Groq is no longer supported
    }

    for key, default in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default


# Call the initialization function
initialize_session_state()

# Sidebar components
st.sidebar.title("Web Scraper Settings")

# API Keys (Removed Groq API Key as it's no longer supported)
with st.sidebar.expander("API Keys", expanded=False):
    st.session_state['openai_api_key'] = st.text_input("OpenAI API Key", type="password", key='openai_key')
    st.session_state['gemini_api_key'] = st.text_input("Gemini API Key", type="password", key='gemini_key')
    # Removed Groq API Key input
    # st.session_state['groq_api_key'] = st.text_input("Groq API Key", type="password", key='groq_key')

# Model selection with descriptions
model_options = list(PRICING.keys())
model_descriptions = {
    "gpt-4o-mini": "OpenAI's GPT-4o-mini model",
    "gpt-4o-2024-08-06": "OpenAI's GPT-4o-2024-08-06 model",
    "gemini-1.5-flash": "Google Gemini Model",
    "Llama3.1 8B": "Local Llama 3.1 Model (8B parameters)"
    # "Groq Llama3.1 70b" has been removed as it's no longer supported
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=model_options,
    format_func=lambda x: f"{x} - {model_descriptions.get(x, '')}",
    index=0
)

# URL input with validation
url_input = st.sidebar.text_input("🔗 Enter URL(s) separated by whitespace", placeholder="https://example.com")

# Process URLs
urls = url_input.strip().split()
num_urls = len(urls)

# Fields to extract using Streamlit's native text_area
st.sidebar.markdown("### 📝 Enter Fields to Extract:")
fields_input = st.sidebar.text_area(
    "Enter each field on a new line:",
    placeholder="Name of item\nPrice",
    height=150,
    key='fields_input'
)
fields = [field.strip() for field in fields_input.split("\n") if field.strip()]

st.sidebar.markdown("---")

# Conditionally display Pagination and Attended Mode options
if num_urls <= 1:
    # Pagination settings
    use_pagination = st.sidebar.checkbox("Enable Pagination", value=False)
    pagination_details = ""
    if use_pagination:
        pagination_details = st.sidebar.text_input(
            "Enter Pagination Details (optional)",
            help="Describe how to navigate through pages (e.g., 'Next' button class, URL pattern)"
        )

    st.sidebar.markdown("---")

    # Attended mode toggle
    attended_mode = st.sidebar.checkbox("Enable Attended Mode", value=False)
else:
    # Multiple URLs entered; disable Pagination and Attended Mode
    use_pagination = False
    attended_mode = False
    # Inform the user
    st.sidebar.info("Pagination and Attended Mode are disabled when multiple URLs are entered.")

st.sidebar.markdown("---")

# Main action button
if st.sidebar.button("🚀 Scrape", type="primary", key="scrape_button"):
    if url_input.strip() == "":
        st.error("Please enter at least one URL.")
    elif num_urls <= 1 and use_pagination and pagination_details == "":
        st.error("Please provide pagination details or disable pagination.")
    elif num_urls <= 1 and len(fields) == 0 and use_pagination:
        st.error("Please enter at least one field to extract.")
    else:
        # Set up scraping parameters in session state
        st.session_state['urls'] = urls
        st.session_state['fields'] = fields
        st.session_state['model_selection'] = selected_model
        st.session_state['attended_mode'] = attended_mode
        st.session_state['use_pagination'] = use_pagination
        st.session_state['pagination_details'] = pagination_details
        st.session_state['scraping_state'] = 'waiting' if attended_mode else 'scraping'
        logging.info("Scraping initiated.")
        st.rerun()

# Scraping logic
if st.session_state['scraping_state'] == 'waiting':
    # Attended mode: set up driver and wait for user interaction
    if st.session_state['driver'] is None:
        try:
            st.session_state['driver'] = setup_selenium(attended_mode=True)
            st.session_state['driver'].get(st.session_state['urls'][0])
            st.info("Browser window has been opened in attended mode. Perform any required actions.")
            st.write("When ready, click the 'Resume Scraping' button below.")
        except Exception as e:
            st.error(f"Failed to initialize attended mode: {e}")
            logging.error(f"Failed to initialize attended mode: {e}")
            st.session_state['scraping_state'] = 'idle'

    if st.button("Resume Scraping"):
        st.session_state['scraping_state'] = 'scraping'
        logging.info("Resuming scraping after attended mode.")
        st.rerun()

elif st.session_state['scraping_state'] == 'scraping':
    with st.spinner('🔍 Scraping in progress...'):
        try:
            # Retrieve API keys from session state using the updated get_api_key function
            openai_api_key = get_api_key('OPENAI_API_KEY', 'openai_api_key')
            google_api_key = get_api_key('GOOGLE_API_KEY', 'gemini_api_key')
            # groq_api_key = get_api_key('GROQ_API_KEY', 'groq_api_key')  # Remove if Groq is no longer supported

            if not openai_api_key and not google_api_key:
                st.error("API keys are missing. Please provide them in the sidebar.")
                logging.error("API keys are missing.")
                st.session_state['scraping_state'] = 'idle'
                st.rerun()

            # Perform scraping
            if num_urls > 1:
                # Multiple URLs: use scrape_multiple_urls for efficiency
                output_folder, total_input_tokens, total_output_tokens, total_cost, all_data, markdown = scrape_multiple_urls(
                    st.session_state['urls'],
                    st.session_state['fields'],
                    st.session_state['model_selection'],
                    openai_api_key=openai_api_key,
                    google_api_key=google_api_key
                )
                pagination_info = None
            else:
                # Single URL: handle attended mode and pagination
                if st.session_state['attended_mode'] and st.session_state['driver']:
                    # Attended mode: scrape the current page without navigating
                    raw_html = fetch_html_selenium(
                        st.session_state['urls'][0],
                        attended_mode=True,
                        driver=st.session_state['driver']
                    )
                    markdown = html_to_markdown_with_readability(raw_html)
                    output_folder = os.path.join('output', generate_unique_folder_name(st.session_state['urls'][0]))
                    os.makedirs(output_folder, exist_ok=True)
                    save_raw_data(markdown, output_folder, 'rawData_1.md')

                    current_url = st.session_state[
                        'driver'].current_url  # Use the current URL for logging and saving purposes

                    # Detect pagination if enabled
                    pagination_info = None
                    if st.session_state['use_pagination']:
                        pagination_data, token_counts, pagination_price = detect_pagination_elements(
                            current_url,
                            st.session_state['pagination_details'],
                            st.session_state['model_selection'],
                            markdown,
                            openai_api_key=openai_api_key,
                            google_api_key=google_api_key
                        )
                        pagination_info = {
                            "page_urls": pagination_data.page_urls,
                            "token_counts": token_counts,
                            "price": pagination_price
                        }

                    # Scrape data if fields are specified
                    all_data = []
                    total_input_tokens = 0
                    total_output_tokens = 0
                    total_cost = 0.0

                    if len(st.session_state['fields']) > 0:
                        # Create dynamic models
                        DynamicListingModel = create_dynamic_listing_model(st.session_state['fields'])
                        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)

                        # Format data
                        formatted_data, token_counts = format_data(
                            markdown,
                            DynamicListingsContainer,
                            DynamicListingModel,
                            st.session_state['model_selection'],
                            openai_api_key=openai_api_key,
                            google_api_key=google_api_key
                        )
                        input_tokens, output_tokens, cost = calculate_price(token_counts,
                                                                            st.session_state['model_selection'])
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        total_cost += cost

                        # Save formatted data
                        save_formatted_data(
                            formatted_data,
                            output_folder,
                            'sorted_data_1.json',
                            'sorted_data_1.xlsx'
                        )
                        all_data.append(formatted_data)

                else:
                    # Non-attended mode: scrape without pagination
                    output_folder, total_input_tokens, total_output_tokens, total_cost, all_data, markdown = scrape_multiple_urls(
                        st.session_state['urls'],
                        st.session_state['fields'],
                        st.session_state['model_selection'],
                        openai_api_key=openai_api_key,
                        google_api_key=google_api_key
                    )
                    pagination_info = None
                    if st.session_state['use_pagination']:
                        # Detect pagination for the first URL
                        pagination_data, token_counts, pagination_price = detect_pagination_elements(
                            st.session_state['urls'][0],
                            st.session_state['pagination_details'],
                            st.session_state['model_selection'],
                            markdown,
                            openai_api_key=openai_api_key,
                            google_api_key=google_api_key
                        )
                        pagination_info = {
                            "page_urls": pagination_data.page_urls,
                            "token_counts": token_counts,
                            "price": pagination_price
                        }

            # Clean up driver if used
            if st.session_state['attended_mode'] and st.session_state['driver']:
                st.session_state['driver'].quit()
                st.session_state['driver'] = None
                logging.info("Closed Selenium WebDriver after scraping.")

            # Save results to session state
            st.session_state['results'] = {
                'data': all_data,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_cost': total_cost,
                'output_folder': output_folder,
                'pagination_info': pagination_info
            }
            st.session_state['scraping_state'] = 'completed'
            logging.info("Scraping completed successfully.")

        except Exception as e:
            st.error(f"An error occurred during scraping: {e}")
            logging.error(f"An error occurred during scraping: {e}")
            st.session_state['scraping_state'] = 'idle'

# Display results
if st.session_state['scraping_state'] == 'completed' and st.session_state['results']:
    results = st.session_state['results']
    all_data = results['data']
    total_input_tokens = results['input_tokens']
    total_output_tokens = results['output_tokens']
    total_cost = results['total_cost']
    output_folder = results['output_folder']
    pagination_info = results.get('pagination_info', None)

    if len(all_data) > 0:
        st.subheader("📊 Scraped Data")
        for i, data in enumerate(all_data, start=1):
            st.write(f"**Data from URL {i}:**")

            # Handle string data (convert to dict if it's JSON)
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    st.error(f"Failed to parse data as JSON for URL {i}")
                    continue

            if isinstance(data, dict):
                if 'listings' in data and isinstance(data['listings'], list):
                    df = pd.DataFrame(data['listings'])
                else:
                    # If 'listings' is not in the dict or not a list, use the entire dict
                    df = pd.DataFrame([data])
            elif hasattr(data, 'listings') and isinstance(data.listings, list):
                # Handle the case where data is a Pydantic model
                listings = [item.dict() for item in data.listings]
                df = pd.DataFrame(listings)
            else:
                st.error(f"Unexpected data format for URL {i}")
                continue

            # Display the dataframe
            st.dataframe(df, use_container_width=True)

        # Display scraping details in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Scraping Details")
        st.sidebar.markdown("#### Token Usage")
        st.sidebar.markdown(f"*Input Tokens:* {total_input_tokens}")
        st.sidebar.markdown(f"*Output Tokens:* {total_output_tokens}")
        st.sidebar.markdown(f"**Total Cost:** :green[**${total_cost:.4f}**]")

        # Download options
        st.subheader("💾 Download Extracted Data")
        col1, col2 = st.columns(2)
        with col1:
            json_data = json.dumps(all_data, default=lambda o: o.dict() if hasattr(o, 'dict') else str(o), indent=4)
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name=f"scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        with col2:
            # Convert all data to a single DataFrame
            all_listings = []
            for data in all_data:
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                if isinstance(data, dict) and 'listings' in data:
                    all_listings.extend(data['listings'])
                elif hasattr(data, 'listings'):
                    all_listings.extend([item.dict() for item in data.listings])
                else:
                    all_listings.append(data)

            if len(all_listings) > 0:
                combined_df = pd.DataFrame(all_listings)
                st.download_button(
                    "Download CSV",
                    data=combined_df.to_csv(index=False),
                    file_name=f"scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No listings data available to download as CSV.")

        st.success(f"Scraping completed successfully! Results saved in `{output_folder}`.")

    # Display pagination info
    if pagination_info:
        st.markdown("---")
        st.subheader("🔗 Pagination Information")

        # Display pagination details in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Pagination Details")
        st.sidebar.markdown(f"**Number of Page URLs:** {len(pagination_info['page_urls'])}")
        st.sidebar.markdown("#### Pagination Token Usage")
        st.sidebar.markdown(f"*Input Tokens:* {pagination_info['token_counts']['input_tokens']}")
        st.sidebar.markdown(f"*Output Tokens:* {pagination_info['token_counts']['output_tokens']}")
        st.sidebar.markdown(f"**Pagination Cost:** :blue[**${pagination_info['price']:.4f}**]")

        # Display page URLs in a table with clickable links
        st.write("**Page URLs:**")
        pagination_df = pd.DataFrame(pagination_info["page_urls"], columns=["Page URLs"])

        st.dataframe(
            pagination_df,
            column_config={
                "Page URLs": st.column_config.LinkColumn("Page URLs")
            },
            use_container_width=True
        )

        # Download pagination URLs
        st.subheader("💾 Download Pagination URLs")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Pagination JSON",
                data=json.dumps(pagination_info["page_urls"], indent=4),
                file_name=f"pagination_urls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        with col2:
            st.download_button(
                "Download Pagination CSV",
                data=pagination_df.to_csv(index=False),
                file_name=f"pagination_urls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # Display combined totals if both scraping and pagination were performed
        if len(all_data) > 0 and pagination_info and st.session_state['use_pagination']:
            st.markdown("---")
            total_combined_input_tokens = total_input_tokens + pagination_info['token_counts']['input_tokens']
            total_combined_output_tokens = total_output_tokens + pagination_info['token_counts']['output_tokens']
            total_combined_cost = total_cost + pagination_info['price']
            st.markdown("### Total Counts and Cost (Including Pagination)")
            st.markdown(f"**Total Input Tokens:** {total_combined_input_tokens}")
            st.markdown(f"**Total Output Tokens:** {total_combined_output_tokens}")
            st.markdown(f"**Total Combined Cost:** :rainbow[**${total_combined_cost:.4f}**]")

    # Add a clear results button
    if st.sidebar.button("🚀 Scrape", type="primary", key="clear_results_button"):
        if url_input.strip() == "":
            st.error("Please enter at least one URL.")
        elif num_urls <= 1 and use_pagination and pagination_details == "":
            st.error("Please provide pagination details or disable pagination.")
        elif num_urls <= 1 and len(fields) == 0 and use_pagination:
            st.error("Please enter at least one field to extract.")
        else:
            # Retrieve API keys using get_api_key with both required arguments
            openai_api_key = get_api_key('OPENAI_API_KEY', 'openai_api_key')
            google_api_key = get_api_key('GOOGLE_API_KEY', 'gemini_api_key')
            # groq_api_key = get_api_key('GROQ_API_KEY', 'groq_api_key')  # Remove if Groq is no longer supported

            if not openai_api_key and not google_api_key:
                st.error("API keys are missing. Please provide them in the sidebar.")
                logging.error("API keys are missing.")
            else:
                # Set up scraping parameters in session state
                st.session_state['urls'] = urls
                st.session_state['fields'] = fields
                st.session_state['model_selection'] = selected_model
                st.session_state['attended_mode'] = attended_mode
                st.session_state['use_pagination'] = use_pagination
                st.session_state['pagination_details'] = pagination_details
                st.session_state['scraping_state'] = 'waiting' if attended_mode else 'scraping'
                logging.info("Scraping initiated.")
                st.rerun()  # Correct usage