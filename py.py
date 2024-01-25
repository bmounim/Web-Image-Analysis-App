import streamlit as st

import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image


import xlsxwriter


import vertexai
from vertexai.language_models import TextGenerationModel
import pandas as pd
from image_analysis import analyze_image_for_criteria


import streamlit as st
import pandas as pd
import os
import io

from selenium import webdriver
#from PIL import Image

from google.cloud import vision
from selenium import webdriver


import vertexai
from vertexai.language_models import TextGenerationModel

from google.api_core.client_options import ClientOptions

# Set endpoint to EU 
options = ClientOptions(api_endpoint="eu-documentai.googleapis.com:443")

# Add any other specific imports you need for Vertex AI image analysis
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GCP_key.json"

# Define Chrome options for headless mode
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Initialize the Chrome driver with the defined options
driver = webdriver.Chrome(options=chrome_options)

# Function to capture, display, save, and return full-page screenshot
def capture_and_return_fullpage_screenshot(url):
    driver.get(url)

    # Try to close cookie notice if it exists
    #lose_cookie_notice(driver)

    # Trigger JavaScript to get the full page screenshot
    result = driver.execute_script("return document.body.parentNode.scrollHeight")
    driver.set_window_size(800, result)  # Width, Height
    png = driver.get_screenshot_as_png()

    # Save the screenshot to a file
    screenshot_path = "screenshotZ.png"  # Local path
    with open(screenshot_path, "wb") as file:
        file.write(png)

    print(f"Screenshot saved at {screenshot_path}")

    return png


def analyze_image_for_textual_criterias(image_data):
    """Analyze the image for specific text criteria and store results in a DataFrame."""
    client = vision.ImageAnnotatorClient()

    # Use the byte array of the image directly
    image = vision.Image(content=image_data)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("No text detected in the image.")
        return

    # Extracting the first element which contains the entire extracted text
    full_text = texts[0].description.lower()

    # Define your criteria
    criteria = {
        "trusted shops": "trusted shops" in full_text,
        #"sale or similar": any(word in full_text for word in ["sale", "rabatt", "ermäßigung", "schlussverkauf"]),
        #"return or similar": any(word in full_text for word in ["return", "rücksendung", "rückversand", "rückgabe", "rücksendung informationen", "rückerstattung"]),
        #"free returns": "free returns" in full_text or "rücksendung kostenlos" in full_text or "kostenlose lieferung und rücksendung" in full_text,
        "free delivery": any(word in full_text for word in ["kostenlose lieferung", "gratisversand", "standardlieferung - kostenlos ab", "kostenfreier versand", "kostenloser versand ab", "kostenloser versand"]),
        "FAQ": "faq" in full_text or "fragen und antworten" in full_text or "fragen & antworten" in full_text,
    }

    # Convert the dictionary to a DataFrame
    criteria_df = pd.DataFrame(list(criteria.items()), columns=['Criteria', 'Result (1=Yes, 0=No)'])

    return full_text,criteria_df


def interview_and_store(
    temperature: float,
    project_id: str,
    location: str,
    full_text: str
) -> pd.DataFrame:

    vertexai.init(project=project_id, location=location)
    parameters = {
        "temperature": temperature,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")

    # Create prompts for each of the criteria, including "trusted shops"
 # Create prompts for each of the criteria, including "trusted shops"
    prompts = [
        "Determine if the following text is in German. Respond with 'yes' or 'no' and briefly explain your reasoning in one sentence: {full_text}",
        "Assess if the following text contains delivery-related information. Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}",
        "Identify any occurrences of 'sale', 'Rabatt', 'Ermäßigung', or 'Schlussverkauf', or similar terms in English or German in the text. Respond 'yes' or 'no'. If 'yes', indicate the location in the text and provide the original German phrase. Note: The text is OCR-extracted from an image. Also, specify its position in the image: {full_text}",
        "Search for terms related to returns like 'return', 'Rücksendung', 'Rückversand', 'Rückgabe', 'Rücksendung Informationen', 'Rückerstattung', or similar in English or German. Reply 'yes' or 'no'. If 'yes', locate these terms in the text, give the original German text, and describe their location in the image (OCR-extracted): {full_text}",
        "Look for phrases like 'free returns', 'Rücksendung kostenlos', 'Kostenlose Lieferung und Rücksendung', or similar in English or German. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text, provide the original German phrase, and their location in the OCR-extracted image: {full_text}",
        "Search for 'free delivery', 'Kostenlose Lieferung', 'gratisversand', 'Standardlieferung - Kostenlos ab', 'Kostenfreier Versand', 'Kostenloser Versand ab', or similar terms in English or German. Respond 'yes' or 'no'. If 'yes', specify their location in the text, include the German original, and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of FAQ, 'Fragen und Antworten', or 'Fragen & Antworten', or similar in English or German. Answer 'yes' or 'no'. If 'yes', locate these in the text, provide the original German phrase, and their location in the OCR-extracted image: {full_text}",
        "Determine if there's any mention of 'trusted shops' in English or German. Reply 'yes' or 'no'. If 'yes', indicate where it is in the text, provide the German original, and describe its location in the OCR-extracted image: {full_text}"
    ]



    column_names = [
        'is_german',
        'contains_delivery_info',
        'contains_phone_number',
        'contains_sale_info',
        'contains_return_info',
        'contains_free_returns_info',
        'contains_free_delivery_info',
        'contains_FAQ',
        'contains_trusted_shops_info'
    ]



    #results = []

    results = []
    prompt_to_column_name = {
    prompts[0]: 'is_german',
    prompts[1]: 'contains_delivery_info',
    prompts[2]: 'contains_phone_number',
    prompts[3]: 'contains_sale_info',
    prompts[4]: 'contains_return_info',
    prompts[5]: 'contains_free_returns_info',
    prompts[6]: 'contains_free_delivery_info',
    prompts[7]: 'contains_FAQ_info',
    prompts[8]: 'contains_trusted_shops_info'
}


    for prompt in prompts:
        response = model.predict(prompt, **parameters)
        response_text = response.text.lower()
        print(response_text)

        if "yes" in response_text:
            criteria_result = "yes"
            additional_info = response_text.split("yes")[1].strip()
        elif "no" in response_text:
            criteria_result = "no"
            additional_info = "No"
        else:
            criteria_result = "unknown"
            additional_info = "Unknown"

        column_name = prompt_to_column_name.get(prompt, "Unknown Column")

        results.append({
            "criteria": column_name,  # Using the column name instead of the full prompt
            "yes or no": criteria_result,
            "additional infos": additional_info
        })


        # Create a DataFrame from the results
    df = pd.DataFrame(results)

    return df


    #from PIL import Image


def init_vertex_ai(project_id, region):
    vertexai.init(project=project_id, location=region)

def initialize_model():
    return GenerativeModel("gemini-pro-vision")

def analyze_image(model, prompt, image):
    response = model.generate_content([prompt, image])
    return response.text
    
def process_response(response_text):
    yes_no = "yes" if "yes" in response_text.lower() else "no" if "no" in response_text.lower() else "unknown"
    return {"yes/no": yes_no, "additional_infos": response_text}

def create_dataframe(data):
    return pd.DataFrame(data)

def load_css():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def render_navbar():
    navbar_html = """
    <nav class="navbar">
        <a href="#home">Home</a>
        <a href="#create-account">Create Account</a>
        <a href="#login">Log In</a>
        <a href="#pricing">Pricing</a>
        <a href="#about">About</a>
    </nav>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)

def render_header():
    header_html = """
        <div class="app-header">
            <h1 class="app-title">Webpage Image Analysis Tool</h1>
            <p class="app-description">Capture and analyze website screenshots with AI-powered technology.</p>
        </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_input_section():
    st.markdown('<div class="input-section"><h2>Start Analysis</h2></div>', unsafe_allow_html=True)

    # List of countries
    countries = ["USA", "Canada", "Germany", "France", "Japan", "Australia", "India"]
    selected_country = st.selectbox("Choose a country:", countries)

    url_input = st.text_input("Enter the URL of the website:", "")

    return url_input, selected_country

def render_about_section():
    about_html = """
        <div class="info-section">
            <h2>About the Tool</h2>
            <p>This tool utilizes state-of-the-art AI algorithms to analyze website images and generate insights.</p>
        </div>
    """
    st.markdown(about_html, unsafe_allow_html=True)

def render_footer():
    footer_html = '<div class="footer">© 2024 Web Analysis Tool. All rights reserved.</div>'
    st.markdown(footer_html, unsafe_allow_html=True)


def convert_df_to_xlsx(df):
    """
    Convert the provided dataframe to an XLSX file.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
    processed_data = output.getvalue()
    return processed_data

#url = "input"

def main():
    load_css()
    render_navbar()
    render_header()
    url, selected_country = render_input_section()
    if st.button('Analyze'):
        screenshot_data = capture_and_return_fullpage_screenshot(url)
        full_text, df2 = analyze_image_for_textual_criterias(screenshot_data)
        df1 = interview_and_store(0.1, 'jovial-circuit-412017', 'us-central1', full_text)
        print(df1)
        # Main execution
        PROJECT_ID = "jovial-circuit-412017"
        REGION = "us-central1"
        IMAGE_FILE = "screenshotZ.png"
        data = analyze_image_for_criteria(IMAGE_FILE, PROJECT_ID, REGION)


        df3 = create_dataframe(data)
        print(df3)
        #Csv_formatting 
        #df2.rename(columns={df2.columns[1]: 'yes/no(1/0)'}, inplace=True)
        #df2['yes/no(1/0)'] = df2['yes/no(1/0)'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', 'true'] else 0)
        # Rename the second column
        df1.rename(columns={'yes or no': 'yes/no(1/0)'}, inplace=True)

        # Convert values in the 'yes/no(1/0)' column to 1 or 0 based on the criteria
        df1['yes/no(1/0)'] = df1['yes/no(1/0)'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', 'true'] else 0)

        df1.head()

        # Rename the second column and update its values
        df3.rename(columns={df3.columns[0]: 'yes/no(1/0)'}, inplace=True)
        df3['yes/no(1/0)'] = df3['yes/no(1/0)'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', 'true'] else 0)

        df1 = df1.rename(columns={'additional infos': 'additional_infos'})
        #df2 = df2.rename(columns={'Criteria': 'criteria'})

        # Add an empty column 'additional_infos' to df2
        #df2['additional_infos'] = ''

        result = pd.concat([df1,df3], axis=0, ignore_index=True)

        result.insert(0, 'Company_Name', 'HouseOfShoes')

        # Add 'Company_Url' column at the second position
        result.insert(1, 'Company_Url', url)

        xlsx_data = convert_df_to_xlsx(result)

        # Create a download link for the XLSX file
        st.download_button(
            label="Download Results as XLSX",
            data=xlsx_data,
            file_name="analysis_results.xlsx",
            mime="application/vnd.ms-excel"
        )


        pass
        #process_url(url_input)
    # Add another button for adding another URL
    if st.button('Add Another URL'):
        # Clear the previous URL input or perform any other desired action
        st.text_input("Enter another URL:", "")
    render_about_section()
    render_footer()

if __name__ == "__main__":
    main()
