# main.py

import streamlit as st
from web_scraper import WebScraper
from image_analysis import analyze_image_for_criteria
from text_detection import TextDetector
from text_generation import TextGenerator
from data_manager import DataManager
import app_ui
from config import GOOGLE_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS, VERTEX_AI_REGION
import os
def main():
    os.environ['GRPC_DNS_RESOLVER'] = 'native'
    # Load CSS for the Streamlit app
    app_ui.load_css()

    # Render the UI components
    app_ui.render_navbar()
    app_ui.render_header()
    url, selected_country = app_ui.render_input_section()

    if st.button('Analyze'):
        # Initialize WebScraper and capture a screenshot
        scraper = WebScraper()
        screenshot_data = scraper.capture_and_return_fullpage_screenshot(url)
        scraper.close()

        # Initialize TextDetector and analyze the image for text
        text_detector = TextDetector()
        detected_texts = text_detector.analyze_image_for_text(screenshot_data)
        text_criteria_results = text_detector.process_detected_text(detected_texts)

        # Initialize TextGenerator and generate text responses
        text_generator = TextGenerator(GOOGLE_PROJECT_ID, VERTEX_AI_REGION)
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
        parameters = {"temperature": 0.7, "max_output_tokens": 256, "top_p": 0.8, "top_k": 40}
        text_responses = text_generator.generate_text_responses(prompts, parameters)
        processed_text_results = text_generator.process_responses(text_responses, prompts)

        # Analyze the image for specific criteria using Image Analysis
        image_analysis_results = analyze_image_for_criteria('screenshot.png', GOOGLE_PROJECT_ID, VERTEX_AI_REGION)

        # Data Management and Export
        rename_mappings = {'yes or no': 'yes/no(1/0)'}
        convert_columns = {'yes/no(1/0)': lambda x: 1 if str(x).strip().lower() in ['yes', 'true'] else 0}
        #processed_text_results=DataManager.preprocess_dataframe(processed_text_results,rename_mappings=rename_mappings,convert_columns=convert_columns)
        #image_analysis_results=DataManager.preprocess_dataframe(image_analysis_results)
        final_results = DataManager.merge_dataframes([processed_text_results, image_analysis_results])
        xlsx_data = DataManager.convert_df_to_xlsx(final_results)

        # Render the download button for the results
        app_ui.render_download_button(xlsx_data)

    app_ui.render_about_section()
    app_ui.render_footer()

if __name__ == "__main__":
    main()
