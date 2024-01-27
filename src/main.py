# main.py
import zipfile
import pandas as pd
from urllib.parse import urlparse
import streamlit as st
from web_scraper import WebScraper
from image_analysis import analyze_image_for_criteria
from text_detection import TextDetector
from text_generation import TextGenerator
from data_manager import DataManager
import app_ui
from config import GOOGLE_PROJECT_ID, VERTEX_AI_REGION
import os
import google.generativeai as genai
import io
#import zip

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Data Professor</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">YouTube</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Twitter</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)


def get_prompts_for_country_text(country):
    prompts_dict = {
        "Germany": [
        "Determine if there's any mention of 'trusted shops' in English or German. Reply 'yes' or 'no'. If 'yes', indicate where it is in the text, provide the German original, and describe its location in the OCR-extracted image: {full_text}"
        "Determine if the following text is in German. Respond with 'yes' or 'no' and briefly explain your reasoning in one sentence: {full_text}",
        "Identify any occurrences of 'sale', 'Rabatt', 'Ermäßigung', or 'Schlussverkauf', or similar terms in English or German in the text. Respond 'yes' or 'no'. If 'yes', indicate the location in the text and provide the original German phrase. Note: The text is OCR-extracted from an image. Also, specify its position in the image: {full_text}",
        "Search for terms related to returns like 'return', 'Rücksendung', 'Rückversand', 'Rückgabe', 'Rücksendung Informationen', 'Rückerstattung', or similar in English or German. Reply 'yes' or 'no'. If 'yes', locate these terms in the text, give the original German text, and describe their location in the image (OCR-extracted): {full_text}",
        "Look for phrases like 'free returns', 'Rücksendung kostenlos', 'Kostenlose Lieferung und Rücksendung', or similar in English or German. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text, provide the original German phrase, and their location in the OCR-extracted image: {full_text}",
        "Assess if the following text contains delivery-related information. Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Search for 'free delivery', 'Kostenlose Lieferung', 'gratisversand', 'Standardlieferung - Kostenlos ab', 'Kostenfreier Versand', 'Kostenloser Versand ab', or similar terms in English or German. Respond 'yes' or 'no'. If 'yes', specify their location in the text, include the German original, and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of FAQ, 'Fragen und Antworten', or 'Fragen & Antworten', or similar in English or German. Answer 'yes' or 'no'. If 'yes', locate these in the text, provide the original German phrase, and their location in the OCR-extracted image: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}"

   ],
    "Spain" : [
        "Check if the text mentions 'trustpilot', 'avis verifies', or 'google reviews'. Respond 'yes' or 'no'. If 'yes', indicate where it is in the text, provide the Spanish original, and describe its location in the OCR-extracted image: {full_text}",
        "Determine if the text in the image is in Spanish. Reply with 'yes' or 'no' and briefly explain your reasoning: {full_text}",
        "Identify any occurrences of 'sale', 'venta', 'rebajas', 'oferta', 'descuento', or 'promoción' in the text. Answer 'yes' or 'no'. If 'yes', indicate their location in the text, provide the Spanish phrases, and specify their position in the OCR-extracted image: {full_text}",
        "Search for terms related to returns like 'return', 'devolución', 'retorno', or 'regreso' in the text. Respond 'yes' or 'no'. If 'yes', locate these terms in the text, give the Spanish phrases, and describe their location in the OCR-extracted image: {full_text}",
        "Look for phrases like 'free returns', 'devolución gratuita', 'devolución gratis', or 'retorno gratuito' in the text. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text, provide the Spanish phrases, and their location in the OCR-extracted image: {full_text}",
        "Assess if the text contains the word 'delivery', 'entrega', or 'envío'. Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Check for mentions of 'free delivery', 'entrega gratuita', 'entrega gratis', or 'envío gratis' in the text. Respond 'yes' or 'no'. If 'yes', specify their location in the text, include the Spanish phrases, and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of 'FAQ', 'preguntas frecuentes', 'P+F', or 'preguntas más comunes' in the text. Answer 'yes' or 'no'. If 'yes', locate these in the text, provide the Spanish phrases, and their location in the OCR-extracted image: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}"
    ],

    "France" : [
        "Check if the text mentions 'avis verifies', 'google reviews', or 'TrustPilot'. Respond 'yes' or 'no'. If 'yes', indicate where it is in the text, provide the French original, and describe its location in the OCR-extracted image: {full_text}",
        "Determine if the text in the image is in French. Reply with 'yes' or 'no' and briefly explain your reasoning: {full_text}",
        "Identify any occurrences of 'sale', 'soldes', 'promotions', 'rabais', or 'offres spéciales' in the text. Answer 'yes' or 'no'. If 'yes', indicate their location in the text, provide the French phrases, and specify their position in the OCR-extracted image: {full_text}",
        "Search for terms related to returns like 'return', 'retour', 'renvoi', or 'remboursement' in the text. Respond 'yes' or 'no'. If 'yes', locate these terms in the text, give the French phrases, and describe their location in the OCR-extracted image: {full_text}",
        "Look for phrases like 'free returns', 'retour gratuit', 'renvoi gratuit', or 'remboursement sans frais' in the text. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text, provide the French phrases, and their location in the OCR-extracted image: {full_text}",
        "Assess if the text contains the words 'delivery', 'livraison', 'remise', or 'distribution'. Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Check for mentions of 'free delivery', 'livraison gratuite', 'livraison offerte', or 'frais de port offerts' in the text. Respond 'yes' or 'no'. If 'yes', specify their location in the text, include the French phrases, and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of 'FAQ', 'questions fréquentes', or 'questions courantes' in the text. Answer 'yes' or 'no'. If 'yes', locate these in the text, provide the French phrases, and their location in the OCR-extracted image: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}",
    ],

    "Brazil" : [
        "Check if the text mentions 'Trustpilot', 'Google Reviews', 'Reclame Aqui', 'Ebit', or 'Opinion Box'. Respond 'yes' or 'no'. If 'yes', indicate where it is in the text, provide the Portuguese original, and describe its location in the OCR-extracted image: {full_text}",
        "Determine if the text in the image is in Portuguese. Reply with 'yes' or 'no' and briefly explain your reasoning: {full_text}",
        "Identify any occurrences of 'sale', 'liquidação', 'promoção', 'oferta', or 'desconto' in the text. Answer 'yes' or 'no'. If 'yes', indicate their location in the text, provide the Portuguese phrases, and specify their position in the OCR-extracted image: {full_text}",
        "Search for terms related to returns like 'return', 'devolução', 'troca', or 'reembolso' in the text. Respond 'yes' or 'no'. If 'yes', locate these terms in the text, give the Portuguese phrases, and describe their location in the OCR-extracted image: {full_text}",
        "Look for phrases like 'free returns', 'devolução gratuita', 'devolução grátis', 'troca gratuita', or 'frete de devolução grátis' in the text. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text, provide the Portuguese phrases, and their location in the OCR-extracted image: {full_text}",
        "Assess if the text contains any of these words 'delivery', 'entrega', 'remesa', or 'frete'. Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Check for mentions of 'free delivery', 'frete grátis', 'entrega grátis', or 'entrega gratuita' in the text. Respond 'yes' or 'no'. If 'yes', specify their location in the text, include the Portuguese phrases, and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of 'FAQ', 'perguntas frequentes', 'dúvidas frequentes', or 'perguntas e respostas' in the text. Answer 'yes' or 'no'. If 'yes', locate these in the text, provide the Portuguese phrases, and their location in the OCR-extracted image: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}",
    ],

    "Italy" : [
        "Check if the text mentions 'TrustPilot', 'Google Recensioni', or 'Altroconsumo'. Respond 'yes' or 'no'. If 'yes', indicate where it is in the text, provide the Italian original, and describe its location in the OCR-extracted image: {full_text}",
        "Determine if the text in the image is in Italian. Reply with 'yes' or 'no' and briefly explain your reasoning: {full_text}",
        "Identify any occurrences of 'sale', 'saldi', 'sconti', 'offerte', or 'promozioni' in the text. Answer 'yes' or 'no'. If 'yes', indicate their location in the text, provide the Italian phrases, and specify their position in the OCR-extracted image: {full_text}",
        "Search for terms related to returns like 'return', 'reso', 'restituzione', or 'rimborso' in the text. Respond 'yes' or 'no'. If 'yes', locate these terms in the text, give the Italian phrases, and describe their location in the OCR-extracted image: {full_text}",
        "Look for phrases like 'free returns', 'reso gratuito', 'reso gratis', or 'restituzione gratuita' in the text. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text, provide the Italian phrases, and their location in the OCR-extracted image: {full_text}",
        "Assess if the text contains any of these words 'delivery', 'consegna', 'spedizione', or 'recapito'. Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Check for mentions of 'free delivery', 'spedizione gratuita', or 'consegna gratuita' in the text. Respond 'yes' or 'no'. If 'yes', specify their location in the text, include the Italian phrases, and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of 'FAQ', 'domande frequenti' in the text. Answer 'yes' or 'no'. If 'yes', locate these in the text, provide the Italian phrases, and their location in the OCR-extracted image: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}",
    ],

    "UAE" : [
        "Check if the text mentions 'Google Reviews' (جوجل ريفيوز), 'Trustpilot' (ترست بايلوت), or 'Yallacompare' (يلا كومبير). Respond 'yes' or 'no'. If 'yes', indicate where it is in the text, provide the Arabic original, and describe its location in the OCR-extracted image: {full_text}",
        "Determine if the text in the image is in Arabic. Reply with 'yes' or 'no' and briefly explain your reasoning: {full_text}",
        "Identify any occurrences of 'sale', 'تخفيضات' (Takhfidat), 'تنزيلات' (Tanzilat), 'عروض' (Urood), 'خصومات' (Khusumat) in the text. Answer 'yes' or 'no'. If 'yes', indicate their location in the text, provide the Arabic phrases, and specify their position in the OCR-extracted image: {full_text}",
        "Search for terms related to returns like 'return', 'إرجاع' (Irja'), 'استبدال' (Istibdal), 'استرداد' (Istirdad) in the text. Respond 'yes' or 'no'. If 'yes', locate these terms in the text, give the Arabic phrases, and describe their location in the OCR-extracted image: {full_text}",
        "Look for phrases like 'free returns', 'إرجاع مجاني' (Irja' majani), 'استبدال مجاني' (Istibdal majani), 'استرداد مجاني' (Istirdad majani) in the text. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text, provide the Arabic phrases, and their location in the OCR-extracted image: {full_text}",
        "Assess if the text contains any of these words 'delivery', 'توصيل' (Tawseel), 'تسليم' (Tasleem), 'شحن' (Shahn). Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Check for mentions of 'free delivery', 'توصيل مجاني' (Tawseel majani), 'شحن مجاني' (Shahn majani) in the text. Respond 'yes' or 'no'. If 'yes', specify their location in the text, include the Arabic phrases, and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of 'FAQ', 'الأسئلة الشائعة' (Al-as'ila al-shai'ea), 'الأسئلة المتكررة' (Al-as'ila al-mutakarira) in the text. Answer 'yes' or 'no'. If 'yes', locate these in the text, provide the Arabic phrases, and their location in the OCR-extracted image: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}",
    ],

    "Japan" : [
        "Check if the text mentions 'Google レビュー' (Google rebyu) or '価格.com' (Kakaku.com). Respond 'yes' or 'no'. If 'yes', indicate where it is in the text, provide the Japanese original, and describe its location in the OCR-extracted image: {full_text}",
        "Determine if the text in the image is in Japanese. Reply with 'yes' or 'no' and briefly explain your reasoning: {full_text}",
        "Identify any occurrences of 'sale', 'セール' (se-ru), 'バーゲン' (ba-gen), '割引' (waribiki), '特売' (tokubai) in the text. Answer 'yes' or 'no'. If 'yes', indicate their location in the text, provide the Japanese phrases, and specify their position in the OCR-extracted image: {full_text}",
        "Search for terms related to returns like 'return', '返品' (henpin), '交換' (koukan), '返金' (henkin) in the text. Respond 'yes' or 'no'. If 'yes', locate these terms in the text, give the Japanese phrases, and describe their location in the OCR-extracted image: {full_text}",
        "Look for phrases like 'free returns', '返品無料' (henpin muryou), '送料無料返品' (souryou muryou henpin) in the text. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text, provide the Japanese phrases, and their location in the OCR-extracted image: {full_text}",
        "Assess if the text contains any of these words 'delivery', '配送' (haisou), '配達' (haitatsu), '発送' (hasou). Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Check for mentions of 'free delivery', '送料無料' (souryou muryou) in the text. Respond 'yes' or 'no'. If 'yes', specify their location in the text, include the Japanese phrases, and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of 'FAQ' in the text. Answer 'yes' or 'no'. If 'yes', locate these in the text, provide the Japanese phrases, and their location in the OCR-extracted image: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}",
    ],

    "US" :[
        "Check if there's a badge from the 'Better Business Bureau', 'google reviews', 'reviews.io', or 'Yotpo' in the text. Respond 'yes' or 'no'. If 'yes', indicate where it is in the text and describe its location in the OCR-extracted image: {full_text}",
        "Determine if the text in the image is in US English. Reply with 'yes' or 'no' and briefly explain your reasoning: {full_text}",
        "Identify any occurrences of 'sale', 'clearance', 'discount', 'promotion' in the text. Answer 'yes' or 'no'. If 'yes', indicate their location in the text and specify their position in the OCR-extracted image: {full_text}",
        "Search for terms related to returns like 'returns', 'exchange', or 'refund' in the text. Respond 'yes' or 'no'. If 'yes', locate these terms in the text and describe their location in the OCR-extracted image: {full_text}",
        "Look for phrases like 'free returns' or 'hassle-free returns' in the text. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text and their location in the OCR-extracted image: {full_text}",
        "Assess if the text contains any of these words 'delivery', 'shipping', or 'fulfillment'. Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Check for mentions of 'free delivery' or 'free shipping' in the text. Respond 'yes' or 'no'. If 'yes', specify their location in the text and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of 'FAQ' or 'frequently asked questions' in the text. Answer 'yes' or 'no'. If 'yes', locate these in the text and their location in the OCR-extracted image: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}",
    ]
    }
    return prompts_dict.get(country, [])

def get_prompts_for_country_images(country):
    prompts_dict = {
        "Germany": [
        "Scan for delivery logos: DHL, Hermes, DPD, UPS, FedEx, Deutsche Post. Yes or no? If yes, which and where?",
        "Detect payment logos: Visa, Mastercard, PayPal. Present, yes or no? If yes, specify brands and locations.",
        "Look for payment logos: Klarna, Sofort, Giropay. Are they in the image? If yes, identify brands and their locations.",
        "Look for chat support icons , Are they in the image? If yes,  identify their locations."
    ],
        "Spain": [
    "Scan for Spanish delivery company logos: Correos, SEUR, MRW, DHL, UPS. Are they present in the image? If yes, identify which logos are visible and their locations.",
    "Detect payment logos: Visa, Mastercard, PayPal. Are these logos in the image? If yes, specify which brands are present and their locations.",
    "Look for the Bizum payment logo. Is it present in the image? If yes, identify its location.",
    "Identify if there is a chat support icon in the image. If yes, locate and describe its position."
],

    "France" : [
    "Detect payment logos: Visa, Mastercard, PayPal. Are these logos in the image? If yes, specify which brands are present and their locations.",
    "Look for the Carte Bancaire payment logo. Is it present in the image? If yes, identify its location.",
    "Check for the presence of the text 'FAQ' or its French equivalents 'questions fréquentes' or 'questions courantes' in the image. If found, specify the location of the text.",
    "Identify if there is a chat support icon in the image. If yes, locate and describe its position."
],

    "Brazil" : [
    "Detect payment logos: Visa, Mastercard, PayPal. Are these logos in the image? If yes, specify which brands are present and their locations.",
    "Look for payment logos: Elo, Hipercard, boleto bancário, PIX. Are these logos in the image? If yes, specify which brands are present and their locations.",
    "Check for the presence of the text 'FAQ' or its Portuguese equivalents 'perguntas frequentes' or 'dúvidas frequentes' or 'perguntas e respostas' in the image. If found, specify the location of the text.",
    "Identify if there is a chat support icon in the image. If yes, locate and describe its position."
],

    "Italy" :[
    "Detect payment logos: Visa, Mastercard, PayPal. Are these logos in the image? If yes, specify which brands are present and their locations.",
    "Look for payment logos: Postepay, Bancomat, Bancomat Pay, il circuito PagoBancomat. Are these logos in the image? If yes, specify which brands are present and their locations.",
    "Check for the presence of the text 'FAQ' or its Italian equivalent 'domande frequenti' in the image. If found, specify the location of the text.",
    "Identify if there is a chat support icon in the image. If yes, locate and describe its position."
], 

    "UAE" : [
    "Scan for delivery company logos: Aramex (أرامكس), FedEx (فيديكس), DHL, and Emirates Post (بريد الإمارات). Are they present in the image? If yes, identify which logos are visible and their locations.",
    "Detect payment logos: Visa, Mastercard, PayPal. Are these logos in the image? If yes, specify which brands are present and their locations.",
    "Look for payment logos: Cash on Delivery (COD) (الدفع عند الاستلام). Are these logos in the image? If yes, specify their locations.",
    "Identify if there is a chat support icon in the image. If yes, locate and describe its position."
],

    "Japan" : [
    "Scan for delivery company logos: ヤマト運輸(Yamato unyu), 佐川急便(Sagawa kyuubin), 日本郵便(ゆうパック) (Nippon yuubin (yuupakku)). Are they present in the image? If yes, identify which logos are visible and their locations.",
    "Detect payment logos: Visa, Mastercard, PayPal. Are these logos in the image? If yes, specify which brands are present and their locations.",
    "Look for payment logos: 各種クレジットカード(kurejitto ka-do: credit cards), コンビニ決済(konbini kesai: convenience store payment), 代金引換(daikin hikikae: cash on delivery). Are these logos in the image? If yes, specify their locations.",
    "Identify if there is a chat support icon in the image. If yes, locate and describe its position."
],

    "US" : [
    "Scan for delivery logos: UPS, FedEx, USPS (United States Postal Service). Yes or no? If yes, which and where?",
    "Detect payment logos: Visa, Mastercard, PayPal. Present, yes or no? If yes, specify brands and locations.",
    "Look for payment logos: American Express, Venmo, Cash App. Are they in the image? If yes, identify brands and their locations.",
    "Look for chat support icons. Are they in the image? If yes, identify their locations."
]

    
        # ... Add for other countries
    }
    return prompts_dict.get(country, [])


def main():
    os.environ['GRPC_DNS_RESOLVER'] = 'native'

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(135deg, #ffffff, #e6e8eb);
    }
    .icon {
        font-size: 1.5em;
        vertical-align: middle;
        margin-right: 10px;
    }
    /* Additional styles here */
    </style>
""", unsafe_allow_html=True)


    # Load CSS for the Streamlit app
    app_ui.load_css()

    # Render the UI components
    #app_ui.render_navbar2()
    app_ui.render_header()
    url, selected_country,analyze_button = app_ui.render_input_section2()


    def create_zip_file(files):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for file_name, data in files.items():
                zip_file.writestr(file_name, data.getvalue())
        return zip_buffer.getvalue()
    if analyze_button:
        xlsx_files = {}
        # Initialize WebScraper and capture a screenshot
        for index, url in enumerate(url): 
            scraper = WebScraper()
            scraper.handle_cookies(url)
            screenshot_data,screenshot_path = scraper.capture_and_return_fullpage_screenshot(url)
            scraper.close()

            # Initialize TextDetector and analyze the image for text
            text_detector = TextDetector()
            detected_texts = text_detector.analyze_image_for_text(screenshot_data)
            text_criteria_results = text_detector.process_detected_text(detected_texts)

            # Initialize TextGenerator and generate text responses
            text_generator = TextGenerator(GOOGLE_PROJECT_ID, VERTEX_AI_REGION)
            prompts = get_prompts_for_country_text(selected_country)  
            parameters = {"temperature": 0.7, "max_output_tokens": 256, "top_p": 0.8, "top_k": 40}
            text_responses = text_generator.generate_text_responses(prompts, parameters)
            processed_text_results = text_generator.process_responses(text_responses, prompts)
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


            # Analyze the image for specific criteria using Image Analysis
            image_analysis_results = analyze_image_for_criteria(screenshot_path, GOOGLE_PROJECT_ID, VERTEX_AI_REGION,prompts=get_prompts_for_country_images(selected_country))

            # Data Management and Export
            rename_mappings = {'yes or no': 'yes/no(1/0)'}
            convert_columns = {'yes/no(1/0)': lambda x: 1 if str(x).strip().lower() in ['yes', 'true'] else 0}
            #processed_text_results=DataManager.preprocess_dataframe(processed_text_results,rename_mappings=rename_mappings,convert_columns=convert_columns)
            #image_analysis_results=DataManager.preprocess_dataframe(image_analysis_results)
            final_results = DataManager.merge_dataframes([processed_text_results, image_analysis_results])
            final_results['yes or no'] = final_results['yes or no'].map({'yes': 1, 'no': 0})
            
            parsed_url = urlparse(url)
            domain_name = parsed_url.netloc

    # Extracting just the 'kaufland' part from the domain name
            extracted_name = domain_name.split('.')[1] 
            
            final_results.insert(0, 'Company_Name', extracted_name)

            # Add 'Company_Url' column at the second position
            final_results.insert(1, 'Company_Url', url)
            # This assumes the format is [subdomain].[name].[tld]

            xlsx_data = DataManager.convert_df_to_xlsx(final_results)
            
            file_name = f"analysis_results_{index}.xlsx"
            xlsx_files[file_name] = xlsx_data

            # Create a zip file with all the XLSX files
        if xlsx_files:
            zip_data = create_zip_file(xlsx_files)
            st.download_button(
                label="Download All Results as ZIP",
                data=zip_data,
                file_name="all_analysis_results.zip",
                mime="application/zip"
            )# Render the download button for the results
            # Create a unique key for each download button
            #download_button_key = f"download_button_{index}"

            # Render the download button with the unique key
            #app_ui.render_download_button(xlsx_data, key=download_button_key)

    app_ui.render_about_section()
    app_ui.render_footer()

if __name__ == "__main__":
    main()
