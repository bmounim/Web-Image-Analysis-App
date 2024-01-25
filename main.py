# main.py
import pandas as pd
from urllib.parse import urlparse
import streamlit as st
from web_scraper import WebScraper
from image_analysis import analyze_image_for_criteria
from text_detection import TextDetector
from text_generation import TextGenerator
from data_manager import DataManager
import app_ui
from config import GOOGLE_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS, VERTEX_AI_REGION
import os

def get_prompts_for_country_image(country):
    prompts_dict = {
        "Germany": ["Determine if the following text is in German. Respond with 'yes' or 'no' and briefly explain your reasoning in one sentence: {full_text}",
        "Assess if the following text contains delivery-related information. Reply with 'yes' or 'no'. If 'yes', summarize the delivery details: {full_text}",
        "Check if there's a phone number in the text. Answer 'yes' or 'no'. If 'yes', please provide the phone number: {full_text}",
        "Identify any occurrences of 'sale', 'Rabatt', 'Ermäßigung', or 'Schlussverkauf', or similar terms in English or German in the text. Respond 'yes' or 'no'. If 'yes', indicate the location in the text and provide the original German phrase. Note: The text is OCR-extracted from an image. Also, specify its position in the image: {full_text}",
        "Search for terms related to returns like 'return', 'Rücksendung', 'Rückversand', 'Rückgabe', 'Rücksendung Informationen', 'Rückerstattung', or similar in English or German. Reply 'yes' or 'no'. If 'yes', locate these terms in the text, give the original German text, and describe their location in the image (OCR-extracted): {full_text}",
        "Look for phrases like 'free returns', 'Rücksendung kostenlos', 'Kostenlose Lieferung und Rücksendung', or similar in English or German. Answer 'yes' or 'no'. If 'yes', mention where they are found in the text, provide the original German phrase, and their location in the OCR-extracted image: {full_text}",
        "Search for 'free delivery', 'Kostenlose Lieferung', 'gratisversand', 'Standardlieferung - Kostenlos ab', 'Kostenfreier Versand', 'Kostenloser Versand ab', or similar terms in English or German. Respond 'yes' or 'no'. If 'yes', specify their location in the text, include the German original, and indicate their position in the OCR-extracted image: {full_text}",
        "Identify if there are any instances of FAQ, 'Fragen und Antworten', or 'Fragen & Antworten', or similar in English or German. Answer 'yes' or 'no'. If 'yes', locate these in the text, provide the original German phrase, and their location in the OCR-extracted image: {full_text}",
        "Determine if there's any mention of 'trusted shops' in English or German. Reply 'yes' or 'no'. If 'yes', indicate where it is in the text, provide the German original, and describe its location in the OCR-extracted image: {full_text}"
   ],



        "Spain": [
    "Check if the text in the image mentions 'trustpilot', ‘avis verifies’, or ‘google reviews’. Respond with 'yes' or 'no'. If 'yes', indicate where it is found in the image: {full_text}",
    "Determine if the text in the image is in Spanish. Answer with 'yes' or 'no' and explain your reasoning briefly in one sentence: {full_text}",
    "Identify any occurrences of 'sale', 'venta', 'rebajas', 'oferta', or 'descuento', or 'promoción' in the image. Reply 'yes' or 'no'. If 'yes', indicate where these words are found in the image: {full_text}",
    "Search for terms related to returns like 'return', 'devolución', 'retorno', or 'regreso' in the image. Answer 'yes' or 'no'. If 'yes', locate these terms in the image: {full_text}",
    "Look for phrases like 'free returns', 'devolución gratuita', 'devolución gratis', or 'retorno gratuito' in the image. Respond 'yes' or 'no'. If 'yes', mention where they are found in the image: {full_text}",
    "Determine if the image contains the word 'delivery', 'entrega', or 'envío'. Reply 'yes' or 'no'. If 'yes', specify their location in the image: {full_text}",
    "Check if the image mentions 'free delivery', 'entrega gratuita', 'entrega gratis', or 'envío gratis'. Answer 'yes' or 'no'. If 'yes', indicate where in the image: {full_text}",
    "Identify if there are images from delivery companies like Correos, SEUR, MRW, DHL, and UPS in the image. Reply 'yes' or 'no'. If 'yes', describe their location in the image: {full_text}",
    "Does the image contain logos or images from payment brands such as Visa, Mastercard, PayPal? Answer 'yes' or 'no', and if 'yes', indicate their location in the image: {full_text}",
    "Check if the image contains logos or images from the payment brand bizum. Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Search for text like 'FAQ', 'preguntas frecuentes', 'P+F', or 'preguntas más comunes' in the image. Answer 'yes' or 'no'. If 'yes', locate these texts in the image: {full_text}",
    "Determine if there's a phone number in the image. Respond 'yes' or 'no'. If 'yes', please indicate its location in the image: {full_text}",
    "Check if the image contains a chat support icon. Answer 'yes' or 'no'. If 'yes', specify its location in the image: {full_text}"
],

    "France" : [
    "Is there text saying avis verifies, google reviews or TrustPilot in this. Respond with 'yes' or 'no'. If 'yes', indicate where it is found in the image: {full_text}",
    "Is the text in the image in French? Answer with 'yes' or 'no' and briefly explain your reasoning: {full_text}",
    "Does the image contain any of these words: 'sale', 'soldes', 'promotions', 'rabais', or 'offres spéciales'? Reply 'yes' or 'no'. If 'yes', specify where these words are located in the image: {full_text}",
    "Does the image contain any of these words: 'return', 'retour', 'renvoi', or 'remboursement'? Respond 'yes' or 'no'. If 'yes', locate these words in the image: {full_text}",
    "Does the image contain any of these words: 'free returns', 'retour gratuit', 'renvoi gratuit', or 'remboursement sans frais'? Answer 'yes' or 'no'. If 'yes', mention where they are found in the image: {full_text}",
    "Does this image mention any of these words: 'delivery', 'livraison', 'remise', or 'distribution'? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image contain any of these words: 'free delivery', 'livraison gratuite', 'livraison offerte', or 'frais de port offerts'? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Are they showing images from delivery companies La Poste (Colisimo), Chronopost, DPD, Mondial Relay, and UPS? Answer 'yes' or 'no'. If 'yes', specify where in the image: {full_text}",
    "Does this image contain images from payment brands Visa, Mastercard, PayPal? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image contain images from payment brands Carte Bancaire? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Does the image contain the text 'FAQ' or 'questions fréquentes', or 'questions courantes'? Answer 'yes' or 'no'. If 'yes', locate these texts in the image: {full_text}",
    "Does the image contain a phone number? Respond 'yes' or 'no'. If 'yes', please indicate its location in the image: {full_text}",
    "Does the image contain a chat support icon? Answer 'yes' or 'no'. If 'yes', specify its location in the image: {full_text}"
],

    "Brazil" : [
    "Is there text saying Trustpilot, Google Reviews, Reclame Aqui, Ebit, or Opinion Box in this? Respond with 'yes' or 'no'. If 'yes', indicate where it is found in the image: {full_text}",
    "Is the text in the image in Portuguese? Answer 'yes' or 'no' and briefly explain your reasoning: {full_text}",
    "Does the image contain any of these words: 'sale', 'liquidação', 'promoção', 'oferta', 'desconto'? Reply 'yes' or 'no'. If 'yes', specify where these words are located in the image: {full_text}",
    "Does the image contain any of these words: 'return', 'devolução', 'troca', or 'reembolso'? Respond 'yes' or 'no'. If 'yes', locate these words in the image: {full_text}",
    "Does the image contain any of these words: 'free returns', 'devolução gratuita', 'devolução grátis', 'troca gratuita', or 'frete de devolução grátis'? Answer 'yes' or 'no'. If 'yes', mention where they are found in the image: {full_text}",
    "Does this image contain any of these words: 'delivery', 'entrega', 'remesa', or 'frete'? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image mention free delivery or 'frete grátis', 'entrega grátis', or 'entrega gratuita'? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Are they showing images from delivery companies Correios, Loggi, Total Express, Jadlog, or Azul Cargo Express? Answer 'yes' or 'no'. If 'yes', specify where in the image: {full_text}",
    "Does this image contain images from payment brands Visa, Mastercard, PayPal? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image contain images from payment brands Elo, Hipercard, boleto bancário, or PIX? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Does the image contain the text 'FAQ', 'perguntas frequentes', 'dúvidas frequentes', or 'perguntas e respostas'? Answer 'yes' or 'no'. If 'yes', locate these texts in the image: {full_text}",
    "Does the image contain a phone number? Respond 'yes' or 'no'. If 'yes', please indicate its location in the image: {full_text}",
    "Does the image contain a chat support icon? Answer 'yes' or 'no'. If 'yes', specify its location in the image: {full_text}"
],

    "Italy" : [
    "Is there any text saying TrustPilot, Google Recensioni, or Altroconsumo in this? Respond with 'yes' or 'no'. If 'yes', indicate where it is found in the image: {full_text}",
    "Is the text in the image in Italian? Answer 'yes' or 'no' and briefly explain your reasoning: {full_text}",
    "Does the image contain any of these words: 'sale', 'saldi', 'sconti', 'offerte', 'promozioni'? Reply 'yes' or 'no'. If 'yes', specify where these words are located in the image: {full_text}",
    "Does the image contain any of these words: 'return', 'reso', 'restituzione', 'rimborso'? Respond 'yes' or 'no'. If 'yes', locate these words in the image: {full_text}",
    "Does the image contain any of these words: 'free returns', 'reso gratuito', 'reso gratis', 'restituzione gratuita'? Answer 'yes' or 'no'. If 'yes', mention where they are found in the image: {full_text}",
    "Does this image contain any of these words: 'delivery', 'consegna', 'spedizione', 'recapito'? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image mention free delivery or 'spedizione gratuita', 'consegna gratuita'? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Are they showing images from delivery companies Poste Italiane (SDA), BRT (Bartolini), DHL, UPS, or GLS? Answer 'yes' or 'no'. If 'yes', specify where in the image: {full_text}",
    "Does this image contain images from payment brands Visa, Mastercard, PayPal? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image contain images from payment brands Postepay, Bancomat, Bancomat Pay, or il circuito PagoBancomat? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Does the image contain the text 'FAQ' or 'domande frequenti'? Answer 'yes' or 'no'. If 'yes', locate these texts in the image: {full_text}",
    "Does the image contain a phone number? Respond 'yes' or 'no'. If 'yes', please indicate its location in the image: {full_text}",
    "Does the image contain a chat support icon? Answer 'yes' or 'no'. If 'yes', specify its location in the image: {full_text}"
],

    "UAE" : [
    "Is there text saying Google Reviews (جوجل ريفيوز), Trustpilot (ترست بايلوت), or Yallacompare (يلا كومبير) in this? Respond with 'yes' or 'no'. If 'yes', indicate where it is found in the image: {full_text}",
    "Is the text in the image in Arabic? Answer 'yes' or 'no' and briefly explain your reasoning: {full_text}",
    "Does the image contain any of these words: 'sale', تخفيضات (Takhfidat - Discounts), تنزيلات (Tanzilat - Price reductions), عروض (Urood - Offers), خصومات (Khusumat)? Reply 'yes' or 'no'. If 'yes', specify where these words are located in the image: {full_text}",
    "Does the image contain any of these words: 'return', إرجاع (Irja' - Return), استبدال (Istibdal - Exchange), استرداد (Istirdad - Refund)? Respond 'yes' or 'no'. If 'yes', locate these words in the image: {full_text}",
    "Does the image contain any of these words: 'free returns', إرجاع مجاني (Irja' majani - Free return), استبدال مجاني (Istibdal majani - Free exchange), استرداد مجاني (Istirdad majani - Free refund)? Answer 'yes' or 'no'. If 'yes', mention where they are found in the image: {full_text}",
    "Does this image contain any of these words: 'delivery', توصيل (Tawseel - Delivery), تسليم (Tasleem - Delivery), شحن (Shahn - Shipping)? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image mention free delivery or توصيل مجاني (Tawseel majani - Free delivery), شحن مجاني (Shahn majani - Free shipping)? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Are they showing images from delivery companies Aramex (أرامكس), FedEx (فيديكس), DHL, and Emirates Post (بريد الإمارات)? Answer 'yes' or 'no'. If 'yes', specify where in the image: {full_text}",
    "Does this image contain images from payment brands Visa, Mastercard, PayPal? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image contain images from payment brands Cash on Delivery (COD) (الدفع عند الاستلام)? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Does the image contain the text 'FAQ', الأسئلة الشائعة (Al-as'ila al-shai'ea), الأسئلة المتكررة (Al-as'ila al-mutakarira)? Answer 'yes' or 'no'. If 'yes', locate these texts in the image: {full_text}",
    "Does the image contain a phone number? Respond 'yes' or 'no'. If 'yes', please indicate its location in the image: {full_text}",
    "Does the image contain a chat support icon? Answer 'yes' or 'no'. If 'yes', specify its location in the image: {full_text}"
],

    "Japan" : [
    "Is there text saying Google レビュー (Google rebyu) or 価格.com (Kakaku.com)? Respond with 'yes' or 'no'. If 'yes', indicate where it is found in the image: {full_text}",
    "Is the text in the image in Japanese? Answer 'yes' or 'no' and briefly explain your reasoning: {full_text}",
    "Does the image contain any of these words: 'sale', セール (se-ru - general sale), バーゲン (ba-gen - bargain sale), 割引 (waribiki - discount), 特売 (tokubai - special sale)? Reply 'yes' or 'no'. If 'yes', specify where these words are located in the image: {full_text}",
    "Does the image contain any of these words: 'return', 返品 (henpin - product return), 交換 (koukan - product exchange), 返金 (henkin - refund)? Respond 'yes' or 'no'. If 'yes', locate these words in the image: {full_text}",
    "Does the image contain any of these words: 'free returns', 返品無料 (henpin muryou - free return shipping), 送料無料返品 (souryou muryou henpin - free return shipping)? Answer 'yes' or 'no'. If 'yes', mention where they are found in the image: {full_text}",
    "Does this image contain any of these words: 'delivery', 配送 (haisou - general delivery), 配達 (haitatsu - delivery service), 発送 (hasou - shipping)? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image mention free delivery or 送料無料 (souryou muryou - free shipping)? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Are they showing images from delivery companies ヤマト運輸 (Yamato unyu), 佐川急便 (Sagawa kyuubin), 日本郵便 (ゆうパック) (Nippon yuubin (yuupakku))? Answer 'yes' or 'no'. If 'yes', specify where in the image: {full_text}",
    "Does this image contain images from payment brands Visa, Mastercard, PayPal? Reply 'yes' or 'no'. If 'yes', indicate their location in the image: {full_text}",
    "Does this image contain images from payment brands 各種クレジットカード (kurejitto ka-do: credit cards), コンビニ決済 (konbini kesai: convenience store payment), 代金引換 (daikin hikikae: cash on delivery)? Respond 'yes' or 'no'. If 'yes', describe where in the image: {full_text}",
    "Does the image contain the text 'FAQ' or 'Fragen und Antworten' or 'Fragen & Antworten'? Answer 'yes' or 'no'. If 'yes', locate these texts in the image: {full_text}",
    "Does the image contain a phone number? Respond 'yes' or 'no'. If 'yes', please indicate its location in the image: {full_text}",
    "Does the image contain a chat support icon? Answer 'yes' or 'no'. If 'yes', specify its location in the image: {full_text}"
]
    }
    return prompts_dict.get(country, [])

def get_prompts_for_country_text(country):
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

    "Brazil " : [
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
]

    
        # ... Add for other countries
    }
    return prompts_dict.get(country, [])


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
        scraper.handle_cookies(url)
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

        # Render the download button for the results
        app_ui.render_download_button(xlsx_data)

    app_ui.render_about_section()
    app_ui.render_footer()

if __name__ == "__main__":
    main()
