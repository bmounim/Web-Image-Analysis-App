# image_analysis.py
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image
import os 
import pandas as pd
from io import BytesIO
from PIL import Image
import google.ai.generativelanguage as glm
import google.generativeai as genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GCP_key.json"

def init_vertex_ai(project_id, region):
    vertexai.init(project=project_id, location=region,api_endpoint='us-central1-aiplatform.googleapis.com')

def initialize_model():
    return genai.GenerativeModel("gemini-pro-vision")

def analyze_image(model, prompt, image):
        #bytes_data = image.getvalue()

        image = Image.open(image)

        bytes_data = BytesIO()
        image.save(bytes_data, format='PNG')
        bytes_data = bytes_data.getvalue()



        response = model.generate_content(
        glm.Content(
            parts = [
                glm.Part(text=prompt),
                glm.Part(
                    inline_data=glm.Blob(
                        mime_type='image/png',
                        data=bytes_data
                    )
                ),
            ],
        ),      
        stream=True)

        print(response)
        #response = model.generate_content([prompt, image])
        response.resolve()
        return response
    
def process_response(response_text):
    yes_no = "yes" if "yes" in response_text.lower() else "no" if "no" in response_text.lower() else "unknown"
    return {"yes or no": yes_no, "additional_infos": response_text}

def analyze_image_for_criteria(image_file, project_id, region,prompts):
    init_vertex_ai(project_id, region)
    #image = Image.open(image_file)
    model = initialize_model()
    image=image_file
    prompts = prompts


    data = []
    for prompt in prompts:
        response_text = analyze_image(model, prompt, image)
        processed_data = process_response(response_text)
        processed_data["criteria"] = prompt  # Moving this line here to adjust the column order
        row = {"criteria": prompt, "yes or no": processed_data["yes or no"], "additional_infos": processed_data["additional_infos"]}
        data.append(row)
    data = pd.DataFrame(data)
    return data
