# image_analysis.py
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import os 
import pandas as pd
from io import BytesIO
import google.ai.generativelanguage as glm
import google.generativeai as genai
import io
import streamlit as st
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Image,
    Part,
)

Image.LOAD_TRUNCATED_IMAGES = True

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GCP_keys.json"
def split_image_vertically(image_path, num_splits):
    image = Image.open(image_path)
    split_height = image.height // num_splits
    split_image_paths = []

    for i in range(num_splits):
        bbox = (0, i * split_height, image.width, (i + 1) * split_height if (i + 1) < num_splits else image.height)
        split_image = image.crop(bbox)
        split_image_path = f'split_image_{i}.png'
        split_image.save(split_image_path)
        split_image_paths.append(split_image_path)

    return split_image_paths

def zoom_image(image_path, zoom_factor):
    image = Image.open(image_path)
    new_size = (int(image.width * zoom_factor), int(image.height * zoom_factor))
    zoomed_image = image.resize(new_size, Image.LANCZOS)
    zoomed_image_path = f'zoomed_{image_path.split("/")[-1]}'
    zoomed_image.save(zoomed_image_path)

    return zoomed_image_path


def init_vertex_ai(project_id, region):
    vertexai.init(project=project_id, location=region,api_endpoint='us-central1-aiplatform.googleapis.com')

def initialize_model():
    #genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    return GenerativeModel("gemini-pro-vision")

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

def analyze_image(model, prompt, image):
        #bytes_data = image.getvalue()
        with Image.open(image) as img:
            img_format = img.format  # Preserve the original format

            # Compress or resize the image if needed
            # Example: Resize if width or height is greater than a certain value
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024))

            # Convert to bytes
            bytes_io = io.BytesIO()
            img.save(bytes_io, format=img_format, quality=85)  # Adjust quality for size
            bytes_data = bytes_io.getvalue()
        
            # Check size
        if len(bytes_data) > 4194304:  # 4 MB
            raise ValueError("Image size after compression is still too large.")


        image = Image.load_from_file(image)
        contents=[image,prompt]
        response = model.generate_content(contents,   
        safety_settings=safety_settings,stream=True)

        print(response)
        #response = model.generate_content([prompt, image])
        #response.resolve()
        b=[]
        for a in response : 
            a=a.text
            b.append(a)
        return ''.join(b)
    
def process_response(response_text):
    yes_no = "yes" if "yes" in response_text.lower() else "no" if "no" in response_text.lower() else "unknown"
    return {"yes or no": yes_no, "additional_infos": response_text}

def analyze_image_for_criteria(image_file, project_id, region,prompts):
    
    split_image_paths=split_image_vertically(image_file, 3)
    all_data=[]
    for image in split_image_paths : 
        
        init_vertex_ai(project_id, region)
        #image = Image.open(image_file)
        model = initialize_model()
        image= zoom_image(image,3)
        prompts = prompts


        data = []
        
        for prompt in prompts:
            response_text = analyze_image(model, prompt, image)
            processed_data = process_response(response_text)
            processed_data["criteria"] = prompt  # Moving this line here to adjust the column order
            row = {"criteria": prompt, "yes or no": processed_data["yes or no"], "additional_infos": processed_data["additional_infos"]}
            data.append(row)
        data = pd.DataFrame(data)
        all_data.append(data)

    return all_data,split_image_paths
