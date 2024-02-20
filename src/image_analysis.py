# image_analysis.py
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import os 
import pandas as pd
from io import BytesIO
#from PIL import Image
import google.ai.generativelanguage as glm
import google.generativeai as genai
import io
import streamlit as st
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GCP_keys3.json"
#from PIL import Image
from google.cloud import aiplatform


from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    Image,
)

from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import Part
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

PIL_Image.MAX_IMAGE_PIXELS = None  # Disable the limit (not recommended for untrusted images)


def split_image_horizontally(image_path, num_splits):
    """
    Splits an image into a specified number of horizontal segments.

    Args:
    - image_path: Path to the input image.
    - num_splits: Number of horizontal segments to split the image into.

    Returns
    - List of paths to the saved horizontal segments of the original image.
    """
    image = PIL_Image.open(image_path)
    width, height = image.size
    split_width = width // num_splits  # Calculate the width of each split.
    split_image_paths = []

    for i in range(num_splits):
        # Define the bounding box for the current split.
        bbox = (i * split_width, 0, (i + 1) * split_width if (i + 1) < num_splits else width, height)
        split_image = image.crop(bbox)
        split_image_path = f'horizontal_split_image_{i}_{image_path}.png'
        split_image.save(split_image_path)
        split_image_paths.append(split_image_path)

    return split_image_paths


def further_split_images_horizontally(split_image_paths, num_horizontal_splits):
    """
    Applies horizontal splitting to each image path in split_image_paths.

    Args:
    - split_image_paths: List of paths to images that were previously split vertically.
    - num_horizontal_splits: Number of horizontal segments to split each image into.

    Returns:
    - A nested list where each sublist contains paths to the horizontal segments of the original vertical segment.
    """
    all_horizontal_splits_paths = []

    for vertical_split_path in split_image_paths:
        horizontal_split_paths = split_image_horizontally(vertical_split_path, num_horizontal_splits)
        all_horizontal_splits_paths.append(horizontal_split_paths)

    return all_horizontal_splits_paths

def split_image_vertically(image_path, num_splits):

    
    image = PIL_Image.open(image_path)
    width, height = image.size
    
    split_height = height // num_splits
    split_image_paths = []

    for i in range(num_splits):
        bbox = (0, i * split_height, width, (i + 1) * split_height if (i + 1) < num_splits else height)
        split_image = image.crop(bbox)
        split_image_path = f'split_image_{i}.png'
        split_image.save(split_image_path)
        split_image_paths.append(split_image_path)

    return split_image_paths

def zoom_image(image_path, zoom_factor):
    image = PIL_Image.open(image_path)
    width, height = image.size
    new_size = (int(width * zoom_factor), int(height * zoom_factor))
    zoomed_image = image.resize(new_size, PIL_Image.LANCZOS)
    zoomed_image_path = f'zoomed_{image_path.split("/")[-1]}'
    zoomed_image.save(zoomed_image_path)

    return zoomed_image_path


def init_vertex_ai(project_id, region):
    vertexai.init(project=project_id, location=region,api_endpoint='us-central1-aiplatform.googleapis.com')

def initialize_model():
    genai.configure(api_key='AIzaSyCtuEVO7fQKttTO7MHGa0vFKs2E6MV9do4')
    return genai.GenerativeModel('gemini-pro-vision')

generationConfig = {
    "temperature" : 0.3,
    "max_output_tokens": 100
}

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
]



import logging
def analyze_image2(prompt, image_path):
    response_text=''
    try : 
        # Open the image from the path
        image = PIL_Image.open(image_path)
        
        # Check the image size and thumbnail if necessary
        width, height = image.size
        if width > 1024 or height > 1024:
            image.thumbnail((1024, 1024))
        
        # Convert the image to RGB if it's not already in that mode
        if image.mode in ("RGBA", "LA"):
            background = PIL_Image.new(image.mode[:-1], image.size, (255, 255, 255))
            background.paste(image, image.split()[-1])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
        
        # Save the processed image
        save_path = "save_path.jpg"
        image.save(save_path)

        # Assuming there's a way to load the image into your model, 
        # which might look something like this
        # Note: This part is pseudo-code since the exact implementation depends on your setup
        image_for_model = Image.load_from_file(save_path)  # This needs to be replaced with your actual method

        # Again assuming the existence of a GenerativeModel class
        gemini_pro_vision_model = GenerativeModel("gemini-1.0-pro-vision")

        # Generate content using the model
        # This is also pseudo-code and will depend on the specifics of your library/setup
        model_response = gemini_pro_vision_model.generate_content([prompt, image_for_model], stream=True)
        
        # Process model response
        temp_list = [response.text for response in model_response]  # Assuming this is how you get text from responses
        response_text = ''.join(temp_list)
   
    except ValueError as e:
            # Handle the case where the model response contains no parts
            logging.error(f"Failed to extract text from model response for {image_path}: {e}")
            response_text=f"An error occurred with {image_path}: could not analyze image."

    except Exception as e:
            # Handle any other exceptions
            logging.error(f"An unexpected error occurred with {image_path}: {e}")
            response_text=f"An unexpected error occurred during image analysis of {image_path}."

    return response_text




def analyze_image(model, prompt, image):
        with PIL_Image.open(image) as img:
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
        print('mounim')
        print(prompt)
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
        safety_settings=safety_settings,
        stream=True,
        generation_config =generationConfig # Setting a maximum output token limit
)
        
                

        print(response)
        #response = model.generate_content([prompt, image])
        #response.resolve()
        
        #response_text=response.text
        
        temp_list = [response2.text for response2 in response]
        response_text = ''.join(temp_list)

        return response_text
    
def process_response(response_text):
    yes_no = "yes" if "yes" in response_text.lower() else "no" if "no" in response_text.lower() else "unknown"
    return {"yes or no": yes_no, "additional_infos": response_text}

def analyze_image_for_criteria(image_file, project_id, region,prompts):
    #init_vertex_ai(project_id,region)
    split_image_paths=split_image_vertically(image_file, 10)
    num_horizontal_splits=2
    all_horizontal_splits = further_split_images_horizontally(split_image_paths, num_horizontal_splits)

    all_data=[]
    for image2 in all_horizontal_splits :
        for image in image2 : 
            
            #init_verftex_ai(project_id, region)
            #image = Image.open(image_file)
            model = initialize_model()
            #image= zoom_image(image,50)
            prompts = prompts


            data = []
            
            for prompt in prompts:
                response_text = analyze_image2(prompt, image)
                processed_data = process_response(response_text)
                processed_data["criteria"] = prompt  # Moving this line here to adjust the column order
                row = {"criteria": prompt, "yes or no": processed_data["yes or no"], "additional_infos": processed_data["additional_infos"]}
                data.append(row)
            data = pd.DataFrame(data)
            all_data.append(data)

    return all_data,all_horizontal_splits
