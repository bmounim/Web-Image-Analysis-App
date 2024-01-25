# text_detection.py

from google.cloud import vision
import pandas as pd

# text_detection.py

from google.cloud import vision
import pandas as pd
import os 

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GCP_key.json"

class TextDetector:
    def __init__(self):
        """
        Initializes the Google Cloud Vision client. 
        The environment variable GOOGLE_APPLICATION_CREDENTIALS should be set.
        """
        self.client = vision.ImageAnnotatorClient()

    # ... rest of your class methods ...

    def analyze_image_for_text(self, image_data):
        """
        Analyzes the given image for text.
        :param image_data: Image data in bytes.
        :return: Detected text in the image.
        """
        image = vision.Image(content=image_data)
        response = self.client.text_detection(image=image)
        texts = response.text_annotations

        return texts

    def process_detected_text(self, texts):
        """
        Processes the detected texts and checks for certain criteria.
        :param texts: List of detected texts from Google Vision API.
        :return: DataFrame with criteria and their results.
        """
        if not texts:
            print("No text detected in the image.")
            return pd.DataFrame()

        full_text = texts[0].description.lower()

        # Define your criteria for analysis
        criteria = {
            "free delivery": "free delivery" in full_text,
            # Add more criteria as needed
        }

        # Convert the dictionary to a DataFrame
        criteria_df = pd.DataFrame(list(criteria.items()), columns=['Criteria', 'Result (1=Yes, 0=No)'])

        return criteria_df

# Example usage
if __name__ == "__main__":
    detector = TextDetector("GCP_key.json")
    with open("path_to_image.jpg", "rb") as image_file:
        image_data = image_file.read()
    
    detected_texts = detector.analyze_image_for_text(image_data)
    results = detector.process_detected_text(detected_texts)
    print(results)
