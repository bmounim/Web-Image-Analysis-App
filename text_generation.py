# text_generation.py

import vertexai
from vertexai.language_models import TextGenerationModel
import pandas as pd

class TextGenerator:
    def __init__(self, project_id, location):
        """
        Initializes Vertex AI with the given project ID and location.
        :param project_id: Google Cloud project ID.
        :param location: Google Cloud location for Vertex AI.
        """
        vertexai.init(project=project_id, location=location)
        self.model = TextGenerationModel.from_pretrained("text-bison@001")

    def generate_text_responses(self, prompts, parameters):
        """
        Generates text responses for a list of prompts using Vertex AI.
        :param prompts: List of prompts for text generation.
        :param parameters: Parameters for text generation (like temperature, max tokens).
        :return: List of generated text responses.
        """
        responses = []
        for prompt in prompts:
            response = self.model.predict(prompt, **parameters)
            responses.append(response.text)
        return responses

    def process_responses(self, responses, criteria):
        """
        Processes the generated text responses.
        :param responses: List of generated text responses.
        :param criteria: List of criteria corresponding to each response.
        :return: DataFrame with processed responses and criteria.
        """
        results = []
        for response_text, criterion in zip(responses, criteria):
            yes_no = "yes" if "yes" in response_text.lower() else "no" if "no" in response_text.lower() else "unknown"
            results.append({
                "criteria": criterion,
                "yes or no": yes_no,
                "additional_infos": response_text
            })

        return pd.DataFrame(results)
