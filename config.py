# config.py

import os
from utils import get_env_variable

# Google Cloud settings
GOOGLE_PROJECT_ID = get_env_variable('GOOGLE_PROJECT_ID', "jovial-circuit-412017")
GOOGLE_APPLICATION_CREDENTIALS = get_env_variable('GOOGLE_APPLICATION_CREDENTIALS', "GCP_key.json")

# Vertex AI settings
VERTEX_AI_REGION = get_env_variable('VERTEX_AI_REGION', 'us-central1')

# Add other configuration settings as needed
