import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # Secret key for session management and CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-development-only'
    
    # Upload folder for safetensor files
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'safetensors'}
    
    # Maximum file size (500MB)
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024
