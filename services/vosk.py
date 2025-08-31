import os
import json
import logging
from vosk import KaldiRecognizer, Model


def load_vosk_model(lang_code):
    """Load Vosk model for specified language"""
    try:
        # Default model paths - adjust these based on your setup
        model_paths = {
            'en': './vosk_models/vosk-model-en-us-0.22',
            'ru': './vosk_models/vosk-model-ru-0.42',
            'az': './vosk_models/vosk-model-small-az-0.3',  # if you have Azerbaijani model
            'tr': './vosk_models/vosk-model-small-tr-0.3',  # if you have Turkish model
        }

        model_path = model_paths.get(lang_code, './vosk_models/vosk-model-small-en-us-0.15')

        if not os.path.exists(model_path):
            logging.warning(f"Vosk model not found at {model_path}. Using fallback.")
            # Try to find any available model
            vosk_base = './vosk_models'
            if os.path.exists(vosk_base):
                for item in os.listdir(vosk_base):
                    potential_path = os.path.join(vosk_base, item)
                    if os.path.isdir(potential_path):
                        model_path = potential_path
                        break

        logging.info(f"Loading Vosk model from: {model_path}")
        model = Model(model_path)
        logging.info("Vosk model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading Vosk model: {e}")
        raise e


def transcribe_vosk(file_path, model):
    """Transcribe audio file using Vosk model"""
    try:
        logging.info('Audio recognition started (vosk)')

        with open(file_path, 'rb') as file:
            audio_data = file.read()

        recognizer = KaldiRecognizer(model, 16000)
        recognizer.AcceptWaveform(audio_data)
        result_json = recognizer.FinalResult()
        result = json.loads(result_json)
        text = result['text']

        logging.info(f'Recognized: {text}')
        return text
    except Exception as e:
        logging.error(f"Error in Vosk transcription: {e}")
        return ""