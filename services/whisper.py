import re
import logging
from whisper.model import Whisper
from whisper import load_model


def transcribe_whisper(model, file_path, language="ru"):
    """Transcribe audio file using Whisper model"""
    try:
        whisper_model: Whisper = model
        logging.info("Audio recognition started (whisper)")
        result = whisper_model.transcribe(file_path, verbose=False, language=language, fp16=False)
        rawtext = " ".join([segment["text"].strip() for segment in result["segments"]])
        rawtext = re.sub(" +", " ", rawtext)
        alltext = re.sub(r"([\.\!\?]) ", r"\1\n", rawtext)
        logging.info(f"Recognized: {alltext}")
        return alltext
    except Exception as e:
        logging.error(f"Error in whisper transcription: {e}")
        return ""


def load_whisper_model(lang_code) -> Whisper:
    """Load Whisper model for specified language"""
    try:
        # Use base model by default, can be configured
        whisper_model_name = "base"
        if lang_code == "en":
            whisper_model_name = "base.en"

        logging.info(f"Loading Whisper model: {whisper_model_name}")
        model = load_model(whisper_model_name)
        logging.info("Whisper model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading Whisper model: {e}")
        # Fallback to tiny model if base fails
        try:
            logging.info("Trying to load tiny model as fallback")
            return load_model("tiny")
        except Exception as e2:
            logging.error(f"Error loading fallback model: {e2}")
            raise e2