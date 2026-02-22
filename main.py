#!/usr/bin/env python
# coding: utf-8


import os
from dotenv import load_dotenv
load_dotenv()  

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
api_key = os.getenv("API_KEY")


import re
import io
import pickle
import pandas as pd
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import speech_recognition as sr
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests  # Added for ElevenLabs API
import tempfile

# =============================
# CONFIGURATION
# =============================
CSV_PATH = r'C:\Users\DELL\PycharmProjects\Ecustomig\customs_faq - customs_faq.csv.csv'
EMBEDDINGS_FILE = 'embeddings_save.pkl'
CATEGORY_FILE = 'cat.txt'


CATEGORY_THRESHOLD = 0.56  # strict for category classification
RETRIEVAL_THRESHOLD = 0.37  # lenient for answer retrieval

# =============================
# INITIALIZATION
# =============================
print("Starting chatbot initialization...")

# Load CSV
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Categories: {df['Kateqoriya'].unique()}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)


# Clean questions
def clean_questions(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-ZəƏçÇəÖöŞşÜüİı0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


df['Sual2'] = df['Sual'].apply(clean_questions)
print("Questions cleaned.")

# Initialize OpenAI client
try:
    client = OpenAI(
        api_key = api_key)
    print("OpenAI client initialized.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit(1)


# =============================
# EMBEDDINGS
# =============================
def get_embedding(text: str) -> np.ndarray:
    try:
        response = client.embeddings.create(model="text-embedding-3-large", input=[str(text)])
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding for '{text}': {e}")
        return np.zeros(3072, dtype=np.float32)


# Load or create embeddings
if os.path.exists(EMBEDDINGS_FILE):
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            saved_data = pickle.load(f)
            df['embeddings'] = saved_data['embeddings']
            print("Embeddings loaded successfully from file.")
    except Exception as e:
        print(f"Error loading embeddings: {e}. Recreating...")
        df['embeddings'] = df['Sual2'].apply(get_embedding)
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump({'embeddings': df['embeddings']}, f)
else:
    df['embeddings'] = df['Sual2'].apply(get_embedding)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'embeddings': df['embeddings']}, f)

# Create FAISS index
embedding_matrix = np.vstack(df['embeddings'].to_numpy()).astype('float32')
faiss.normalize_L2(embedding_matrix)
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embedding_matrix)
print(f"FAISS index created. Dimension: {dimension}")


# =============================
# UTILITY FUNCTIONS
# =============================
def embed_query(text: str) -> np.ndarray:
    text = clean_questions(str(text))
    try:
        response = client.embeddings.create(model="text-embedding-3-large", input=[text])
        return np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)
    except Exception as e:
        print(f"Error embedding query '{text}': {e}")
        return np.zeros((1, 3072), dtype='float32')


def top_k_questions_fixed(user_question, category_df, k=3, similarity_floor=0.52):
    if len(category_df) == 0:
        return [], []

    q_emb = embed_query(user_question)
    category_embeddings = np.vstack(category_df['embeddings'].to_numpy()).astype('float32')
    sims = cosine_similarity(q_emb, category_embeddings)[0]
    idxs = sims.argsort()[::-1]

    final_indices, final_similarities = [], []
    for i in idxs:
        if sims[i] >= similarity_floor:
            final_indices.append(i)
            final_similarities.append(sims[i])
        if len(final_indices) == k:
            break
    return final_indices, final_similarities


def get_best_answer(user_question, category_df, top_indices, similarities):
    if not top_indices or not similarities:
        return "Uyğun cavab tapılmadı."
    best_idx_position = similarities.index(max(similarities))
    best_idx = top_indices[best_idx_position]
    answer = str(category_df.iloc[best_idx]["Cavab"]).strip()
    return answer if answer and answer != 'nan' else "Uyğun cavab tapılmadı."


def classify_max(question, similarity_threshold=CATEGORY_THRESHOLD):
    try:
        q_emb = embed_query(question)
        faiss.normalize_L2(q_emb)
        best_cat, best_sim = "Bilinmir", 0

        for cat in df['Kateqoriya'].unique():
            if pd.isna(cat):
                continue
            cat_df = df[df['Kateqoriya'] == cat]
            if len(cat_df) == 0:
                continue

            cat_embs = np.vstack(cat_df['embeddings'].to_numpy()).astype('float32')
            faiss.normalize_L2(cat_embs)
            sims = cosine_similarity(q_emb, cat_embs)[0]
            max_sim = sims.max()

            if max_sim > best_sim:
                best_sim, best_cat = max_sim, cat
        print(best_sim)
        return best_cat if best_sim >= similarity_threshold else "Bilinmir"
    except Exception as e:
        print(f"Error in classify_max: {e}")
        return "Bilinmir"


def process_question(question: str):
    """Process a question and return category and answer"""
    if not question.strip():
        return "Bilinmir", "Boş sual, yenidən cəhd edin."

    # Load last category if exists
    last_cat = ""
    if os.path.exists(CATEGORY_FILE):
        try:
            with open(CATEGORY_FILE, 'r', encoding='utf-8') as f:
                last_cat = f.read().strip()
        except:
            pass

    # Classify category
    category = classify_max(question)
    category_df = None
    used_last_cat = False

    if category == "Bilinmir":
        if last_cat:
            category_df = df[df['Kateqoriya'] == last_cat].copy().reset_index(drop=True)
            used_last_cat = True
            print(f"Fallback to last category '{last_cat}'")
        else:
            category_df = df.copy().reset_index(drop=True)
            print("No last category found; searching all questions")
    else:
        # Save as last category
        try:
            with open(CATEGORY_FILE, 'w', encoding='utf-8') as f:
                f.write(category)
        except:
            pass
        category_df = df[df['Kateqoriya'] == category].copy().reset_index(drop=True)

    # Get top similar questions
    top_indices, similarities = top_k_questions_fixed(
        question, category_df, k=3, similarity_floor=RETRIEVAL_THRESHOLD
    )

    # If no matches and used last category, search all questions
    if not top_indices and used_last_cat:
        print("No matches in last category. Searching all questions...")
        category_df = df.copy().reset_index(drop=True)
        top_indices, similarities = top_k_questions_fixed(
            question, category_df, k=3, similarity_floor=RETRIEVAL_THRESHOLD
        )

    # Get best answer
    answer = get_best_answer(question, category_df, top_indices, similarities)

    # Determine display category
    if category != "Bilinmir":
        display_category = category
    elif used_last_cat:
        display_category = last_cat
    else:
        display_category = "Bilinmir"

    return display_category, answer


# =============================
# ENHANCED SPEECH-TO-TEXT WITH ELEVENLABS
# =============================
def audio_to_text_elevenlabs(audio_bytes):
    """Convert audio to text using ElevenLabs STT (best for Azerbaijani)"""
    try:
        audio_bytes.seek(0)
        files = {"file": ("voice.ogg", audio_bytes, "audio/ogg")}
        data = {"model_id": "scribe_v1"}

        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVEN_API_KEY},
            files=files,
            data=data
        )

        if response.status_code == 200:
            result = response.json()
            text = result.get("text", "")
            print(f"ElevenLabs transcription successful: '{text}'")
            return text
        else:
            print(f"ElevenLabs STT error: {response.status_code} - {response.text}")
            return ""

    except Exception as e:
        print(f"ElevenLabs STT error: {e}")
        return ""


def audio_to_text_whisper(audio_bytes):
    """Fallback: Convert audio to text using OpenAI Whisper"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
            audio_bytes.seek(0)
            temp_file.write(audio_bytes.read())
            temp_file_path = temp_file.name

        with open(temp_file_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="az"
            )

        try:
            os.unlink(temp_file_path)
        except:
            pass

        return response.text
    except Exception as e:
        print(f"OpenAI Whisper STT error: {e}")
        return ""


def audio_to_text_google(audio_bytes):
    """Second fallback: Use Google Speech Recognition for Azerbaijani"""
    try:
        recognizer = sr.Recognizer()

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio_bytes.seek(0)
            temp_file.write(audio_bytes.read())
            temp_file_path = temp_file.name

        language_codes = ['az-AZ', 'az', 'tr-TR']

        with sr.AudioFile(temp_file_path) as source:
            audio = recognizer.record(source)

        for lang_code in language_codes:
            try:
                text = recognizer.recognize_google(audio, language=lang_code)
                print(f"Google SR successful with language: {lang_code}")
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                return text
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Google SR error for {lang_code}: {e}")
                continue

        try:
            os.unlink(temp_file_path)
        except:
            pass

        return ""
    except Exception as e:
        print(f"Google Speech Recognition error: {e}")
        return ""


def audio_to_text(audio_bytes):
    """Enhanced audio to text conversion with multiple fallbacks"""
    if not audio_bytes:
        return ""

    # First try ElevenLabs (best for Azerbaijani)
    print("Attempting transcription with ElevenLabs...")
    text = audio_to_text_elevenlabs(audio_bytes)

    if text.strip():
        print(f"ElevenLabs transcription successful: '{text}'")
        return text.strip()

    # Fallback to OpenAI Whisper
    print("Fallback to OpenAI Whisper...")
    text = audio_to_text_whisper(audio_bytes)

    if text.strip():
        print(f"Whisper transcription successful: '{text}'")
        return text.strip()

    # Final fallback to Google Speech Recognition
    print("Final fallback to Google Speech Recognition...")
    text = audio_to_text_google(audio_bytes)

    if text.strip():
        print(f"Google SR transcription successful: '{text}'")
        return text.strip()

    print("No transcription available from any service")
    return ""


# =============================
# TELEGRAM HANDLERS
# =============================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Salam! Gömrük haqqında sualınızı yazın və ya səsli mesaj göndərin. 🤖"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    question = update.message.text.strip()
    if not question:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Boş sual, zəhmət olmasa yenidən cəhd edin."
        )
        return

    try:
        category, answer = process_question(question)
        response = f"📂 Kateqoriya: {category}\n💬 Cavab: {answer}"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    except Exception as e:
        print(f"Error processing text: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Xəta baş verdi, zəhmət olmasa yenidən cəhd edin."
        )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages with enhanced Azerbaijani support"""
    try:
        # Get the voice message file
        file = await context.bot.get_file(update.message.voice.file_id)

        # Download the file to memory
        audio_bytes = io.BytesIO()
        await file.download_to_memory(audio_bytes)
        audio_bytes.seek(0)

        # Convert audio to text using enhanced STT with ElevenLabs
        text = audio_to_text(audio_bytes)

        if not text:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Səs tanınmadı, zəhmət olmasa yenidən cəhd edin."
            )
            return

        # Process the transcribed text
        category, answer = process_question(text)
        response = f"🎤 Sual: {text}\n📂 Kateqoriya: {category}\n💬 Cavab: {answer}"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

    except Exception as e:
        print(f"Error processing audio: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Səsli mesaj emal edilərkən xəta baş verdi."
        )


# =============================
# MAIN FUNCTION
# =============================
def main():
    """Start the bot"""
    # Ensure category file exists
    if not os.path.exists(CATEGORY_FILE):
        with open(CATEGORY_FILE, 'w', encoding='utf-8') as f:
            f.write("")

    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.VOICE, handle_audio))

    print("Bot started and ready to receive messages...")

    # Start the bot
    application.run_polling()


if __name__ == '__main__':
    main()