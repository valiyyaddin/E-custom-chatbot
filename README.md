# E-Custom Chatbot 🤖

An intelligent Telegram chatbot for answering customs-related questions in Azerbaijani. The bot uses OpenAI embeddings for semantic search and supports both text and voice messages with multiple speech-to-text engines.

## Features ✨

- **Text Message Support**: Ask questions via text and receive categorized answers
- **Voice Message Support**: Send voice messages in Azerbaijani
- **Multi-Engine Speech Recognition**: 
  - ElevenLabs Speech-to-Text (primary)
  - OpenAI Whisper (fallback)
  - Google Speech Recognition (secondary fallback)
- **Smart Category Classification**: Automatically categorizes questions using semantic similarity
- **Semantic Search**: Uses OpenAI embeddings and FAISS for fast, accurate answer retrieval
- **Context Awareness**: Remembers last category for better follow-up question handling

## Technology Stack 🛠️

- **Python 3.x**
- **OpenAI API**: Text embeddings (text-embedding-3-large)
- **FAISS**: Vector similarity search
- **Telegram Bot API**: User interface
- **ElevenLabs API**: Azerbaijani speech-to-text
- **Pandas & NumPy**: Data processing
- **scikit-learn**: Cosine similarity calculations

## Installation 📦

### 1. Clone the Repository

```bash
git clone https://github.com/valiyyaddin/E-custom-chatbot.git
cd E-custom-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
ELEVEN_API_KEY=your_elevenlabs_api_key
API_KEY=your_openai_api_key
```

### 4. Prepare Data

Place your customs FAQ CSV file in the project directory. The CSV should have the following columns:
- `Sual`: Questions
- `Cavab`: Answers
- `Kateqoriya`: Categories

Update the `CSV_PATH` in `main.py` to point to your CSV file.

## Usage 🚀

### Start the Bot

```bash
python main.py
```

### Interact with the Bot

1. Open Telegram and search for your bot
2. Send `/start` to begin
3. Ask questions in text or send voice messages
4. Receive categorized answers instantly

### Example Interactions

**Text Message:**
```
User: Gömrük rüsumu necə ödənilir?
Bot: 📂 Kateqoriya: Ödəniş
     💬 Cavab: [Detailed answer about customs payment]
```

**Voice Message:**
```
User: [Sends voice message]
Bot: 🎤 Sual: Gömrük rüsumu necə ödənilir?
     📂 Kateqoriya: Ödəniş
     💬 Cavab: [Detailed answer]
```

## Configuration ⚙️

### Similarity Thresholds

Adjust these in `main.py`:

```python
CATEGORY_THRESHOLD = 0.56  # Strictness for category classification
RETRIEVAL_THRESHOLD = 0.37  # Leniency for answer retrieval
```

### File Paths

```python
CSV_PATH = 'path/to/your/customs_faq.csv'
EMBEDDINGS_FILE = 'embeddings_save.pkl'  # Cached embeddings
CATEGORY_FILE = 'cat.txt'  # Last category cache
```

## Architecture 🏗️

### Core Components

1. **Embedding System**: 
   - Generates OpenAI embeddings for all questions
   - Caches embeddings for performance
   - Uses FAISS index for fast similarity search

2. **Question Processing Pipeline**:
   ```
   User Question → Clean Text → Generate Embedding → 
   Classify Category → Search Similar Questions → 
   Return Best Answer
   ```

3. **Speech Recognition Chain**:
   ```
   Voice Message → ElevenLabs STT → 
   [If fails] → Whisper STT → 
   [If fails] → Google SR → Text
   ```

4. **Category Memory**:
   - Saves last successfully classified category
   - Uses for context in ambiguous questions
   - Falls back to full search if needed

## Project Structure 📁

```
E-custom-chatbot/
├── main.py                           # Main bot application
├── requirements.txt                  # Python dependencies
├── customs_faq - customs_faq.csv.csv # FAQ dataset
├── .gitignore                       # Git ignore rules
├── .env                             # Environment variables (create this)
├── embeddings_save.pkl              # Cached embeddings (auto-generated)
├── cat.txt                          # Last category cache (auto-generated)
├── config/                          # Configuration files
└── services/                        # Additional services
    ├── vosk.py                      # Vosk STT service
    ├── whisper.py                   # Whisper STT service
    └── filesUtils.py                # File utilities
```

## API Keys Required 🔑

1. **Telegram Bot Token**: Get from [@BotFather](https://t.me/botfather)
2. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com)
3. **ElevenLabs API Key**: Get from [ElevenLabs](https://elevenlabs.io)

## Performance Optimization 💡

- Embeddings are cached in `embeddings_save.pkl` to avoid regeneration
- FAISS index enables fast similarity search (O(log n))
- Category memory reduces search space for follow-up questions
- Multiple STT engines ensure high transcription success rate

## Error Handling 🛡️

The bot includes comprehensive error handling:
- Graceful fallbacks for STT failures
- Category classification fallbacks
- User-friendly error messages in Azerbaijani
- Detailed logging for debugging

## Limitations ⚠️

- CSV file path is currently hardcoded
- Embeddings regeneration required for data updates
- Voice messages require adequate audio quality
- Azerbaijani language optimization for speech recognition

## Future Enhancements 🚀

- [ ] Support for multiple languages
- [ ] Admin panel for FAQ management
- [ ] Analytics dashboard
- [ ] Conversation history
- [ ] Feedback system
- [ ] Docker containerization
- [ ] Database integration (replace CSV)

## Contributing 🤝

Contributions are welcome! Please feel free to submit issues or pull requests.

## License 📄

This project is open source and available under the [MIT License](LICENSE).

## Support 💬

For questions or support, please open an issue in the GitHub repository.




