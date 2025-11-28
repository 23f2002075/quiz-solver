# Quiz Solver Agent

An autonomous AI agent that solves multi-level quizzes using FastAPI, LangGraph, and Google Gemini.

## Features

- 🤖 **Autonomous Agent**: Uses LangGraph state machine for intelligent decision-making
- 🔄 **Round-robin API Keys**: Automatic failover with multiple Gemini API keys
- ⏱️ **3-minute Timer per Question**: Retry incorrect answers within time limit
- 🛠️ **10 Specialized Tools**:
  - HTML rendering (Selenium)
  - PDF text extraction
  - Audio transcription
  - Image OCR (Tesseract)
  - Steganography extraction
  - Code execution
  - File operations
  - And more...
- 📊 **Detailed Logging**: Structured output with [TAG] format

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **AI**: Google Gemini 2.5 Flash
- **Graph**: LangGraph + LangChain
- **Processing**: Selenium, Tesseract, librosa, pandas, pdfplumber
- **Language**: Python 3.12

## Local Setup

### Prerequisites
- Python 3.12+
- Chrome/Chromium browser
- Tesseract OCR

### Installation

1. Clone repository:
```bash
git clone https://github.com/YOUR_USERNAME/quiz-solver.git
cd quiz-solver
```

2. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment:
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. Run server:
```bash
python app.py
```

Server runs on `http://localhost:8000`

## Configuration

### Environment Variables (.env)
```
EMAIL=your_email@example.com
SECRET=your_secret_key
GOOGLE_API_KEY=your_gemini_api_key_1
GOOGLE_API_KEY_2=your_gemini_api_key_2
GOOGLE_API_KEY_3=your_gemini_api_key_3
```

### Get Gemini API Keys
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy and add to `.env`

## API Endpoints

### Health Check
```bash
GET /healthz
```
Response: `{"status": "ok", "uptime_seconds": 123}`

### Solve Quiz
```bash
POST /solve
Content-Type: application/json

{
  "email": "your_email@example.com",
  "secret": "your_secret",
  "url": "https://quiz-url.com/q1"
}
```

## How It Works

1. **Request**: Quiz URL is submitted via `/solve` endpoint
2. **Agent Loop**: 
   - Fetches HTML page
   - Extracts question and submission details
   - Uses appropriate tools to solve
   - Submits answer
3. **Retry Logic**: 
   - If incorrect: Retries within 3-minute window
   - If correct: Moves to next question
4. **Completion**: Stops when no more URLs available

## Retry Strategy

- **Per-Question Timer**: 3 minutes per question
- **API Key Rotation**: Automatic failover on quota exhaustion
- **Max Retries**: Limited by number of API keys configured
- **Early Stop**: Moves to next question after timer expires

## Project Structure

```
.
├── app.py              # FastAPI server & endpoints
├── agent.py            # LangGraph agent & routing logic
├── tools.py            # 10 specialized tools
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
├── Procfile            # Process file for deployment
├── render.yaml         # Render.com configuration
└── .env                # Environment variables (not in git)
```

## Deployment

### Render.com

1. Push to GitHub
2. Connect GitHub to Render
3. Create Web Service from repository
4. Add environment variables
5. Deploy!

### Docker

```bash
docker build -t quiz-solver .
docker run -p 8000:8000 --env-file .env quiz-solver
```

## Testing

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "secret": "secret123",
    "url": "https://quiz-site.com/q1"
  }'
```

## Monitoring

- Check logs in terminal (development)
- Monitor `/healthz` endpoint for uptime
- Use Render dashboard for production logs

## Known Limitations

- Free tier on Render: Service spins down after 15 min inactivity
- Audio transcription requires internet (Google Speech Recognition)
- OCR accuracy depends on image quality
- Steganography: LSB extraction only

## Performance

- **Level 1-3**: Simple calculations - instant
- **Level 4**: OCR on images - 5-10s
- **Level 5**: Audio transcription - 10-30s
- **Level 6+**: Complex data analysis - varies

## Troubleshooting

### Server won't start
- Check Python version: `python --version` (need 3.12+)
- Verify all dependencies: `pip install -r requirements.txt`
- Check port 8000 is available

### API key quota exceeded
- Add more API keys to `.env`
- System auto-rotates keys
- Wait 24 hours for quota reset

### Tools failing
- Tesseract not found: Install `tesseract-ocr` system package
- Chrome issues: Ensure Chrome is installed
- Audio errors: Install `ffmpeg` for audio processing

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/improvement`
3. Commit changes: `git commit -am 'Add improvement'`
4. Push to branch: `git push origin feature/improvement`
5. Submit pull request

## License

MIT License - see LICENSE file for details

## Author

Created for IITM online degree program

## Support

- Issues: GitHub Issues
- Email: your_email@example.com
- Render Dashboard: Monitor logs and metrics
