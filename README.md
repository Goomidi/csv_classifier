# CSV Text Classification System

A powerful, user-friendly text classification system built with Streamlit and OpenAI. This application allows non-technical users to upload data, define custom categories, and apply AI-powered classification to their text data.

## Features

- 📊 **Intuitive UI**: Easy-to-use interface for uploading and classifying data
- 🧠 **AI-Powered Classification**: Leverages advanced language models for accurate text classification
- 🔍 **Transparent Explanations**: Provides detailed explanations for each classification
- 📱 **Visual Analytics**: Visualize category distributions and confidence metrics
- 🛠️ **User Correction Mode**: Review and improve classifications with feedback
- 📋 **Predefined Categories**: Use built-in category templates or define your own

## Requirements

- Python 3.12+
- Docker (optional, for containerized deployment)
- API keys for:
  - OpenAI

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/goomidi/csv_classifier.git
cd csv_classifier
```

2.  **Set up a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install uv
uv sync
```

4. **Set up environment variables**

Copy `.env.example` into a `.env` file in the project root and modify the following variables:

```
LITE_LLM_BASE_URL=your_lite_llm_url
OPENAI_API_KEY=your_openai_api_key
MODEL=your_model_for_classification
```

5. **Update default categories**

```bash
sudo vim data/categories.json
```

## Running the Application

### Local Development

```bash
streamlit run app.py
```

### Docker

```bash
docker compose up -d
```

## Usage

1. Access the application at `http://localhost:8501`
2. Follow the step-by-step process:
   - Upload your data (CSV or Excel file)
   - Define categories or use predefined templates
   - Run the classification process
   - Review results with visualizations
   - Correct any misclassifications (optional)
   - Download the results

## Project Structure

```
allobrain/
├── app.py                  # Main Streamlit application
├── data/                   # Data directory for predefined categories
├── src/
│   ├── classification.py   # Text classification logic
│   ├── core/               # Core configuration and settings
│   ├── evaluation.py       # Performance evaluation tools
│   ├── explanation.py      # Results explanation utilities
│   ├── schemas/            # Data validation schemas
│   └── utils/              # Utility functions
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Docker build instructions
├── pyproject.toml          # Project metadata and dependencies
└── .env.example            # Example environment variables
```

## License

[MIT License](LICENSE)
