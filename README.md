# Masader Bot

## Overview
A Python script that searches arXiv for research papers, extracts dataset metadata using AI models, and validates the extracted information using Masader Examples.

## Features
- Search arXiv papers by keywords, month, and year
- Extract detailed metadata using AI models (Claude, ChatGPT, Gemini)
- Validate extracted metadata against a reference dataset
- Clean and process LaTeX/PDF papers
- Compute extraction costs and filling ratio

## Requirements
- Python 3.8+
- Dependencies: 
  - anthropic
  - openai
  - google-generativeai
  - arxiv
  - pdfplumber
  - python-dotenv

## Installation
```bash
git clone https://github.com/yourusername/research-metadata-extractor.git
cd research-metadata-extractor
pip install -r requirements.txt
```

## Usage
```bash
python script.py -k "arabic,dataset" -m 3 -y 2024 -n claude-3-5-sonnet-latest
```

### Arguments
- `-k, --keywords`: Comma-separated research keywords
- `-m, --month`: Search month (1-12)
- `-y, --year`: Search year
- `-n, --model_name`: AI model to use (optional)
- `-c, --check_abstract`: Whether to validate abstract (optional)

## Output
Generates a `results.json` with:
- Extracted metadata
- Extraction cost
- Validation score
- Configuration details

## License
[Add your license here]