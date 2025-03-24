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
git clone https://github.com/arbml/masader-bot.git
cd masader-bot
pip install -r requirements.txt
```

## Validation
```bash
model_name="gemini-flash-1.5"
for schema in ar
do
    python evaluate.py --models $model_name -mv --schema $schema -o --few_shot 0 --results_path results_pdf_docling --use_pdf --pdf_mode docling
    python evaluate.py --models $model_name -mv --schema $schema -o --few_shot 0 --results_path results_pdf_plumber --use_pdf --pdf_mode plumber
    python evaluate.py --models $model_name -mv --schema $schema -o --few_shot 0 --results_path results_latex
done

```