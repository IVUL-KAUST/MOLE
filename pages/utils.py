import os
import base64
import requests
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import PdfFormatOption


def get_paper_content_from_docling(paper_path, output_mode="markdown"):
    pipeline_options = PdfPipelineOptions()

    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True

    pipeline_options.do_ocr = True
    pipeline_options.ocr_options.lang = ["en"]

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert with limits to prevent memory issues
    result = converter.convert(paper_path, max_num_pages=200)

    # Use the resulting document
    doc = result.document
    markdown = doc.export_to_markdown()
    html = doc.export_to_html()
    if output_mode == "markdown":
        return markdown
    elif output_mode == "html":
        return html
    else:
        raise ValueError("output_mode must be 'markdown' or 'html'")

def process_pdf_with_llm(pdf_path, prompt="Can you please summarize this paper to me in no more than 5 lines?", model="google/gemma-3-27b-it"):
    # Check if file exists
    if not os.path.exists(pdf_path):
        return f"Error: File not found at path: {pdf_path}"
    
    # Get OpenRouter API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY environment variable not set"
    
    # Read and encode the PDF file
    try:
        with open(pdf_path, "rb") as file:
            pdf_bytes = file.read()
            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"
    
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",  # Optional: your site URL for attribution
    }
    
    # Prepare request data
    request_data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": os.path.basename(pdf_path),
                            "file_data": f"data:application/pdf;base64,{pdf_base64}"
                        }
                    }
                ]
            }
        ]
    }
    
    response = requests.post(
        url,
        headers=headers,
        json=request_data
    )
    
    if response.status_code == 200:
        response_data = response.json()
        if "error" in response_data:
            raise Exception(f"API error: {response_data['error']}")
        return response_data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")