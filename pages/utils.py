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
