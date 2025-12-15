from argparse import ArgumentParser

parser = ArgumentParser(
        description="Process keywords, month, and year parameters"
    )

parser.add_argument(
    "--paper_link", type=str, required=False, default="", help="paper link"
)   

parser.add_argument(
    "--model_name",
    type=str,
    required=False,
    default="gemini-1.5-flash",
    help="Name of the models to use",
)

parser.add_argument(
    "--title", type=str, required=False, default="", help="paper title"
)

parser.add_argument(
    "--abstract", type=str, required=False, default="", help="paper abstract"
)
parser.add_argument(
    "-b", "--browse_web", action="store_true", help="whether to browse the web"
)

parser.add_argument(
    "-o",
    "--overwrite",
    action="store_true",
    help="overwrite the extracted metadata",
)

parser.add_argument("--split", type=str, default="test")

parser.add_argument("--schema_name", type=str, default="ar")

parser.add_argument(
    "--format",
    type=str,
    default="pdf_plumber",
    help="format to use",
)
parser.add_argument(
    "--few_shot",
    type=int,
    required=False,
    default=0,
    help="number of few shot examples to use",
)
parser.add_argument(
    "--results_path",
    type=str,
    default="results",
    help="path to save the results",
)
parser.add_argument(
    "--save_paper_text",
    action="store_true",
    help="save paper text",
)
parser.add_argument(
    "--backend",
    type=str,
    default="openrouter",
    help="backend to use",
)

parser.add_argument(
    "--repeat_on_error",
    action="store_true",
    help="repeat on error",
)

parser.add_argument(
    "--context",
    type=str,
    default="all",
    help="context size to use",
)

parser.add_argument(
    "--max_model_len",
    type=int,
    default=None,
    help="context size to use",
)

parser.add_argument(
    "--max_output_len",
    type=int,
    default=None,
    help="max output length",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=0,
    help="timeout for each prediction",
)
parser.add_argument(
    "--log",
    action="store_false",
    help="log the progress",
)
parser.add_argument(
    "--version",
    type=str,
    default="2.0",
    help="version to use",
)
# Parse arguments conditionally
import sys

def get_default_args():
    """Return default arguments as a dictionary"""
    return {
        'paper_link': '',
        'model_name': 'gemini-1.5-flash',
        'title': '',
        'abstract': '',
        'browse_web': False,
        'overwrite': False,
        'split': 'test',
        'schema_name': 'ar',
        'format': 'pdf_plumber',
        'few_shot': 0,
        'results_path': 'results',
        'save_paper_text': False,
        'backend': 'openrouter',
        'repeat_on_error': False,
        'context': 'all',
        'max_model_len': None,
        'max_output_len': None,
        'timeout': 0,
        'log': False,
        'version': '2.0'
    }

# Create a simple class to mimic argparse.Namespace behavior
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Check if we're in a FastAPI/web context (no command line args)
try:
    # Try to parse args, but handle the case where there are no CLI args
    if len(sys.argv) == 1:  # Only script name, no arguments
        args = Args(**get_default_args())
    else:
        args = parser.parse_args()
except SystemExit:
    # argparse calls sys.exit() when there are issues, catch this for web contexts
    args = Args(**get_default_args())
    