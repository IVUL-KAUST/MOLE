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
# Parse arguments
args = parser.parse_args()
    