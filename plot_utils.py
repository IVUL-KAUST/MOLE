from tabulate import tabulate  # type: ignore
import re

def remove_average(results, headers):
    # if both Weighted Average and Average are in the headers, remove the Average
    output_results =[]
    if "Weighted Average" in headers and "Average" in headers:
        headers.remove("Average")
        for row in results:
            row = row[:-2]+[row[-1]]
            output_results.append(row)
        headers = headers[:-1] + ["Average"]
    else:
        output_results = results
    return output_results, headers

def print_table(results, headers, title="", format=False):
    results, headers = remove_average(results, headers)
    RED = "\033[103m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    if not format:
        print(
            tabulate(
                sorted(
                    results,
                    key=lambda x: x[-1],
                    reverse=False,
                ),
                headers=headers,
                tablefmt="github",
                floatfmt=".2f",
            )
        )
        return
    # Format the numbers - highlight max in red and underline second max for each column
    formatted_results = []
    numeric_columns = list(zip(*[row[1:] for row in results]))  # Exclude model names
    for row in results:
        formatted_row = [row[0]]  # Start with model name
        for i, value in enumerate(row[1:]):
            column_values = numeric_columns[i]
            max_val = max(column_values)
            if len(sorted(column_values)) >= 2:
                second_max = sorted(column_values)[-2]
            else:
                second_max = max_val
            if abs(value - max_val) < 1e-10:  # Using small epsilon for float comparison
                formatted_row.append(f"{RED}{value}{END}")
            elif abs(value - second_max) < 1e-10:
                formatted_row.append(f"{UNDERLINE}{value}{END}")
            else:
                formatted_row.append(f"{value}")
        formatted_results.append(formatted_row)
    # Show table title if provided
    if title:
        print(f"\n{title}\n")

    print(
        tabulate(
            sorted(
                formatted_results,
                key=lambda x: float(
                    x[-1].replace(RED, "").replace(UNDERLINE, "").replace(END, "")
                ),
                reverse=False,
            ),
            headers=headers,
            tablefmt="github",
            floatfmt=".2f"
        )
    )

def print_latex_table(results, headers, title="", caption="", label=""):
    """Generate LaTeX table format for copying to Overleaf"""
    results, headers = remove_average(results, headers)
    
    # Sort results by the last column (typically average score)
    sorted_results = sorted(results, key=lambda x: x[-1], reverse=False)
    
    # Clean headers for LaTeX (replace spaces with proper formatting)
    latex_headers = []
    for header in headers:
        # Replace common characters that need escaping in LaTeX
        clean_header = header.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
        latex_headers.append(clean_header)
    
    # Start building the LaTeX table
    num_cols = len(headers)
    col_spec = "l" + "c" * (num_cols - 1)  # Left align first column, center others
    
    latex_output = []
    latex_output.append("\\begin{table}[htbp]")
    latex_output.append("\\centering")
    if caption:
        latex_output.append(f"\\caption{{{caption}}}")
    if label:
        latex_output.append(f"\\label{{{label}}}")
    latex_output.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_output.append("\\toprule")
    
    # Add headers (make them bold)
    bold_headers = [f"\\textbf{{{header}}}" for header in latex_headers]
    header_row = " & ".join(bold_headers) + " \\\\"
    latex_output.append(header_row)
    latex_output.append("\\midrule")
    
    # Find max and second max for each numeric column for formatting
    numeric_columns = list(zip(*[row[1:] for row in sorted_results]))  # Exclude model names
    
    # Add data rows with formatting for max/second max values
    for row in sorted_results:
        latex_row = []
        for i, cell in enumerate(row):
            if i == 0:  # Model name - escape special characters
                clean_cell = str(cell).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
                latex_row.append(clean_cell)
            else:  # Numeric values
                if isinstance(cell, (int, float)):
                    # Get the column values for comparison
                    column_values = numeric_columns[i-1]  # i-1 because we excluded model names
                    max_val = max(column_values)
                    if len(sorted(column_values)) >= 2:
                        second_max = sorted(column_values)[-2]
                    else:
                        second_max = max_val
                    
                    # Format the cell value
                    formatted_value = f"{cell:.2f}"
                    
                    # Apply formatting based on ranking
                    if abs(cell - max_val) < 1e-10:  # Using small epsilon for float comparison
                        latex_row.append(f"\\textbf{{{formatted_value}}}")
                    elif abs(cell - second_max) < 1e-10:
                        latex_row.append(f"\\underline{{{formatted_value}}}")
                    else:
                        latex_row.append(formatted_value)
                else:
                    latex_row.append(str(cell))
        
        row_str = " & ".join(latex_row) + " \\\\"
        latex_output.append(row_str)
    
    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    
    # Print the LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE (Copy to Overleaf):")
    print("="*60)
    for line in latex_output:
        print(line)
    print("="*60)
    print("Note: Make sure to include \\usepackage{booktabs} in your LaTeX preamble for \\toprule, \\midrule, and \\bottomrule commands.")
    print("="*60 + "\n")
