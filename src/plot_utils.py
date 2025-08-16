from tabulate import tabulate  # type: ignore
import numpy as np
def remove_average(results, headers):
    # if both Weighted Average and Average are in the headers, remove the Average
    if "Weighted Average" in headers and "Average" in headers:
        new_results = []
        for row in results:
            new_results.append(row[:-2]+[row[-1]])
        headers = headers[:-2] + ["Average"]
        return new_results, headers
    else:
        return results, headers

def get_max_per_row(results):
    max_per_row = []
    row_len = len(results[0])
    for i in range(row_len):
        results_per_column = []
        for j in range(len(results)):
            value = results[j][i]
            try:
                value = float(value)
                value = f"{value:.2f}"
            except:
                value = str(value)
            results_per_column.append(value)
        max_per_row.append(max([len(r) for r in results_per_column]))
    return max_per_row
def print_table(results, headers, title="", format=False):
    RED = "\033[105m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    if not format:
        # add horizontal line before the last row
        print(
            tabulate(
                results,
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
            try:
                max_val = max([val for val in column_values if val > 0])
            except:
                max_val = 0
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

    rows = sorted(
                formatted_results,
                key=lambda x: float(
                    x[-1].replace(RED, "").replace(UNDERLINE, "").replace(END, "")
                ),
                reverse=False,
            )

    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            floatfmt=".2f"
        )
    )
