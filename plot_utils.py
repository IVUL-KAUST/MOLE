from tabulate import tabulate  # type: ignore


def print_table(results, headers, title="", format=False):
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
