import logging
import sys
import time
import threading
from functools import wraps
import pandas as pd
from datasets import load_dataset
from constants import *
import difflib

def get_masader_test():
    sheet_id = "1-07izL_VBZfdKT0fBllZHW8E1psOU-VM"
    sheet_name = "Sheet1"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    df = pd.read_csv(url, usecols=range(35))
    df.columns.values[0] = "No."
    df.columns.values[1] = "Name"
    return df

def get_masader_valid():
    sheet_id = "1awxq3QkWBQVRZnEVhx7ClKuw1JFM8k4gf-jh2GFPJwc"
    sheet_name = "Sheet1"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    df = pd.read_csv(url, usecols=range(35))
    df.columns.values[0] = "No."
    df.columns.values[1] = "Name"
    return df

def spinner_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Event to stop the spinner
        stop_event = threading.Event()
        
        # Spinner thread
        spinner_thread = threading.Thread(target=spinner_animation, args=(stop_event,))
        spinner_thread.start()
        
        # Run the wrapped function
        try:
            result = func(*args, **kwargs)
        finally:
            # Stop the spinner and wait for the thread to finish
            stop_event.set()
            spinner_thread.join()
            clear_line()  # Clear the spinner from the terminal
        
        return result
    
    return wrapper

def spinner_animation(stop_event):
    spinner = ['|', '/', '-', '\\']
    idx = 0

    while not stop_event.is_set():
        sys.stdout.write('\r' + spinner[idx])
        sys.stdout.flush()
        idx = (idx + 1) % len(spinner)
        time.sleep(0.1)

def clear_line():
    """Clear the current line in the terminal."""
    sys.stdout.write('\r\033[K')  # Move to the start of the line and clear it
    sys.stdout.flush()

def setup_logger() -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('results')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

def match_titles(title, masader_title):
    if isinstance(masader_title, float):
        return 0
    return difflib.SequenceMatcher(None, title, masader_title).ratio()

def validate(metadata):
    dataset = load_dataset('arbml/masader', trust_remote_code=True)
    results = {
        'CONTENT':0,
        'ACCESSABILITY':0,
        'DIVERSITY':0,
        'EVALUATION':0,
        'AVERAGE':0,
    }

    matched_row = None
    for row in dataset['train']:
        if match_titles(str(metadata['Paper Title']), row['Paper Title']) > 0.8:
            matched_row = row
    if not matched_row:
        return results
    
    for column in validation_columns:

        gold_answer = matched_row[column]
        if str(gold_answer) == 'nan':
            gold_answer = ''
        pred_answer = metadata[column]
        if column == 'Subsets':
            if len(pred_answer) != len(gold_answer):
                continue
            for subset in gold_answer:
                for key in subset:
                    if key not in pred_answer:
                        continue
                    if subset[key] != pred_answer[key]:
                        continue
            
            results['AVERAGE'] += 1/len(validation_columns) 
            results['DIVERSITY']+= 1/3
            continue

        if pred_answer.strip().lower() == gold_answer.strip().lower():
            results['AVERAGE'] += 1/len(validation_columns)
            if column in publication_columns:
                results['PUBLICATION'] += 1/6
            elif column in content_columns:
                results['CONTENT'] += 1/8
            elif column in accessability_columns:
                results['ACCESSABILITY']+= 1/5
            elif column in diversity_columns:
                results['DIVERSITY']+= 1/3
            elif column in evaluation_columns:
                results['EVALUATION'] += 1/3
        else:
            print(pred_answer, gold_answer)
    return results

from collections import Counter

def majority_vote(dicts):
    result = {}
    
    for key in columns:
        if key == 'Subsets':
            result[key] = []
            continue
        
        # only use smarter models as a judge       
        values = [dicts[model_name][key] for model_name in dicts if any([m in model_name for m in ['gemini-1.5-flash','pro','sonnet']])]
        
        # Count the occurrences of each value
        value_counts = Counter(values)
        # Find the value with the highest count (majority vote)
        majority_value, score = value_counts.most_common(1)[0]
        # if score > 3:
        #     result[key] = majority_value
        # else:
        #     for model_name in dicts:
        #         if 'pro' in model_name: #bias towards sonnet
        #             result[key] = dicts[model_name][key]
        result[key] = majority_value
    
    return result

def get_metadata_judge(dicts):
    all_metadata = {d['config']['model_name']:d['metadata'] for d in dicts}
    return '', majority_vote(all_metadata)

def get_metadata_human(paper_title):
    dataset = load_dataset('arbml/masader', trust_remote_code=True)
    for row in dataset['train']:
        if match_titles(str(paper_title), row['Paper Title']) > 0.8:
            return '', row

def compare_results(rs, show_diff = False):
    results = {}
    
    for c in columns:
        for r in rs:
            model_name = r['config']['model_name']
            value = r['metadata'][c]
            if c not in results:
                results[c] = {}        
            results[c][model_name] = value
        
        if show_diff:
            if all([results[c][m]==value for m in results[c]]):
                del results[c]
        
    df = pd.DataFrame(results)
    return df.transpose()

def find_best_match(text, options):
    """
    Find the option from the provided list that is most similar to the given text.
    
    Args:
        text (str): The text to be compared.
        options (list): A list of strings to compare the text against.
        
    Returns:
        str: The option from the list that is most similar to the text.
    """
    # Create a SequenceMatcher object to compare the text with each option
    matcher = difflib.SequenceMatcher(None, text.lower(), None)
    
    # Initialize variables to track the best match
    best_match = None
    best_ratio = 0
    
    # Iterate through the options and find the best match
    for option in options:
        matcher.set_seq2(option.lower())
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_match = option
            best_ratio = ratio
    
    return best_match