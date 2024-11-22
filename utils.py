import logging
import sys
import time
import threading
from functools import wraps

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


import difflib

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