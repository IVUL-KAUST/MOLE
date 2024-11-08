import logging

def _setup_logger() -> logging.Logger:
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