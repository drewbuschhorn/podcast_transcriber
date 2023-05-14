import os
import re

def sanitize_file_name(file_name):
    file_name = os.path.basename(file_name)
    # Remove non-alphanumeric characters (except dot) using regular expressions
    sanitized_name = re.sub(r'[^a-zA-Z0-9]', '_', file_name)
    # Remove leading and trailing underscores
    sanitized_name = sanitized_name.strip('_')
    return sanitized_name

def confirm_action(msg):
    while True:
        user_input = input(msg)
        if user_input.lower() == "y":
            return True
        elif user_input.lower() == "n":
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def open_file_wdirs(path, mode="wb", newline=None, encoding=None):
    # Ensure the parent directory exists
    dirname = os.path.dirname(path)
    if (dirname):
        os.makedirs(dirname, exist_ok=True)

    # Open file for binary writing
    file = open(path, mode, newline=newline, encoding=encoding)

    return file