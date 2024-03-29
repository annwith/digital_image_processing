"""
Generates a message with a given number of bytes.
"""

import argparse
import numpy as np


def configure_command_line_arguments():
    """
    Configure and retrieve command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command-line 
        arguments.

    This function sets up an ArgumentParser object to parse command-line 
    arguments.
    It adds arguments for the image path, message path, and bit plans selection.
    The function then parses the command-line arguments and returns the parsed
    arguments.
    """
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Generates a message')

     # Add an argument for the image path
    parser.add_argument(
        '-b', 
        '--bytes', 
        type=str, 
        help='Bytes number of the message'
    )

    return parser.parse_args()


def generate_message():
    """
    Generates a message with a given number of bytes.
    """

    # Parse command-line arguments
    args = configure_command_line_arguments()

    # Generate random ASCII characters
    random_chars = np.random.randint(
        256,
        size=int(args.bytes)
    )

    # Convert ASCII codes to characters
    random_text = ''.join(map(chr, random_chars))

    # Save the random text to a file
    with open("input_messages/"+args.bytes+'_message.txt', 'w', encoding="utf-8") as file:
        file.write(random_text)
    file.close()


if __name__ == "__main__":
    generate_message()
