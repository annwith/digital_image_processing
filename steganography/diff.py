"""
Compare two files.
"""

import argparse


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
    parser = argparse.ArgumentParser(description='Compare two files')

     # Add an argument for the image path
    parser.add_argument(
        '-i', 
        '--input_message', 
        type=str,
        help='Path to the input message file',
        required=True
    )
    
    # Add an argument for bit plans selection
    parser.add_argument(
        '-o', 
        '--output_message', 
        type=str,
        help='Path to the output message file',
        required=True
    )

    return parser.parse_args()


def compare():
    """
    Compare two files.
    """

    # Parse command-line arguments
    args = configure_command_line_arguments()

    # Get input message
    with open(args.input_message, 'r', encoding="utf-8") as input_message_file:
        input_message = input_message_file.read()
    input_message_file.close()

    # Get output message
    with open(args.output_message, 'r', encoding="utf-8") as output_message_file:
        output_message = output_message_file.read()
    output_message_file.close()
    
    # Compare the two messages
    if input_message == output_message:
        print("The messages are equal.")
    else:
        print("The messages are different.")


if __name__ == "__main__":
    compare()
