"""
Trabalho 1: Esteganografia
Nome: Juliana Midlej do Espírito Santo
RA: 200208
"""

import argparse
import cv2

from utils import \
    usable_bit_plans_str_to_list, \
    convert_image_to_bit_plans, \
    read_message_from_image_bit_plans, \
    decode_message_from_binary


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
    parser = argparse.ArgumentParser(description='Decode a message from a image')

     # Add an argument for the image path
    parser.add_argument(
        '-i', 
        '--image_path', 
        type=str, 
        help='Path to the image file'
    )
    
    # Add an argument for bit plans selection
    parser.add_argument(
        '-b', 
        '--bit_plans', 
        type=str, 
        help='Which bit plans to use'
    )

    return parser.parse_args()


def decode_message_in_image():
    """
    Decode a message in an image.
    """

    # Parse command-line arguments
    args = configure_command_line_arguments()

    # Get image name
    image_name = (args.image_path).split("/")[-1].split(".")[0]

    # Get which bit plans to use
    usable_bit_plans = usable_bit_plans_str_to_list(args.bit_plans)

    # Get image
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)

    # Convert image data type to uint8
    image_uint8 = cv2.convertScaleAbs(image)

    # Get image bit plans
    image_bit_plans = convert_image_to_bit_plans(image=image_uint8)

    # Get message from the image bit plans
    binary_message = read_message_from_image_bit_plans(
        image_bit_plans=image_bit_plans,
        usable_bit_plans=usable_bit_plans
    )

    # Convert binary message to string
    message = decode_message_from_binary(binary_message=binary_message)

    # Save message on a file
    with open("output_messages/" + image_name + ".txt", 'w', encoding="utf-8") as file:
        file.write(message)


if __name__ == "__main__":
    decode_message_in_image()
