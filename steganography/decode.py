"""
Trabalho 1: Esteganografia
Nome: Juliana Midlej do Espírito Santo
RA: 200208
"""

import argparse
import time
import cv2

from utils import \
    usable_bits_plans_str_to_list, \
    convert_image_to_bits_plans, \
    read_message_from_image_bits_plans, \
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
        '--bits_plans', 
        type=str, 
        help='Which bit plans to use'
    )

    # Add an argument to force the message decode and use all bits plans.
    parser.add_argument(
        '--force',
        action='store_true',
        help='Flag to force the message decode to use all bits plans.'
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
    if args.force:
        usable_bits_plans = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        usable_bits_plans = usable_bits_plans_str_to_list(args.bits_plans)

    # Get image
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)

    # Convert image data type to uint8
    image_uint8 = cv2.convertScaleAbs(image)

    start_time = time.time() # Set start time

    # Get image bit plans
    image_bits_plans = convert_image_to_bits_plans(image=image_uint8)

    end_time = time.time() # Set end time
    print(f"Convert image to bits plans: {end_time - start_time}") # Print execution time

    start_time = time.time() # Set start time

    # Get message from the image bit plans
    binary_message = read_message_from_image_bits_plans(
        image_bits_plans=image_bits_plans,
        usable_bits_plans=usable_bits_plans
    )

    end_time = time.time() # Set end time
    print(f"Read message: {end_time - start_time}") # Print execution time

    start_time = time.time() # Set start time

    # Convert binary message to string
    message = decode_message_from_binary(binary_message=binary_message)

    end_time = time.time() # Set end time
    print(f"Decode: {end_time - start_time}") # Print execution time

    # Save message on a file
    with open("output_messages/" + image_name + ".txt", 'w', encoding="utf-8") as file:
        file.write(message)
    file.close()


if __name__ == "__main__":
    decode_message_in_image()
