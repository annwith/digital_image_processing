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
    convert_bit_plans_to_image, \
    encode_message_to_binary, \
    write_message_on_image_bit_plans


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
    parser = argparse.ArgumentParser(description='Encode a message on a image')

     # Add an argument for the image path
    parser.add_argument(
        '-i', 
        '--image_path', 
        type=str,
        help='Path to the image file'
    )

    # Add an argument for the message path
    parser.add_argument(
        '-m', 
        '--message_path', 
        type=str,
        help='Path to the message file'
    )

    # Add an argument for bit plans selection
    parser.add_argument(
        '-b', 
        '--bit_plans', 
        type=str,
        help='Which bit plans to use'
    )

    return parser.parse_args()


def encode_message_in_image():
    """
    Encode a message in an image.
    """

    # Parse command-line arguments
    args = configure_command_line_arguments()

    # Get image name
    image_name = (args.image_path).split("/")[-1]

    # Get which bit plans to use
    usable_bit_plans = usable_bit_plans_str_to_list(args.bit_plans)

    # Get image
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR) # cv2.IMREAD_UNCHANGED

    # Check if image was found
    if image is None:
        raise ValueError("Image not found")
    
    # Ensure image data type is uint8
    image_uint8 = cv2.convertScaleAbs(image)

    # Get height and width
    height, width, _ = image_uint8.shape
    
    # Get image available space in bytes based on the selected bit plans
    available_image_space = (height * width * len(usable_bit_plans) * 3) // 8
    
    # Get message
    with open(file=args.message_path, mode='r', encoding="utf-8") as file:
        # Read the entire content of the file
        message = file.read()

    # Encode message to bit plans
    encoded_message = encode_message_to_binary(
        message=message,
        available_image_space=available_image_space
    )

    # Get image bit plans
    image_bit_plans = convert_image_to_bit_plans(image=image_uint8)

    # Write message on image bit plans
    image_bit_plans_with_hiden_message = write_message_on_image_bit_plans(
        message=encoded_message,
        image_bit_plans=image_bit_plans,
        usable_bit_plans=usable_bit_plans
    )

    # Convert image bit plans to image
    image_with_hidden_message = convert_bit_plans_to_image(
        bit_plans=image_bit_plans_with_hiden_message
    )

    # Save image with hidden message
    cv2.imwrite("output_images/" + image_name, image_with_hidden_message)


if __name__ == "__main__":
    encode_message_in_image()
