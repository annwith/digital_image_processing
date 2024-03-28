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
    convert_bits_plans_to_image, \
    encode_message_to_binary, \
    write_message_on_image_bits_plans


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
        '--bits_plans', 
        type=str,
        help='Which bits plans to use'
    )

    # Add an argument to force the message encode and use all bits plans needed,
    # starting from the least significant. If yet, the message does not fit in
    # the image, the program will raise an error.
    parser.add_argument(
        '--force',
        action='store_true',
        help='Flag to force the message encode even if the message is too big'
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
    if args.force:
        usable_bits_plans = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        usable_bits_plans = usable_bits_plans_str_to_list(args.bits_plans)

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
    available_image_space = (height * width * len(usable_bits_plans) * 3) // 8
    print(f"Avaliable image space: {available_image_space} bytes")

    # Get message and read it
    with open(file=args.message_path, mode='r', encoding="utf-8") as file:
        message = file.read()
    file.close()

    start_time = time.time() # Set start time
    
    # Encode message to bit plans
    encoded_message = encode_message_to_binary(
        message=message,
        available_image_space=available_image_space
    )
    
    end_time = time.time() # Set end time
    print(f"Encode: {end_time - start_time}") # Print execution time

    start_time = time.time() # Set start time
    
    # Get image bit plans
    image_bits_plans = convert_image_to_bits_plans(image=image_uint8)
    
    end_time = time.time() # Set end time
    print(f"Convert to bits plans: {end_time - start_time}") # Print execution time

    start_time = time.time() # Set start time
    
    # Write message on image bit plans
    image_bits_plans_with_hiden_message = write_message_on_image_bits_plans(
        message=encoded_message,
        image_bits_plans=image_bits_plans,
        usable_bits_plans=usable_bits_plans
    )
    
    end_time = time.time() # Set end time
    print(f"Write message: {end_time - start_time}") # Print execution time

    start_time = time.time() # Set start time
    
    # Convert image bit plans to image
    image_with_hidden_message = convert_bits_plans_to_image(
        bits_plans=image_bits_plans_with_hiden_message
    )
    
    end_time = time.time() # Set end time
    print(f"Convert to image: {end_time - start_time}") # Print execution time

    # Save image with hidden message
    cv2.imwrite("output_images/" + image_name, image_with_hidden_message)


if __name__ == "__main__":
    encode_message_in_image()
