"""
This module contains utility functions for the steganography application.
"""

import re
from typing import List
import numpy as np


def int_to_binary_deprecated(value: int) -> np.ndarray:
    """
    Convert an integer value to a binary NumPy array.

    Parameters:
        value (int): The integer value to convert to binary.

    Returns:
        np.ndarray: A NumPy array representing the binary representation of the input integer.
    """
    bin_str_list = list(bin(value)[2:].zfill(8))
    bin_array = np.array(bin_str_list, dtype=int)

    return bin_array


def int_to_binary(value: int) -> np.ndarray:
    """
    Convert an integer value to a binary NumPy array.

    Parameters:
        value (int): The integer value to convert to binary.

    Returns:
        np.array: A NumPy array representing the binary representation of the input integer.
    """
    # Get the binary representation of the integer value as a string
    binary_representation = np.binary_repr(value, width=8)

    # Convert the binary string to a NumPy array of integers
    binary_array = np.array([int(bit) for bit in binary_representation], dtype=np.uint8)

    return binary_array


def char_to_binary(value: str) -> np.ndarray:
    """
    Convert a character to a binary NumPy array.

    Parameters:
        value (str): The character to convert to binary.
    Returns:
        np.array: A NumPy array representing the binary representation of the input character.
    """
    # Get the ASCII value of the character
    ascii_value = ord(value)

    # Convert the ASCII value to binary and concatenate it to the binary representation string
    binary_representation = bin(ascii_value)[2:].zfill(8)

    # Convert the binary string to a NumPy array of integers
    binary_array = np.array([int(bit) for bit in binary_representation], dtype=np.uint8)

    return binary_array


def binary_to_int(value: np.ndarray) -> int:
    """
    Convert a binary NumPy array to an integer value.

    Parameters:
        value (np.ndarray): The binary NumPy array to convert to an integer.

    Returns:
        int: The integer value represented by the input binary array.
    """
    # Convert the binary array to a binary string
    binary_str = ''.join([str(bit) for bit in value])

    # Convert the binary string to an integer
    int_value = int(binary_str, 2)

    return int_value


def convert_image_to_bit_plans(image: np.ndarray) -> np.ndarray:
    """
    Extracts the bit plans of each color channel from an image.

    Args:
        image (np.array): A NumPy array representing the image with shape (height, width, channels).

    Returns:
        np.array: A NumPy array containing the bit plans of each color channel for each pixel.
                  The resulting array has shape (height, width, channels, 8), where 8 represents
                  the 8-bit representation of each color channel.

    The function takes an image represented as a NumPy array and extracts the bit plans of each
    color channel (blue, green, and red) for each pixel in the image. It returns a NumPy array
    containing the bit plans for each channel.

    The bit plans are stored in the returned array with the following dimensions:
    - The first two dimensions correspond to the height and width of the image.
    - The third dimension represents the color channels (blue, green, and red).
    - The fourth dimension represents the 8-bit representation of each color channel, with each
      bit plane stored in a separate element of this dimension.
    """
    # Get the height and width of the image
    height, width, channels = image.shape

    # get_bit_plans_vectorized = np.vectorize(int_to_binary)
    # bit_plans_vectorized = get_bit_plans_vectorized(image)
    # print("Bit plans vectorized:", bit_plans_vectorized)

    # Bit plans array to store the bit plans
    bit_plans = np.zeros((height, width, channels, 8), dtype=np.uint8)

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Get the BGR color values of the pixel
            blue, green, red = image[y, x]

            # Extract the bit plans for each color channel and store them in the array
            bit_plans[y, x, 0] = int_to_binary(blue)
            bit_plans[y, x, 1] = int_to_binary(green)
            bit_plans[y, x, 2] = int_to_binary(red)

    return bit_plans


def convert_bit_plans_to_image(bit_plans: np.ndarray) -> np.ndarray:
    """
    Converts the bit plans of an image to a NumPy array representing the image.

    Args:
        bit_plans (np.array): A NumPy array containing the bit plans of an image.

    Returns:
        np.array: A NumPy array representing the image with shape (height, width, channels).

    The function takes a NumPy array containing the bit plans of an image and converts it back
    to a NumPy array representing the image. The input array should have the following dimensions:
    - The first two dimensions correspond to the height and width of the image.
    - The third dimension represents the color channels (blue, green, and red).
    - The fourth dimension represents the 8-bit representation of each color channel, with each
      bit plane stored in a separate element of this dimension.
    """
    # Get the height, width, and channels of the bit plans array
    height, width, channels, _ = bit_plans.shape

    # Image array to store the image
    image = np.zeros((height, width, channels), dtype=np.uint8)

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # print("Pixel:", y, x)

            # Get the BGR color values of the pixel
            blue = binary_to_int(bit_plans[y, x, 0])
            green = binary_to_int(bit_plans[y, x, 1])
            red = binary_to_int(bit_plans[y, x, 2])

            # Store the color values in the image array
            image[y, x] = [blue, green, red]

    return image


def write_message_on_image_bit_plans(
    message: np.ndarray,
    image_bit_plans: np.ndarray,
    usable_bit_plans: List[int]
) -> np.ndarray:
    """
    Write a message on the bit plans of an image.

    Parameters:
        message (np.ndarray): A NumPy array containing the binary representation of the message.
        image_bit_plans (np.ndarray): A NumPy array containing the bit plans of the image.
        usable_bit_plans (List[int]): A list of integers representing the bit plans to use.

    Returns:
        np.ndarray: A NumPy array containing the image with the hidden message.        
    """

    height, width, _, _ = image_bit_plans.shape
    
    # Write the message on the image bit plans
    for bit_plan in usable_bit_plans:
        # Check if the message is bigger than the bit plan
        if len(message) >= (height * width):
            # Write the message on the bit plan
            image_bit_plans[:, :, 0, bit_plan] = message[:height*width].reshape(height, width)
        # If not, write the message on the bit plan and breaks the loop
        else:
            # Flatten the bit plan
            image_bit_plan = np.reshape(image_bit_plans[:, :, 0, bit_plan], -1)
            # Concatenate the message with the remaining bit plan
            message = np.concatenate([message, image_bit_plan[len(message):]], axis=0)
            # Write the message on the bit plan
            image_bit_plans[:, :, 0, bit_plan] = message.reshape(height, width)

            break
        # Keep track of the remaining message
        message = message[height*width:]

    return image_bit_plans


def encode_message_to_binary(
    message: str,
    available_image_space: int
) -> np.ndarray:
    """
    Encode a message into a sequence of bit plans.

    Parameters:
        message (str): The message to encode.
        available_image_space (int): The available space in bytes in the image.

    Returns:
        np.ndarray: A NumPy array containing the binary representation of the message.
    """
    # Include the message size in the message
    message = str(len(message)) + " " + message
    # print("Mensagem:", message)

    # Get the size of the message in bytes
    message_size = len(message)
    # print("Tamanho da mensagem:", message_size)

    # Convert each char from the message to binary
    binary_message = [char_to_binary(char) for char in message]
    # print("Mensagem binária:", binary_message)

    # Concatenate the binary np.array of each character in the message
    binary_array_message = np.concatenate(binary_message)
    # print("Mensagem binária concatenada:", binary_array_message)

    # Check if the message conversion to binary is size consistent
    if not binary_array_message.size == message_size * 8:
        raise ValueError("Error: The binary conversion of the message is not size consistent.")
    
    # Check if the message fits in the available image space
    if message_size > available_image_space:
        raise ValueError(f"Error: The message has {message_size} bytes and is too large to fit in the image.")
    
    return binary_array_message


def usable_bit_plans_str_to_list(
    bit_plans_str: str
) -> List[int]:
    """
    Convert a string representing the bit plans to a list of integers. The string should only contain the numbers 0, 1, or/and 2 separated by commas.

    Parameters:
        bit_plans_str (str): A string representing the bit plans to use.

    Returns:
        List[int]: A list of integers representing the bit plans to use.

    """
    # Define the pattern for the bit plans string
    pattern = r'^[012]+(,[012]+)*$'
    
    # Use re.match to check if the input string matches the pattern
    if not re.match(pattern, bit_plans_str):
        raise ValueError("Invalid bit plans string. Please use only the numbers 0, 1, or 2 separated by commas.")

    # Split the string by commas to get the individual bit plans
    bit_plans_list = bit_plans_str.split(',')

    # Convert the bit plans to integers
    bit_plans = [int(plan) for plan in bit_plans_list]

    # Remove repeated bit plans
    bit_plans = list(set(bit_plans))

    return bit_plans