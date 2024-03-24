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
        np.ndarray: A NumPy array representing the binary representation of the
                    input integer.
    """
    bin_str_list = list(bin(value)[2:].zfill(8))
    bin_array = np.array(bin_str_list, dtype=int)

    return bin_array


def int_to_binary(value: int) -> np.ndarray:
    """
    Convert an integer value to a 1D NumPy array containing the integer binary 
    representation.

    Parameters:
        value (int): The integer value to convert to binary.

    Returns:
        np.ndarray: A 1D unsigned integers NumPy array containing the binary 
                    representation of the input integer. The size of the NumPy 
                    array is 8.
    """
    # Get the binary representation of the integer value as a string
    binary_representation = np.binary_repr(value, width=8)

    # Convert the binary string to a NumPy array of integers
    binary_array = np.array(
        [int(bit) for bit in binary_representation],
        dtype=np.uint8)

    return binary_array


def binary_to_int(value: np.ndarray) -> int:
    """
    Convert a 1D unsigned intergers NumPy array with binary values to an 
    integer.

    Parameters:
        value (np.ndarray): 1D NumPy array with size 8 containing binary 
                            values to convert to an integer.

    Returns:
        int: The integer represented by the input NumPy array.
    """
    # Convert the binary array to a binary string
    binary_str = ''.join([str(bit) for bit in value])

    # Convert the binary string to an integer
    int_value = int(binary_str, 2)

    return int_value


def char_to_binary(value: str) -> np.ndarray:
    """
    Convert a character to binary following the ASCII enconding and stores in a 
    1D NumPy array of unsigned integers.

    Parameters:
        value (str): The character to convert to binary.

    Returns:
        np.ndarray: A 1D NumPy array representing the binary representation of 
                    the input character. The size of the NumPy array is 8.
    """
    # Get the Unicode value of the character
    ascii_value = ord(value)

    # Check if the Unicode value is within the valid ASCII range
    if ascii_value > 255:
        raise ValueError(
            f"Error: The '{value}' character is not in the ASCII table.")

    # Convert the ASCII value to binary representation
    binary_representation = bin(ascii_value)[2:].zfill(8)

    # Convert the binary string to a NumPy array of integers
    binary_array = np.array(
        [int(bit) for bit in binary_representation],
        dtype=np.uint8)

    return binary_array


def binary_to_char(value: np.ndarray) -> str:
    """
    Convert a 1D unsigned intergers NumPy array with binary values to a 
    character.

    Parameters:
        value (np.ndarray): 1D NumPy array with size 8 containing binary 
                            values to convert to a character.

    Returns:
        str: The character represented by the input NumPy array.
    """
    # Convert the binary array to a binary string
    binary_str = ''.join([str(int(bit)) for bit in value])

    # Convert the binary string to an integer
    int_value = int(binary_str, 2)

    # Convert the integer value to a character
    char_value = chr(int_value)

    return char_value


def convert_image_to_bit_plans(image: np.ndarray) -> np.ndarray:
    """
    Extracts the bit plans of each color channel from an image.

    Args:
        image (np.ndarray): A NumPy array representing the image with shape
                          (height, width, channels).

    Returns:
        np.ndarray: A NumPy array containing the bit plans of each color channel
                  for each pixel. The resulting array has shape (height, width,
                  channels, 8), where 8 represents the 8-bit representation of 
                  each color channel.

    The function takes an image represented as a NumPy array and extracts the 
    bit plans of each color channel (blue, green, and red). It returns a NumPy 
    array containing the bit plans for each channel.

    Bit plans are stored in the returned array with the following dimensions:
    - The first two dimensions correspond to the height and width of the image.
    - The third dimension represents the color channels (blue, green, and red).
    - The fourth dimension is the 8-bit representation of the value of each 
      color channel for each pixel.
    """
    # Get the height, width and channels of the image
    height, width, channels = image.shape

    # get_bit_plans_vectorized = np.vectorize(int_to_binary)
    # bit_plans_vectorized = get_bit_plans_vectorized(image)
    # print("Bit plans vectorized:", bit_plans_vectorized)

    #  Initialize the bit plans array to store the image bit plans
    bit_plans = np.zeros((height, width, channels, 8), dtype=np.uint8)

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Get the BGR color values of the pixel
            blue, green, red = image[y, x]

            # Stores the 8-bit representation of each color channel
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
        np.array: A NumPy array representing the image with shape (height, width, 
                  channels).

    The function takes a NumPy array containing the bit plans of an image and 
    converts it back to a NumPy array representing the image. The input array 
    should have the following dimensions:
    - The first two dimensions correspond to the height and width of the image.
    - The third dimension represents the color channels (blue, green, and red).
    - The fourth dimension is the 8-bit representation of the value of each 
      color channel for each pixel.
    """
    # Get the height, width, and channels of the bit plans array
    height, width, channels, _ = bit_plans.shape

    # Image array to store the image
    image = np.zeros((height, width, channels), dtype=np.uint8)

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
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
        message (np.ndarray): A 1D NumPy array containing the binary 
                              representation of the message.
        image_bit_plans (np.ndarray): A NumPy array containing the bit plans of 
                                      the image.
        usable_bit_plans (List[int]): A list of integers representing the bit 
                                      plans to use.

    Returns:
        np.ndarray: A NumPy array containing the image with the hidden message.        
    """

    height, width, _, _ = image_bit_plans.shape

    # Write the message on the image bit plans
    for bit_plan in usable_bit_plans:
        # Check if the message is bigger than the bit plan
        if len(message) >= (height * width):
            # Write the message on the bit plan
            image_bit_plans[:, :, 0, bit_plan] = message[:height*width].reshape(
                height, width)
        # If not, write the message on the bit plan and breaks the loop
        else:
            # Flatten the bit plan
            image_bit_plan = np.reshape(image_bit_plans[:, :, 0, bit_plan], -1)
            # Concatenate the message with the remaining bit plan
            message = np.concatenate(
                [message, image_bit_plan[len(message):]], axis=0)
            # Write the message on the bit plan
            image_bit_plans[:, :, 0, bit_plan] = message.reshape(height, width)

            break
        # Keep track of the remaining message
        message = message[height*width:]

    return image_bit_plans


def read_message_from_image_bit_plans(
    image_bit_plans: np.ndarray,
    usable_bit_plans: List[int]
) -> np.ndarray:
    """
    Read a message from the bit plans of an image.

    Parameters:
        image_bit_plans (np.ndarray): A NumPy array containing the bit plans of 
                                      the image.
        usable_bit_plans (List[int]): A list of integers representing the bit 
                                      plans to use.

    Returns:
        np.ndarray: A NumPy array containing the binary representation of the 
                    message.
    """

    # Initialize the binary message
    binary_message = np.array([])

    # Read the message on the image bit plans
    for bit_plan in usable_bit_plans:
        binary_message = np.concatenate(
            [binary_message, image_bit_plans[:, :, 0, bit_plan].reshape(-1)], axis=0)

    # Ensure the binary message has a size multiple of 8
    binary_message = binary_message[:(binary_message.size // 8) * 8]

    return binary_message


def encode_message_to_binary(
    message: str,
    available_image_space: int
) -> np.ndarray:
    """
    Receives a message, convert it to binary following ASCII enconding and 
    stores in a 1D NumPy array.

    Parameters:
        message (str): The message to encode.
        available_image_space (int): The available space in bytes in the image.

    Returns:
        np.ndarray: A 1D NumPy array containing the binary representation of the 
                    message.
    """
    # Include the message size in the message
    message = str(len(message)) + " " + message

    # Get the size of the message in bytes
    message_size = len(message)

    # Convert each char from the message to binary
    binary_message = [char_to_binary(char) for char in message]

    # Concatenate the binary np.array of each character in the message
    binary_array_message = np.concatenate(binary_message)

    # Check if the message conversion to binary is size consistent
    if not binary_array_message.size == message_size * 8:
        raise ValueError(
            "Error: The binary conversion of the message is not size consistent.")

    # Check if the message fits in the available image space
    if message_size > available_image_space:
        raise ValueError(
            f"Error: The message has {message_size} bytes and "
            f"is too large to fit in the image.")
    
    return binary_array_message


def decode_message_from_binary(
    binary_message: np.ndarray
) -> str:
    """
    Decode a message from a binary array.

    Parameters:
        message (np.ndarray): A NumPy array containing the binary representation 
                              of the message.

    Returns:
        str: The decoded message.
    """

    # Reshape the binary message to a 2D array with 8 columns
    binary_message = np.reshape(binary_message, (binary_message.size // 8, 8))

    # Convert the binary array to a list of chars
    binary_message = [binary_to_char(byte) for byte in binary_message]

    # Join the list of chars to form the message
    decoded_message = "".join(binary_message)

    # Get the size of the message
    decoded_message_size = decoded_message.split(" ", maxsplit=1)[0]

    # Get the message
    decoded_message = decoded_message[
        len(decoded_message_size) + 1:
        int(decoded_message_size) + len(decoded_message_size) + 1]

    return decoded_message


def usable_bit_plans_str_to_list(
    bit_plans_str: str
) -> List[int]:
    """
    Convert a string representing the bit plans to a list of integers. The 
    string should only contain the numbers 0, 1, or/and 2 separated by commas.

    Parameters:
        bit_plans_str (str): A string representing the bit plans to use.

    Returns:
        List[int]: A list of integers representing the bit plans to use.

    """
    # Define the pattern for the bit plans string
    pattern = r'^[012]+(,[012]+)*$'

    # Use re.match to check if the input string matches the pattern
    if not re.match(pattern, bit_plans_str):
        raise ValueError(
            "Invalid bit plans string. Please use only the numbers 0, 1, or 2 separated by commas.")

    # Split the string by commas to get the individual bit plans
    bit_plans_list = bit_plans_str.split(',')

    # Convert the bit plans to integers
    bit_plans = [int(plan) for plan in bit_plans_list]

    # Remove repeated bit plans
    bit_plans = list(set(bit_plans))

    return bit_plans
