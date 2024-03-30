"""
Show the bit plans of an image.
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt

from utils import convert_image_to_bits_plans


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
    parser = argparse.ArgumentParser(description='Show the bit plans of an image')

     # Add an argument for the input image path
    parser.add_argument(
        '-i', 
        '--input_image_path', 
        type=str,
        help='Path to the input image file'
    )

    # Add an argument for the output image path
    parser.add_argument(
        '-o', 
        '--output_image_path', 
        type=str,
        help='Path to the output image file'
    )

    # Add an argument for the bits plans
    parser.add_argument(
        '-b', 
        '--bits_plans', 
        type=str,
        help='Which bit plans to show'
    )

    # Add an argument to save the bit plan images
    parser.add_argument(
        '--save',
        action='store_true',
        help='Flag to save the bits plan images'
    )

    return parser.parse_args()


def plot_bit_plan(input_image_bits_plans, output_image_bits_plans, bit_plan):
    """
    Plots the bit plans of the input and output images for each channel.

    Parameters:
        input_image_bits_plans (numpy.ndarray): The bit plans of the input image.
        output_image_bits_plans (numpy.ndarray): The bit plans of the output image.
        bit_plan (int): The bit plan to plot.
    
    """

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Bit Plan {bit_plan}')

    for i in range(3):
        axs[0, i].imshow(input_image_bits_plans[:,:,i,7 - bit_plan], cmap='gray')
        axs[0, i].set_title(f'Input Image Channel: {i}')
        axs[0, i].axis('off')

        axs[1, i].imshow(output_image_bits_plans[:,:,i,7 - bit_plan], cmap='gray')
        axs[1, i].set_title(f'Output Image Channel: {i}')
        axs[1, i].axis('off')

    plt.show()


def save_bit_plan(image_name, input_image_bits_plans, output_image_bits_plans, bit_plan, output_dir):
    """
    Saves the bit plans of the input and output images for each channel.

    Parameters:    args = configure_command_line_arguments()

        image_name (str): The name of the image.
        input_image_bits_plans (numpy.ndarray): The bit plans of the input image.
        output_image_bits_plans (numpy.ndarray): The bit plans of the output image.
        bit_plan (int): The bit plan to save.
        output_dir (str): The directory to save the bit plans.

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(3):
        input_file_path = os.path.join(
            output_dir,
            f'{image_name}_input_channel_{i}_bit_plan_{bit_plan}.png')
        plt.imsave(
            input_file_path,
            input_image_bits_plans[:,:,i,7 - bit_plan], cmap='gray')

        output_file_path = os.path.join(
            output_dir,
            f'{image_name}_output_channel_{i}_bit_plan_{bit_plan}.png')
        plt.imsave(
            output_file_path,
            output_image_bits_plans[:,:,i,7 - bit_plan], cmap='gray')


def main():
    # Parse command-line arguments
    args = configure_command_line_arguments()

    # Get input image
    input_image = cv2.imread(args.input_image_path, cv2.IMREAD_COLOR)

    # Get image name
    image_name = (args.input_image_path).split("/")[-1].split(".")[0]

    # Check if image was found
    if input_image is None:
        raise ValueError("Image not found")

    # Get output image
    output_image = cv2.imread(args.output_image_path, cv2.IMREAD_COLOR)

    # Check if image was found
    if output_image is None:
        raise ValueError("Image not found")
    
    # Get the bit plans of the input image
    input_image_bits_plans = convert_image_to_bits_plans(input_image)

    # Get the bit plans of the output image
    output_image_bits_plans = convert_image_to_bits_plans(output_image)

    # Plot the bit plans
    for bit_plan in list(args.bits_plans.replace(',', '')):
        bit_plan = int(bit_plan)
        plot_bit_plan(
            input_image_bits_plans,
            output_image_bits_plans,
            bit_plan
        )
        if args.save:
            save_bit_plan(
                image_name,
                input_image_bits_plans,
                output_image_bits_plans,
                bit_plan,
                'output_bits_plans'
            )


if __name__ == "__main__":
    main()