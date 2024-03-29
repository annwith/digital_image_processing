import unittest
import time

import cv2
import numpy as np

from utils import \
    int_to_binary_deprecated, \
    int_to_binary, \
    convert_image_to_bits_plans, \
    convert_image_to_bits_plans_vectorized


class TestIntToBinaryConversion(unittest.TestCase):
    """
    Class to compare deprecated and new version of the int_to_binary function.
    """

    def test_int_to_binary_conversion(self):
        """
        Test the int_to_binary function.
        """

        # Create an array of random integers
        integers = np.random.randint(0, 256, 100000)

        start_time = time.time()

        # Create vectorized function to convert integer to binary
        get_bits_plans_vectorized = np.vectorize(
            int_to_binary_deprecated, signature='()->(n)')

        # Apply the vectorized function to the integers
        deprecated_version = get_bits_plans_vectorized(integers)

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"(Int to Binary Deprecated) Execution time: {execution_time}")

        start_time = time.time()

        # Create vectorized function to convert integer to binary
        get_bits_plans_vectorized = np.vectorize(
            int_to_binary, signature='()->(n)')

        # Apply the vectorized function to the integers)
        version = get_bits_plans_vectorized(integers)

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"(Int to Binary) Execution time: {execution_time}")

        self.assertTrue(np.array_equal(deprecated_version, version))


class TestConvertImageToBitsPlans(unittest.TestCase):
    """
    Class compare the vectorized and not vectorized versions of the
    convert_image_to_bits_plans function.
    """

    def test_convert_image_to_bits_plans(self):
        """
        Test the convert_image_to_bits_plans function.
        """

        # Create a test image
        image = cv2.imread('input_images/test_image.png')
        
        start_time = time.time()
        bits_plans = convert_image_to_bits_plans(image)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"(Non vectorized) Execution time: {execution_time}")

        start_time = time.time()
        bits_plans_vectorized = convert_image_to_bits_plans_vectorized(image)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"(Vectorized) Execution time: {execution_time}")

        self.assertTrue(np.array_equal(bits_plans, bits_plans_vectorized))

if __name__ == '__main__':
    unittest.main()