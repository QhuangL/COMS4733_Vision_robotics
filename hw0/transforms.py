from numba import njit, prange
import numpy as np

def transform_is_valid(t, tolerance=1e-3):
    """ Check if array is a valid transform.
    You can refer to the lecture notes to 
    see how to check if a matrix is a valid
    transform. 

    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.
    """
    # TODO: 

def transform_concat(t1, t2):
    """ Concatenate two transforms. Hint: 
        use numpy matrix multiplication. 

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2.
    """
    # TODO: 

def transform_point3s(t, ps):
    """ Transfrom a list of 3D points
    from one coordinate frame to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    # TODO: 

def transform_inverse(t):
    """Find the inverse of the transfom. Hint:
        use Numpy's linear algebra native methods. 

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    # TODO: 

