import os


def create_dir_if_not_exists(path):
    """
    Creates directory if it does not exist
    :param path: Directory
    :type path: str
    """
    if not os.path.exists(path):
        os.makedirs(path)
