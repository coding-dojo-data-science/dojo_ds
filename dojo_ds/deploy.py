
from .utils import create_directories_from_paths
from .utils import print_and_convert_size
from .utils import get_or_print_filesize
from .utils import deep_getsizeof
from .utils import get_filesize


def save_filepath_config(FPATHS, overwrite=True, output_fpath='config/filepaths.json',
                        verbose=False):
    """
    Save the filepaths to a JSON file.

    Parameters:
    FPATHS (dict): A dictionary containing the filepaths.
    overwrite (bool): Whether to overwrite the existing file if it already exists. Default is True.
    output_fpath (str): The output filepath to save the JSON file. Default is 'config/filepaths.json'.
    verbose (bool): Whether to print the saved filepaths. Default is False.

    Returns:
    dict: The dictionary containing the filepaths.

    Raises:
    Exception: If the output file already exists and overwrite is set to False.
    """
    import os
    import json
    from pprint import pprint
    
    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
    if os.path.exists(output_fpath) and not overwrite:
        raise Exception(f"- {output_fpath} already exists and overwrite is set to False.")

    with open(output_fpath, 'w') as f:
        json.dump(FPATHS, f)

    print(f"[i] Filepath json saved as {output_fpath} ")
    if verbose:
        pprint(FPATHS)
        
    return FPATHS

import os





def load_filepath_config(config_fpath = 'config/filepaths.json', verbose=True):
    """
    Loads the filepaths configuration from a JSON file.

    Parameters:
    - config_fpath (str): The filepath of the JSON configuration file. Default is 'config/filepaths.json'.
    - verbose (bool): Whether to print the loaded filepaths. Default is True.

    Returns:
    - dict: A dictionary containing the loaded filepaths.
    """
    import json 
    from pprint import pprint
    with open (config_fpath) as f:
        FPATHS = json.load(f)
    
    if verbose == True:
        print(f"- Filepaths loaded successfully:")
        pprint(FPATHS)
        
    return FPATHS


