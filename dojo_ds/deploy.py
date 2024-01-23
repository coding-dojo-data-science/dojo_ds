
from .utils import create_directories_from_paths
from .utils import print_and_convert_size
from .utils import get_or_print_filesize
from .utils import deep_getsizeof
from .utils import get_filesize


def save_filepath_config(FPATHS, overwrite=True, output_fpath = 'config/filepaths.json',
                        verbose=False):
    ## Save the filepaths 
    import os, json
    from pprint import pprint
    
    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
    if os.path.exists(output_fpath) & (overwrite==False):
        raise Exception(f"- {output_fpath} already exists and overwrite is set to False.")

    with open(output_fpath, 'w') as f:
        json.dump(FPATHS, f)

    print(f"[i] Filepath json saved as {output_fpath} ")
    if verbose:
        pprint(FPATHS)
        
    return FPATHS

import os





def load_filepath_config(config_fpath = 'config/filepaths.json', verbose=True):
    import json 
    from pprint import pprint
    with open (config_fpath) as f:
        FPATHS = json.load(f)
    
    if verbose == True:
        print(f"- Filepaths loaded successfully:")
        pprint(FPATHS)
        
    return FPATHS


