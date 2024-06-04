"""Filesize analysis functions.
Created with ChatGPT.

# Example usage

# Example usage
>> directory_path = '/path/to/your/directory'
>> df = list_files_in_directory(directory_path, size_threshold=100, recursive=True, return_df=True, details=True, unit='MB', sort=True, only_large=True)
>> display(df)

"""
import os
import re
import pandas as pd
from datetime import datetime
import stat

def get_file_size(size_in_bytes, unit=None):
    if unit is None:
        if size_in_bytes < 1024:
            return f"{size_in_bytes} B", size_in_bytes / (1024**2)
        elif size_in_bytes < 1024**2:
            return f"{size_in_bytes / 1024:.2f} KB", size_in_bytes / (1024**2)
        elif size_in_bytes < 1024**3:
            return f"{size_in_bytes / (1024**2):.2f} MB", size_in_bytes / (1024**2)
        else:
            return f"{size_in_bytes / (1024**3):.2f} GB", size_in_bytes / (1024**2)
    else: 
        unit = unit.upper()
        if unit == 'B':
            return f"{size_in_bytes} B", size_in_bytes / (1024**2)
        elif unit == 'KB':
            return f"{size_in_bytes / 1024:.2f} KB", size_in_bytes / (1024**2)
        elif unit == 'MB':
            return f"{size_in_bytes / (1024**2):.2f} MB", size_in_bytes / (1024**2)
        elif unit == 'GB':
            return f"{size_in_bytes / (1024**3):.2f} GB", size_in_bytes / (1024**2)

def get_permissions_text(mode):
    """Generate a string representing file permissions in Unix format."""
    is_dir = 'd' if stat.S_ISDIR(mode) else '-'
    perm_str = is_dir + \
               ''.join([(('r', '-')[bool(mode & perm)] +
                         ('w', '-')[bool(mode & perm)] +
                         ('x', '-')[bool(mode & perm)]) for perm in [stat.S_IRUSR, stat.S_IWUSR, stat.S_IXUSR,
                                                                    stat.S_IRGRP, stat.S_IWGRP, stat.S_IXGRP,
                                                                    stat.S_IROTH, stat.S_IWOTH, stat.S_IXOTH]])
    return perm_str

def get_filetype_emoji(file_type):
    """Return an emoji representing the file type."""
    file_type_emojis = {
        '.txt': '📄',
        '.pdf': '📄',
        '.doc': '📄',
        '.docx': '📄',
        '.xls': '📊',
        '.xlsx': '📊',
        '.csv': '📊',
        '.py': '🐍',
        '.jpg': '🖼️',
        '.jpeg': '🖼️',
        '.png': '🖼️',
        '.gif': '🖼️',
        '.zip': '🗜️',
        '.tar': '🗜️',
        '.gz': '🗜️',
        '.mp3': '🎵',
        '.wav': '🎵',
        '.mp4': '🎥',
        '.mkv': '🎥',
        '.html': '🌐',
        '.css': '🎨',
        '.js': '💻',
        '.json': '🔧',
        '.xml': '🔧',
        '.yaml': '🔧',
        '.yml': '🔧',
        '.joblib': '🔧',
        '.pb': '🧠',
        '.h5': '🧠',
        '.tfrecord': '🧠',
        '.index': '🧠',
        '.data': '🧠',
        '.ckpt': '🧠',
        '.tflite': '🧠',
        '.tf record': '🧠',
        '.pth': '🧠',
        '.pt': '🧠',
        '.onnx': '🧠',
        '.pkl': '🧠',
        '.pickle': '🧠',
        '.hdf5': '🧠',
        '.npy': '🧠',
        '.safetensor': '🧠',
    }
    # Check for exact match in predefined extensions
    if file_type in file_type_emojis:
        return file_type_emojis[file_type]
    
    
    return '📁'

def is_path_compatible(file_path):
    """Check if the file path is compatible with Windows, macOS, and Linux."""
    windows_invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    mac_invalid_chars = r'[:]'
    linux_invalid_chars = r'[\x00]'
    
    windows_invalid = bool(re.search(windows_invalid_chars, file_path))
    mac_invalid = bool(re.search(mac_invalid_chars, file_path))
    linux_invalid = bool(re.search(linux_invalid_chars, file_path))
    
    return not (windows_invalid or mac_invalid or linux_invalid)

def list_files_in_directory(directory, size_threshold=100, recursive=True, return_df=True, details=False, unit='MB', sort=False, only_large=False, include_dir_details=False, check_compatibility=False, only_incompatible=False):
    """
    List files in a directory and its subdirectories.

    Args:
        directory (str): The path to the directory.
        size_threshold (int, optional): The size threshold in the specified units. Files larger than this threshold will be considered as "large". Defaults to 100.
        recursive (bool, optional): Whether to list files recursively in subdirectories. Defaults to False.
        return_df (bool, optional): Whether to return a DataFrame. If False, prints the results. Defaults to True.
        details (bool, optional): Whether to include detailed information (owner, permissions). Defaults to False.
        unit (str, optional): The unit for displaying file sizes. Defaults to 'MB'.
        sort (bool, optional): Whether to sort the results from largest to smallest. Defaults to False.
        only_large (bool, optional): Whether to include only files larger than the size threshold. Defaults to False.
        include_dir_details (bool, optional): Whether to include total size and number of files for directories. Defaults to False.
        check_compatibility (bool, optional): Whether to check if file paths are compatible with all OS. Defaults to False.
        only_incompatible (bool, optional): Whether to include only files with incompatible paths. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing information about the files in the directory if return_df is True.

    Example usage:
        directory_path = '/path/to/your/directory'
        df = list_files_in_directory(directory_path, size_threshold=100, recursive=True, return_df=True, details=True, unit='MB', sort=True, only_large=True, include_dir_details=True, check_compatibility=True, only_incompatible=False)
        print(df)
    """
    file_list = []

    # Convert the size threshold to bytes based on the specified unit
    if unit == 'B':
        size_threshold_bytes = size_threshold
    elif unit == 'KB':
        size_threshold_bytes = size_threshold * 1024
    elif unit == 'MB':
        size_threshold_bytes = size_threshold * 1024 * 1024
    elif unit == 'GB':
        size_threshold_bytes = size_threshold * 1024 * 1024 * 1024
    else:
        raise ValueError("Invalid unit. Please use 'B', 'KB', 'MB', or 'GB'.")
    
    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        # Add directories to the list if not recursive
        if not recursive or include_dir_details:
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                dir_stats = os.stat(dir_path)
                
                # Get directory creation and modification dates
                creation_date = datetime.fromtimestamp(dir_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                modification_date = datetime.fromtimestamp(dir_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Initialize directory size and file count
                dir_size = 0
                file_count = 0
                
                # Calculate total size and number of files if requested
                if include_dir_details:
                    for dir_root, _, dir_files in os.walk(dir_path):
                        for f in dir_files:
                            file_count += 1
                            f_path = os.path.join(dir_root, f)
                            dir_size += os.path.getsize(f_path)
                
                dir_size_str, _ = get_file_size(dir_size, unit=unit)
                
                # Check path compatibility if requested
                path_compatible = True
                if check_compatibility:
                    path_compatible = is_path_compatible(dir_path)
                    if only_incompatible and path_compatible:
                        continue
                
                # Create a dictionary of directory details
                dir_info = {
                    'Name': dir_name,
                    'Absolute Path': dir_path,
                    'Size': f"{file_count} Files",
                    'Size (MB)': dir_size / (1024**2),
                    'Creation Date': creation_date,
                    'Modification Date': modification_date,
                    'File Type': 'directory',
                    'File Type Icon': '🗂️',
                    'Is Large File': False,
                    # 'Path Compatible': path_compatible
                }
                if check_compatibility:
                    dir_info['Path Compatible'] = path_compatible
                
                if return_df:
                    file_list.append(dir_info)
                else:
                    # Print directory details in a professional format
                    for key, value in dir_info.items():
                        if key not in {'File Type Icon', 'Name', 'Is Large File'}:
                            print(f"{key:25}: {value}")
                    print("-" * 80)
        
        for file in files:
            file_path = os.path.join(root, file)
            file_stats = os.stat(file_path)
            
            # Gather file details
            file_size = file_stats.st_size
            size_str, size_in_mb = get_file_size(file_size, unit=unit)
            is_large_file = file_size > size_threshold_bytes
            
            if only_large and not is_large_file:
                continue
            
            # Get file creation and modification dates
            creation_date = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            modification_date = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

            # Determine file type and its corresponding emoji
            file_type = os.path.splitext(file)[-1].lower()
            if re.match(r'\.data-\d{5}-of-\d{5}', file_type):
                file_type = 'tf record'  # Normalize to generic '.data' file type
                
            file_type_icon = get_filetype_emoji(file_type)
            # Check path compatibility if requested
            path_compatible = True
            if check_compatibility:
                path_compatible = is_path_compatible(file_path)
                if only_incompatible and path_compatible:
                    continue

            # Create a dictionary of file details
            file_info = {
                'Name': file,
                'Absolute Path': file_path,
                'Size': size_str,
                'Size (MB)': size_in_mb,
                'Creation Date': creation_date,
                'Modification Date': modification_date,
                'File Type': file_type,
                'File Type Icon': file_type_icon,
                'Is Large File': is_large_file,
            }
            if check_compatibility:
                file_info['Path Compatible'] = path_compatible

            # Include detailed information if requested
            if details:
                file_info['File Owner'] = file_stats.st_uid
                file_info['File Permissions (Octal)'] = oct(file_stats.st_mode & 0o777)
                file_info['File Permissions (Text)'] = get_permissions_text(file_stats.st_mode)

            file_list.append(file_info)
        
        # If not recursive, break after processing the top directory
        if not recursive:
            break
    
    # Sort the list if requested
    if sort:
        file_list.sort(key=lambda x: x['Size (MB)'], reverse=True)
    
    # Return the DataFrame if requested
    if return_df:
        df = pd.DataFrame(file_list)
        # Reorder the columns
        cols = ['Name', 'File Type', 'File Type Icon', 'Size', 'Creation Date', 'Modification Date'] + [col for col in df.columns if col not in ['Name', 'File Type', 'File Type Icon', 'Size', 'Creation Date', 'Modification Date']]
        df = df[cols].copy()
        return df
    else:
        # Print file details in a professional format, excluding the icon, name, and is_large_file
        for file_info in file_list:
            for key, value in file_info.items():
                if key not in {'File Type Icon', 'Name', 'Is Large File'}:
                    print(f"{key:25}: {value}")
            print("-" * 80)
