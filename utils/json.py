import json
import os

def load_json(file_path: str):
    """
    Load JSON data from a file
    
    Params:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Successfully loaded {len(data)} documents from {file_path}")
            return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return [] # Avoid crashing the program

def save_json(data, file_path: str, append: bool = False, overwrite: bool = True):
    """
    Save data to a JSON file
    
    Params:
        data: Data to save
        file_path: Path to save the file
        append: Add to existing file
        overwrite: Allow overwriting existing file
    
    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    # Handle directory creation
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Check overwrite condition
    if os.path.exists(file_path) and not overwrite and not append:
        raise FileExistsError(f"File {file_path} exists and overwrite is False")

    # Handle append condition
    if append and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                old_data = json.load(file)
                if isinstance(old_data, dict) and isinstance(data, dict):
                    data = {**old_data, **data}
        except (json.JSONDecodeError, IOError):
            pass

    # Write the data with fixed indent
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)