import json
import os

def load_json_data(file_path):
    """
    Load and parse a JSON data file
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def save_json_data(data, file_path):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path (str): Path where to save the JSON file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
