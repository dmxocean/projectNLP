import json
from typing import Dict, List

def load_json_data(file_path: str) -> List[Dict]:
    """
    Load data from JSON file
    """

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Successfully loaded data from: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
    

def explore_json_structure(data) -> Dict:
    """
    Analyze and display JSON structure of the data
    """

    if not data:
        print("No data available for analysis")
        return {}

    record_first = data[0]  # Get main keys from first document
    main_keys = list(record_first.keys()) if isinstance(record_first, dict) else []

    structure = {"main_keys": main_keys}  # Create structure dictionary

    print(f"Type document: {type(record_first)}")  # Print document structure information
    print(f"Keys document: {main_keys}")
    print()

    # Explore each main key
    for main_key in main_keys:
        if main_key not in record_first:
            continue

        print(f"Key: {main_key}")
        print(f"Type: {type(record_first[main_key])}")

        try:  # Get length if possible
            length = len(record_first[main_key])
            print(f"Length: {length}")
        except:
            print("Length: N/A")

        value_str = str(record_first[main_key])  # Print value (truncated if needed)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        print(f"Value: {value_str}")
        print()

        # Explore its nested structure
        if main_key == "data" and isinstance(record_first[main_key], dict):
            structure["data_keys"] = list(record_first["data"].keys())

            for k in record_first["data"].keys():
                print(f"\tKey: {k}")
                print(f"\tType: {type(record_first['data'][k])}")

                # Get length if possible
                try:
                    length = len(str(record_first["data"][k]))
                    print(f"\tLength: {length}")
                except:
                    print("\tLength: N/A")

                value_str = str(record_first["data"][k])
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"\tValue: {value_str}")
                print()

        elif main_key == "predictions" and isinstance(record_first[main_key], list) and record_first[main_key]:
            print("\tPredictions structure:")
            pred = record_first[main_key][0]
            print(f"\tPrediction keys: {list(pred.keys()) if isinstance(pred, dict) else 'N/A'}")

            if isinstance(pred, dict) and "result" in pred and pred["result"]:
                result = pred["result"][0]
                print(f"\tResult keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")

                if isinstance(result, dict) and "value" in result:
                    value = result["value"]
                    print(f"\tValue keys: {list(value.keys()) if isinstance(value, dict) else 'N/A'}")
                    print()

    return structure