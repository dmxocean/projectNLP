

def json_structure(data, max_depth=5, current_depth=0, path="root"):
    """
    Recursively explore and print the structure of JSON data up to a specified depth
    
    Parameters:
        data: The JSON data to explore
        max_depth: Maximum depth to explore
        current_depth: Current depth in the recursion

    Args:
        data: The JSON data to explore
        max_depth: Maximum depth to explore
        current_depth: Current depth in the recursion
        path: Current path in the JSON structure
    """
    
    # Indentation for visualizing depth
    indent = (4*" ") * current_depth
    
    if current_depth >= max_depth:
        print(f"{indent}[Reached max depth at {path}]")
        return
    
    # Handle dictionary
    if isinstance(data, dict):
        print(f"{indent}{path} (dict with {len(data)} keys)")
        for key, value in data.items():
            new_path = f"{path}.{key}" if path != "root" else key
            json_structure(value, max_depth, current_depth + 1, new_path)
    
    # Handle list
    elif isinstance(data, list):
        print(f"{indent}{path} (list with {len(data)} items)")
        if data and current_depth < max_depth - 1:
            # Show structure of first item as an example
            sample_item = data[0]
            new_path = f"{path}[0]"
            json_structure(sample_item, max_depth, current_depth + 1, new_path)
            
            # If there are multiple different structures in the list, show another example
            if len(data) > 1 and not all(type(item) == type(sample_item) for item in data):
                different_type_item = next((item for item in data if type(item) != type(sample_item)), None)
                if different_type_item:
                    new_path = f"{path}[different_type]"
                    json_structure(different_type_item, max_depth, current_depth + 1, new_path)

    # Handle other types
    elif isinstance(data, (str, int, float, bool, type(None))):
        # For primitive types, show a sample value
        if isinstance(data, str) and len(data) > 30:
            sample_value = f"{data[:30]}..." 
        else:
            sample_value = data
        print(f"{indent}{path} ({type(data).__name__}): {sample_value}")


def json_keys(data):
    """
    Extract the key structural components of the dataset
    
    Parameters:
        data (list): The JSON data to explore

    Returns:
        dict: Dictionary containing the structural information
    """
    structure = {
        "main_keys": set(),
        "data_keys": set(),
        "prediction_keys": set(),
        "result_keys": set(),
        "value_keys": set(),
        "label_types": set()
    }
    
    # Process documents to extract generic structure
    for doc in data[:15]:
        # Main level keys
        for key in doc.keys():
            structure["main_keys"].add(key)
        
        # Data level keys
        if "data" in doc:
            for key in doc["data"].keys():
                structure["data_keys"].add(key)
        
        # Prediction level
        if "predictions" in doc and doc["predictions"]:
            for pred in doc["predictions"]:
                for key in pred.keys():
                    structure["prediction_keys"].add(key)
                
                # Result level
                if "result" in pred:
                    for res in pred["result"]:
                        for key in res.keys():
                            structure["result_keys"].add(key)
                        
                        # Value level and labels
                        if "value" in res:
                            for key in res["value"].keys():
                                structure["value_keys"].add(key)
                            
                            # Collect label types
                            if "labels" in res["value"]:
                                for label in res["value"]["labels"]:
                                    structure["label_types"].add(label)
    
    # Convert sets to sorted lists for nicer output
    for key in structure:
        structure[key] = sorted(list(structure[key]))
    
    return structure


def extract_annotations(data):
    """
    Extract all annotations/predictions from the documents to a Dataframe
    
    Parameters:
        data (list): List of document JSON objects

    Returns:
        pd.DataFrame: DataFrame containing all annotations
    """

    # Dataframe new rows
    rows = [] 
    
    # Iterate through all documents
    for doc_index, doc, in enumerate(data):
        doc_id = doc["data"]["id"]
        text = doc["data"]["text"]

        # Empty row for non predicted 
        if "predictions" not in doc or not doc["predictions"]:
            row = {
                "doc_index": doc_index, # Represent the position of each document in your data array
                "doc_id": doc_id,
                "result_id": None,  # No result ID since there are no predictions
                "start": None,      # No start position
                "end": None,        # No end position
                "label": "No Prediction",  # Special label to indicate absence
                "text": None,       # No specific text segment
                "context": text[:50] + "..." if len(text) > 50 else text  # First 50 chars for context
            }
            rows.append(row)
            continue 
        
        # Process predictions
        for pred_index, pred in enumerate(doc["predictions"]):
            if "result" in pred:
                for res_index, res in enumerate(pred["result"]):
                    if "value" in res and "labels" in res["value"]:
                        start = res["value"]["start"]
                        end = res["value"]["end"]
                        
                        # Extract the labeled text segment
                        segment_text = text[start:end]
                        
                        # Create a row for each label
                        for label in res["value"]["labels"]:
                            row = {
                                "doc_index": doc_index,
                                "doc_id": doc_id,
                                "result_id": res["id"],
                                "start": start,
                                "end": end,
                                "label": label,
                                "text":segment_text,
                            }
                            rows.append(row)

                        # Sort rows by start position
                        rows = sorted(rows, key=lambda x: x["start"])
                        rows = sorted(rows, key=lambda x: x["doc_index"])
    
    return pd.DataFrame(rows)


import re

def get_punctuation_stops(text: str) -> list[int]:
    indices = []
    regex = r'\.(?!\d\w)'  # Match dots not followed by digits
    for match in re.finditer(regex, text):
        index = match.start()

        if text[index-3: index] == " dr":
            pass
        elif text[index-3: index] == "(dr":
            pass
        elif text[index-3: index] == "dra":
            pass
        elif text[index-4: index] == "(dra":
            pass
        elif text[index-2: index] == "..":
            pass
        elif text[index-1: index] == ".":
            pass
        elif text[index+2] == ".":
            pass
        elif text[index-1] == " ":
            pass
        else:
            indices.append(index)

    return indices

def extract_documents(data):
    """
    Extract all documents to a DataFrame
    """

    for doc_index, doc in enumerate(data):
        # save to txt
        with open(f"../data/documents/{doc['data']['id']}.txt", "w") as f:
            text = doc["data"]["text"].strip()
            stops = get_punctuation_stops(text)

            start = 0
            for stop in stops:
                part = text[start:stop + 1].strip()  # Include the dot
                if part:
                    f.write(part + "\n")
                start = stop + 1

            # Handle the remaining text after the last stop
            remaining = text[start:].strip()
            if remaining:
                f.write(remaining)
                f.write("\n")