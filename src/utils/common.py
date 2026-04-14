import yaml
import os

def read_yaml(file_path:str):
    """
    Reads a YAML file and return content as dictionary
    """

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at:{file_path}")
        
        with open(file_path,"r") as file:
            content = yaml.safe_load(file)

        print(f"YAML file loaded:{file_path}")
        return content 
    
    except Exception as e:
        raise Exception(f"Error reading YAML file:{str(e)}")