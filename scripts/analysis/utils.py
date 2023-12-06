import yaml


def parse_yaml_file(yaml_file_path: str) -> dict:
    """
    Parse a YAML file and return a dictionary object
    :param yaml_file_path: A string indicating the path to the YAML file
    :return: A dictionary object containing the parsed YAML file
    """
    with open(yaml_file_path, "r") as yaml_file:
        try:
            yaml_obj = yaml.safe_load(yaml_file)
            if yaml_obj is None:
                raise ValueError(f"The supplied YAML file, {yaml_file_path}, is empty")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")
    return yaml_obj
