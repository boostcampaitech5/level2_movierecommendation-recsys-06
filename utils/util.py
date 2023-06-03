import yaml


def get_settings() -> dict:
    '''
    Imports settings from yaml file.

    Returns:
        setting_dict(dict): Dictionary containing settings
    '''

    with open("config.yaml", "r") as f:
        setting_dict: dict = yaml.safe_load(f)
    
    return setting_dict
