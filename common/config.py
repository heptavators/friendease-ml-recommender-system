import yaml

CONFIG_PATH = "config/config.yaml"


def build_config():
    """
    Returns:
        Built configuration
    """

    with open(CONFIG_PATH, "r") as config_file:
        return yaml.safe_load(config_file)


# Initialize config only once when the module is imported
config = build_config()
