class AttrDict(dict):
    """A simple attribute-accessible dictionary."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'SimpleAttrDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value


def read_config(file_path: str):
    import yaml
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            return None

    config = AttrDict(config)
    str2list = lambda s: [float(item) for item in s.split(",")]

    if 'aabb' in config:
        config.aabb = str2list(config.aabb)
    if 'shading_sample_prob' in config:
        config.shading_sample_prob = str2list(config.shading_sample_prob)

    return config