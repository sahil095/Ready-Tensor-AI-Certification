import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_FPATH = os.path.join(ROOT_DIR, ".env")

CODE_DIR = os.path.join(ROOT_DIR, "code")

APP_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "config.yaml")
PROMPT_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "prompt_config.yaml")


OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")


DATA_DIR = os.path.join(ROOT_DIR, "data")
PUBLICATION_FPATH = os.path.join(DATA_DIR, "publication.md")

CONFIG_DIR = os.path.join(ROOT_DIR, "config")
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "config.yaml")
PROMPT_CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "prompt_config.yaml")