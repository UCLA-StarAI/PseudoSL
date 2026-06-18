import os
from pathlib import Path

PERSPECTIVE_API_KEY = os.environ.get("PERSPECTIVE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

##############################
# Paths
##############################
SHARED_DIR = Path('/data/language-model-toxicity')
DATA_DIR = SHARED_DIR / 'data'
OUTPUT_DIR = Path('output')

OPENWEBTEXT_DB = DATA_DIR / 'openwebtext-perspective.db'
WEBTEXT_DB = DATA_DIR / 'webtext.db'

# TODO: remove
TEXTS_DIR = DATA_DIR / 'openwebtext'

##############################
# Perspective API
##############################
PERSPECTIVE_API_LEN_LIMIT = 20480

# All attributes can be found here:
# https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md
PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    'FLIRTATION'
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
