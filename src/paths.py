import os

project_path = os.path.dirname(__file__)
project_path = project_path.removesuffix('/src')

RAW_PATH = project_path + '/data/raw'
VALIDATED_PATH = project_path + '/data/validated/'
TRANSFORMED_PATH = project_path + '/data/transformed/'
MODEL_PATH = project_path + '/model'
ENV_PATH = project_path

if not os.path.exists(RAW_PATH):
    os.mkdir(RAW_PATH)
    
if not os.path.exists(VALIDATED_PATH):
    os.mkdir(VALIDATED_PATH)
    
if not os.path.exists(TRANSFORMED_PATH):
    os.mkdir(TRANSFORMED_PATH)

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
    
