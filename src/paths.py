import os

project_path = os.path.dirname(__file__)
project_path = project_path.removesuffix('/src')

RAW_PATH = project_path + '/data/raw'
VALIDATED_PATH = project_path + '/data/validated/'
TRANSFORMED_PATH = project_path + '/data/transformed/'
MODEL_PATH = project_path + '/model'
ENV_PATH = project_path

if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH,exist_ok=True) #Creates necessary parent directories also
    
if not os.path.exists(VALIDATED_PATH):
    os.makedirs(VALIDATED_PATH,exist_ok=True)
    
if not os.path.exists(TRANSFORMED_PATH):
    os.makedirs(TRANSFORMED_PATH,exist_ok=True)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH,exist_ok=True)
    
