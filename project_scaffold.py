import os
import logging
from pathlib import Path


# Step 1: Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Step 2: Define all directories
dir_list =[
    
    "config", 
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src",
    "src/versioning",
    "src/data",
    "src/utils",
    "src/config",
    "src/models",
    "training",
    "training/logs",
    "training/checkpoints",
    "tests",
    "tests/unit",
    "tests/integration"
    
]

# Step 3: Define all files explicitly
file_list = [
    
    "config/config.yaml",
    "data/raw/train.csv",
    "data/raw/test.csv",
    "data/processed/train.csv",
    "data/processed/test.csv",
    "notebooks/eda.ipynb",
    "notebooks/model_training.ipynb",
    "src/__init__.py",
    "src/versioning/__init__.py",
    "src/data/__init__.py",
    "src/utils/__init__.py",
    "src/config/__init__.py",
    "src/models/__init__.py",
    "training/train.py",
    "training/logs/logging.yaml",
    "training/checkpoints/model.ckpt",
    "tests/__init__.py",
    "tests/unit/test_utils.py",
    "tests/integration/test_training.py", 
    "setup.py", 
    "requirements.txt", 
    "README.md"
    
]

# Step 4: Create the directories first
for directories in dir_list:
    dir_path = Path(directories)
    if not dir_path.exists():
        logging.info(f"Creating directory: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)

# Step 5: Create the files
for files in file_list:
    file_path = Path(files)
    if not file_path.exists():
        logging.info(f"Creating file: {file_path}")
        file_path.touch(exist_ok=True)
