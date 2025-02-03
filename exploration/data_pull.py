#this will download the data from huggingface to your local machine in /exploration/ if it doesn't already exist

import requests
import os
import urllib.request


script_dir = os.path.dirname(os.path.abspath(__file__))

sources = {
    "text8": {
        "url": "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8",
        "path": os.path.join(script_dir, "text8"),
    },
    "marco_test": {
        "url": "https://huggingface.co/datasets/microsoft/ms_marco/blob/main/v1.1/test-00000-of-00001.parquet",
        "path": os.path.join(script_dir, "test-marco.parquet"),
    },
    "marco_train": {
        "url": "https://huggingface.co/datasets/microsoft/ms_marco/blob/main/v1.1/train-00000-of-00001.parquet",
        "path": os.path.join(script_dir, "train-marco.parquet"),
    },
    "marco_validation": {
        "url": "https://huggingface.co/datasets/microsoft/ms_marco/blob/main/v1.1/validation-00000-of-00001.parquet",
        "path": os.path.join(script_dir, "validation-marco.parquet"),
    },
}

# For each source, download the file if it doesn't exist
for name, source in sources.items():
    if not os.path.exists(source["path"]):
        print(f"{name} not found, downloading now...")
        urllib.request.urlretrieve(source["url"], source["path"])
        print(f"{name} downloaded and saved to {source['path']}")






    