import kagglehub
import numpy as np
import os

def get_data():
    path = kagglehub.dataset_download("yorkyong/text8-zip")
    path = os.path.join(path, "text8")

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    
    return text


