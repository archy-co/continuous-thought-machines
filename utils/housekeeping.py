import numpy as np
import random
import torch


import os
import zipfile
import glob

def zip_python_code(output_filename):
    """
    Zips all .py files in the current repository and saves it to the 
    specified output filename.

    Args:
        output_filename: The name of the output zip file. 
                         Defaults to "python_code_backup.zip".
    """

    with zipfile.ZipFile(output_filename, 'w') as zipf:
        files = glob.glob('models/**/*.py', recursive=True) + glob.glob('utils/**/*.py', recursive=True) + glob.glob('tasks/**/*.py', recursive=True) + glob.glob('*.py', recursive=True)
        for file in files:
            root = '/'.join(file.split('/')[:-1])
            nm = file.split('/')[-1]
            zipf.write(os.path.join(root, nm))

def set_seed(seed=42, deterministic=True):
    """
    ... and the answer is ... 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False



import csv

def save_dict_to_csv(dictionary, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['meaning', 'number'])
        for meaning, number in dictionary.items():
            writer.writerow([meaning, number])
    print(f"Dictionary saved to {filename}")

def load_dict_from_csv(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return {}

    dictionary = {}
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                if len(row) == 2:
                    meaning, number = row
                    dictionary[meaning] = int(number)
        print(f"Loaded {len(dictionary)} entries from {filename}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return {}

    return dictionary

def get_meaning(number, dictionary):
    for meaning, num in dictionary.items():
        if num == number:
            return meaning
    return None

def get_number(meaning, dictionary):
    return dictionary.get(meaning, None)
