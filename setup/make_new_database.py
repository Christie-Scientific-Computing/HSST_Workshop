"""
Script to make a fake db with 10x patients 
"""

import os 
import sqlite3
import numpy as np
import random
import pickle

def get_all_files(root_dir):
    all_paths = []
    for root, dirs, files in os.walk(root_dir):
        if files:
            paths = [os.path.join(root, x) for x in files if x.endswith('.npy')]

            all_paths.extend(paths)
    

    print(f'Found {len(all_paths)} in {root_dir}')
    return all_paths

def filename_to_pid(filename):
    splits = filename.split('_')
    if len(splits) == 2:
        # ID in format ID_SLICE_NUM.npy
        patient_id = splits[0]
        if patient_id.startswith('0522c0'):
            # HnN patients
            patient_id = patient_id.replace('0522c0', 'AmazingImage')
        elif patient_id.startswith('TCGA-CV'):
            patient_id = patient_id.replace('TCGA-CV-', 'AmazingImage')
        else:
            raise ValueError("Unexpected ID for HnN patient")
    elif len(splits) == 3:
        ## ID in format (train/test)_ID_SLICE_NUM.npy
        # Abdo patients
        patient_id = f'AmazingImage{splits[1].zfill(3)}'
    else:
        raise ValueError(f"Don't know how to handle filename {path}")
    return patient_id

def main():
    data_dir = '../data/'
    np.random.seed(seed=1873120)
    con = sqlite3.connect("new_amazing_database.db")
    cur = con.cursor()

    cur.execute(
        """CREATE TABLE IF NOT EXISTS Patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        MRN char NOT NULL,
        age int NOT NULL,
        sex char NOT NULL,
        site char NOT NULL,
        num_slices int NOT NULL,
        x_resolution float NOT NULL,
        y_resolution float NOT NULL
        )""")

    num_slices = {}
    for i, path in enumerate(get_all_files(data_dir)):
        for j in range(1, 11):
            print(path)
            filename = os.path.basename(path)
            patient_id = filename_to_pid(filename)

            is_mask = "True" if '/masks/' in path else "False"
            patient_id = patient_id.replace('AmazingImage', f'AmazingImage_{j}')
            
            if patient_id not in num_slices:
                num_slices[patient_id] = 1
            else:
                num_slices[patient_id] += 1
    ## Shuffle them to make it more intereting
    data = list(num_slices.items())
    random.shuffle(data)

    for i, (patient_id, num_slices) in enumerate(data):
        ## Add patient info
        age = np.random.randint(30, 60)
        sex = np.random.choice(['Male', 'Female'], p=[0.9, 0.1])
        res = random.uniform(0.5, 2)
        num_slices = random.randint(20, 75)

        available_contours = "['Body', 'Brainstem', 'Mandible', 'Parotids', 'Spinalcord']"
        site = random.choice(['Pharynx', 'Larynx', 'Tongue', 'Nasal Cavity'])
        
        cur.execute("""
        INSERT OR IGNORE INTO Patients (id, MRN, age, sex, site, num_slices, x_resolution, y_resolution)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (i, patient_id, age, sex, site, num_slices, np.round(res, 4), np.round(res, 4)) )

    con.commit()

if __name__ == '__main__':
    main()