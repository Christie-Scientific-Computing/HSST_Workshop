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
            patient_id = patient_id.replace('0522c0', 'HnNPatient_')
        elif patient_id.startswith('TCGA-CV'):
            patient_id = patient_id.replace('TCGA-CV-', 'HnNPatient_')
        else:
            raise ValueError("Unexpected ID for HnN patient")
    elif len(splits) == 3:
        ## ID in format (train/test)_ID_SLICE_NUM.npy
        # Abdo patients
        patient_id = f'AbdoPatient_{splits[1].zfill(3)}'
    else:
        raise ValueError(f"Don't know how to handle filename {path}")
    return patient_id

def main():
    data_dir = '../data/'
    np.random.seed(seed=87589)
    con = sqlite3.connect("database.db")
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

    cur.execute("""
    CREATE TABLE IF NOT EXISTS Filepaths(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    MRN char NOT NULL,
    path char NOT NULL,
    is_mask char NOT NULL
    )""")

    ## Get Pixel spacing info
    all_spacings = {}
    for root, dirs, files in os.walk(data_dir):
        if 'spacings.pkl' in files:
            with open(os.path.join(root, 'spacings.pkl'), 'rb') as f:
                spacings = pickle.load(f)

            for key, val in spacings.items():
                pid = filename_to_pid(key)
                all_spacings[pid] = val

    num_slices = {}
    for i, path in enumerate(get_all_files(data_dir)):
        print(path)
        filename = os.path.basename(path)
        patient_id = filename_to_pid(filename)
        is_mask = "True" if '/masks/' in path else "False"
        cur.execute("INSERT OR IGNORE INTO Filepaths (id, MRN, path, is_mask) VALUES (?, ?, ?, ?)", (i, patient_id, path, is_mask) )
        
        if patient_id not in num_slices:
            num_slices[patient_id] = 1
        else:
            num_slices[patient_id] += 1

    for i, (patient_id, num_slices) in enumerate(num_slices.items()):
        ## Add patient info
        age = np.random.randint(45, 85)
        sex = random.choice(['Male', 'Female'])
        if patient_id.startswith('Abdo'):
            site = random.choice(['Liver', 'Kidney', 'Stomach', 'Spleen', 'Pancreas'])
        elif patient_id.startswith('HnN'):
            site = random.choice(['Pharynx', 'Larynx', 'Tongue', 'Nasal Cavity'])
        else:
            raise ValueError(f"Don't know how to assign site to ID")
        x_res, y_res = all_spacings[patient_id]
        cur.execute("""
        INSERT OR IGNORE INTO Patients (id, MRN, age, sex, site, num_slices, x_resolution, y_resolution)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (i, patient_id, age, sex, site, num_slices, np.round(x_res, 4), np.round(y_res, 4)) )

    con.commit()

if __name__ == '__main__':
    main()