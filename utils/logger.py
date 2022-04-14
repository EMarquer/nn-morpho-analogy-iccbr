import csv
import logging
import os
import pickle as pkl
from datetime import datetime as dt

def append_pkl(path, row):
    if os.path.exists(path):
        with open(path, "rb") as f: data_dict = pkl.load(f)
    else:
        data_dict = dict()
    timestamp = dt.now().timestamp()
    while timestamp in data_dict.keys(): timestamp+=0.000001 # failsafe, shouldn't be usefull
    data_dict[timestamp] = row
    logging.getLogger(__name__).info(f"writing pickle to {path}")
    with open(path, "wb") as f: pkl.dump(data_dict, f)

def to_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        if isinstance(row, list):
            keys = set()
            for row_ in row:
                keys.update(row_.keys())
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            for row_ in row:
                writer.writerow(row_)
        else:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

def append_csv(path, row):
    write_header = not os.path.exists(path)
    with open(path, "w", newline="") as f:
        if write_header: writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
