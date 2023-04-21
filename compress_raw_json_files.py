import gzip
import json
import os
import sys

# Command: python3 compress_raw_json_files.py chrome 08_12_2022
browser = sys.argv[1]
directory = sys.argv[2]
dir_path = f"data/raw/{browser}/{directory}"

# sometimes it only works when the script creates the directory
try:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Directory {dir_path} created successfully.")
except OSError as error:
    print(f"Directory {dir_path} can not be created.")

entries = [
    entry
    for entry in os.listdir(dir_path)
    if os.path.isfile(os.path.join(dir_path, entry)) & entry.startswith("http")
]

for file in entries:
    print(f"Compress file: {file}")

    with open(f"{dir_path}/{file}", "r") as raw_data:
        raw_data_test = json.loads(raw_data.read())

    with gzip.open(f"{dir_path}/{file}.gz", "wt") as zipfile:
        json.dump(raw_data_test, zipfile)

    os.remove(f"{dir_path}/{file}")
    print("File successfully compressed.")
