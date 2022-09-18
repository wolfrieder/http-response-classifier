import gzip
import os
import sys
import json

# Command: python3 compress_raw_json_files.py chrome 08_12_2022
browser = sys.argv[1]
directory = sys.argv[2]
dir_path = f"data/raw/{browser}/{directory}"

entries = [
    entry
    for entry in os.listdir(dir_path)
    if os.path.isfile(os.path.join(dir_path, entry)) & entry.startswith("http")
]

for file in entries:
    print(f"Compress file: {file}")

    with open(f"{dir_path}/{file}", "r") as raw_data:
        raw_data_test = json.loads(raw_data.read())

    with gzip.open(f"{dir_path}/{file}.gzip", "wt") as zipfile:
        json.dump(raw_data_test, zipfile)

    os.remove(f"{dir_path}/{file}")
    print("File successfully compressed.")
