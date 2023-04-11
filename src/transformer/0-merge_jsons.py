from os import listdir, makedirs, SEEK_CUR
from os.path import join
from urllib.parse import urlparse
from json_stream import load
from json_stream.dump import JSONStreamEncoder, default

import json
import gzip
import pathlib
import sys


def generate_url(r):
    parsed_url = urlparse(r["url"])
    return {
        "hostname": parsed_url.netloc,
        "pathname": parsed_url.path,
        "filetype": parsed_url.path.split("/").pop().split(".").pop(),
        "filename": parsed_url.path.split("/").pop(),
        "protocol": parsed_url.scheme,
        "query": parsed_url.query,
    }


def generate_response_headers(r):
    headers = r["response"]["responseHeaders"]
    transformed = []

    for header in headers:
        transformed.append([header["name"], header["value"]])

    return transformed


if __name__ == "__main__":

    # Inputs
    root = pathlib.Path().resolve()
    path = sys.argv[1]
    data_dir = join(root, path)
    jsons = [
        x for x in listdir(data_dir) if x.startswith("http-") and x.endswith(".json.gz")
    ]

    # Outputs
    output_dir = join(root, sys.argv[2])
    makedirs(output_dir, exist_ok=True)
    out = join(root, sys.argv[2], "merged_data.json.gz")
    output_file = gzip.open(out, "at", encoding="utf-8")
    output_file.write("[")

    print(f"\nSuccessfully created {output_dir}\n")

    print(f"\nRead {len(jsons)} files from {data_dir}\n")

    first = True

    for index, file in enumerate(jsons):
        with gzip.open(join(data_dir, file), "rt") as f:
            requests = load(f)
            for r in requests.persistent():
                if not first:
                    output_file.write(",")

                first = False

                t = {
                    "url": generate_url(r),
                    "labels": json.dumps(r["labels"], cls=JSONStreamEncoder),
                    "response": {
                        "statusCode": r["response"]["statusCode"],
                        "fromCache": r["response"]["fromCache"],
                    },
                    "responseHeaders": generate_response_headers(r),
                }

                output_file.write(str(t))

    output_file.write("]")
    output_file.close()

    print(f"\nSuccessfully written {out}\n")
