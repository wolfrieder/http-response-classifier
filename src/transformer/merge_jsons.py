import gzip
import json
import pathlib
import sys
from os import listdir, makedirs
from os.path import join
from typing import Dict, List, Any, Union
from urllib.parse import urlparse

from alive_progress import alive_bar
from json_stream import load
from json_stream.dump import JSONStreamEncoder


def generate_url(r: Dict[str, Any]) -> Dict[str, Union[str, List[str]]]:
    """
    Extract and generate URL-related information from a given request dictionary.

    Parameters
    ----------
    r : Dict[str, Any]
        The input request dictionary containing the 'url' key.

    Returns
    -------
    Dict[str, Union[str, List[str]]]
        A dictionary containing URL-related information like hostname, pathname,
        filetype, filename_one, protocol, and query.
    """
    parsed_url = urlparse(r["url"])
    return {
        "hostname": parsed_url.netloc,
        "pathname": parsed_url.path,
        "filetype": parsed_url.path.split("/").pop().split(".").pop(),
        "filename_one": parsed_url.path.split("/").pop(),
        "protocol": parsed_url.scheme,
        "query": parsed_url.query,
    }


def generate_headers(r: Dict[str, Any], httpMessage, headers) -> List[List[str]]:
    """
    Extract and generate response headers from a given request dictionary.

    Parameters
    ----------
    r : Dict[str, Any]
        The input request dictionary containing the 'responseHeaders' key.

    Returns
    -------
    List[List[str]]
        A list of lists, where each inner list contains a header name and its value.
    """
    if httpMessage == "response":
        headers = r[httpMessage][headers]
    else:
        headers = r[headers]

    transformed = []

    for header in headers:
        try:
            transformed.append([header["name"], header["value"]])
        except KeyError:
            print(header["name"])
    return transformed


if __name__ == "__main__":
    # Inputs
    root = pathlib.Path().resolve()
    data_dir = sys.argv[1]
    http_message = sys.argv[3]

    jsons = [
        x for x in listdir(data_dir) if x.startswith("http-") and x.endswith(".json.gz")
    ]

    # Outputs
    output_dir = sys.argv[2]
    makedirs(output_dir, exist_ok=True)
    if http_message == "response":
        out = join(sys.argv[2], "merged_data_response.json.gz")
    else:
        out = join(sys.argv[2], "merged_data_request.json.gz")
    output_file = gzip.open(out, "at", encoding="utf-8")
    output_file.write("[")

    print(f"\nSuccessfully created {output_dir}\n")

    print(f"\nRead {len(jsons)} files from {data_dir}\n")

    print(f"Request?: {http_message}")

    first = True

    if http_message == "response":
        print(f"HTTP Message: {http_message}")
        print(
            "\n The upper boundary of parsed items per file (71000) is only an estimate."
        )
        for index, file in enumerate(jsons):
            with alive_bar(71000, title=f"Parse {file}") as bar:
                with gzip.open(join(data_dir, file), "rt") as f:
                    requests = load(f)
                    for r in requests.persistent():
                        if not first:
                            output_file.write(", ")

                        first = False

                        t = {
                            "url": generate_url(r),
                            "labels": r["labels"],
                            "response": {
                                "statusCode": r["response"]["statusCode"],
                                "fromCache": r["response"]["fromCache"],
                                "requestId": r["requestId"],
                            },
                            "responseHeaders": generate_headers(
                                r, "response", "responseHeaders"
                            ),
                        }

                        output_file.write(json.dumps(t, cls=JSONStreamEncoder))
                        bar()

    else:
        print(f"HTTP Message: {http_message}")
        print(
            "\n The upper boundary of parsed items per file (71000) is only an estimate."
        )
        for index, file in enumerate(jsons):
            with alive_bar(71000, title=f"Parse {file}") as bar:
                with gzip.open(join(data_dir, file), "rt") as f:
                    requests = load(f)
                    for r in requests.persistent():
                        if not first:
                            output_file.write(", ")

                        first = False

                        t = {
                            "url": generate_url(r),
                            "labels": r["labels"],
                            "request": {
                                "method": r["method"],
                                "type": r["type"],
                                "requestId": r["requestId"],
                            },
                            "requestHeaders": generate_headers(
                                r, "request", "requestHeaders"
                            ),
                        }

                        output_file.write(json.dumps(t, cls=JSONStreamEncoder))
                        bar()

    output_file.write("]")
    output_file.close()

    print(f"\nSuccessfully written {out}\n")
