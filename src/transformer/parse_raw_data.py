import os

import pyarrow as pa
import pyarrow.parquet as pq
from alive_progress import alive_bar

from src.pipeline_functions.parse_raw_data_functions import *


def run(browser, directory, file_name, data_dir, http_message):
    dir_path = f"{browser}/{directory}"

    try:
        os.makedirs(f"data/interim/{dir_path}", exist_ok=True)
        print(f"Directory {dir_path} created successfully.")
    except OSError as error:
        print(f"Directory {dir_path} can not be created.")

    parse_dataset(
        file_name,
        dir_path,
        file_name,
        f"{data_dir}/{browser}/{directory}",
        50000,
        http_message,
    )


def parse_dataset(
    origin_file_name: str,
    origin_dir_name: str,
    target_file_name: str,
    target_dir_name: str,
    n_chunks: int,
    http_message: str,
) -> None:
    """
    Parse and process a dataset by performing various tasks, such as processing
    HTTP header fields, labels, and URLs. The dataset is then combined, and the
    target column is created. Finally, the resulting NumPy array is written to a
    Parquet file.

    Parameters
    ----------
    http_message
    origin_file_name : str
        The name of the original input file (without extension) containing the dataset.
    origin_dir_name : str
        The name of the directory where the original input file is located.
    target_file_name : str
        The name of the target output file (without extension) where the processed
        DataFrame will be saved.
    target_dir_name : str
        The name of the directory where the target output file will be saved.
    n_chunks : int
        The number of chunks to use for parallel processing.

    Examples
    --------
    >>> parse_dataset("input_data", "raw_data", "output_data", "interim_data", 10)
    """
    with alive_bar(
        100, force_tty=True, manual=True, title="Initial parsing of data"
    ) as bar:

        bar.text("Read-in parameters")
        target_data_dir = "merged" if "merged" in origin_file_name else "raw"
        compression_alg = "gz" if "merged" in origin_file_name else "gzip"

        print(
            f"Prepare initial dataset:\n"
            f"Path: data/{target_data_dir}/{origin_dir_name}/{origin_file_name}.json.{compression_alg}\n"
            f"Chunk-size: {n_chunks}\n",
            f"Target Filename: {target_file_name}\n",
            f"HTTP Message: {http_message}\n",
        )

        bar(0.05)

        bar.text("Read-in dataset")
        http_data = prepare_initial_dataset(
            origin_file_name,
            origin_dir_name,
            target_data_dir,
            compression_alg,
            http_message,
        )
        bar(0.1)

        bar.text("Parse HTTP Header Fields")
        if http_message == "response":
            parsed_headers = [
                process_header_rows(i) for i in http_data["responseHeaders"]
            ]
        else:
            parsed_headers = [
                process_header_rows(i) for i in http_data["requestHeaders"]
            ]

        column_names = list(set().union(*[set(d.keys()) for d in parsed_headers]))
        bar(0.2)

        bar.text("Remove duplicated Header Fields")
        url_rows_column_names = [*http_data["url"][0]]
        duplicates = find_duplicates(
            [*url_rows_column_names, *column_names, "httpMessageId", "tracker"]
        )
        column_names = rename_duplicates(column_names, duplicates)
        key_mapper = create_key_mapping(duplicates)
        parsed_headers = rename_duplicate_keys(parsed_headers, key_mapper)
        bar(0.3)

        bar.text("Concatenate Header Fields")
        final_headers = concatenate_dicts(parsed_headers, column_names)
        bar(0.4)

        bar.text("Parse HTTP Labels")
        final_labels_raw = process_label_rows(http_data["labels"])
        final_labels = create_target_column(final_labels_raw)
        bar(0.5)

        bar.text("Parse HTTP Ids")
        final_request_ids = process_request_ids_rows(http_data[http_message])
        bar(0.55)

        bar.text("Parse URLs")
        final_urls = process_url_rows(http_data["url"])
        bar(0.6)

        bar.text(f"Combine Results and Write to data/interim as data/{target_dir_name}")
        label_column_name = ["tracker"]
        request_ids_column_name = ["httpMessageId"]
        column_names = [
            *url_rows_column_names,
            *column_names,
            *request_ids_column_name,
            *label_column_name,
        ]
        bar(0.65)

        bar.text("Parse data in chunks")
        parsed_chunks = parse_chunks(
            final_headers, final_urls, final_request_ids, final_labels, 50000
        )
        bar(0.8)

        bar.text("Convert chunks to pyarrow tables")
        b = [pa.Table.from_arrays(elem, names=column_names) for elem in parsed_chunks]
        bar(0.9)

        bar.text("Combine list of data chunks to pyarrow table")
        combined_dataset = pa.concat_tables(b, promote=True)
        bar(0.95)

        bar.text("Write to local file")
        pq.write_table(
            combined_dataset,
            f"data/{target_dir_name}/{target_file_name}.parquet.gzip",
            compression="gzip",
        )
        bar(1)
