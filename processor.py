from timeit import default_timer as timer
import numpy as np
import pandas as pd
# import modin.pandas as pd
# from flatten_json import flatten
# import dask.dataframe as dd
# import pyarrow
# import fastparquet
import ray


def read_json_file(path):
    return pd.read_json(path)


def read_parquet_file(name):
    return pd.read_parquet(f'parquet_data/{name}.parquet.gzip')


def write_to_parquet_file(dataframe, name):
    dataframe.to_parquet(f'parquet_data/{name}.parquet.gzip', compression='gzip')
    return print('Finished')


def combine_datasets(names):
    return pd.concat([read_parquet_file(i) for i in names], ignore_index=True)


@ray.remote
def process_rows(row):
    df_row = pd.json_normalize(row)
    columns = np.vectorize(str.lower)(df_row.name.values)
    result = pd.DataFrame(df_row.value.values.reshape(1, -1), columns=columns)
    return result.loc[:, ~result.columns.duplicated()]


@ray.remote
def process_label_rows(row):
    df_row = pd.DataFrame(row)
    columns = np.vectorize(str.lower)(df_row.blocklist.values)
    return pd.DataFrame(df_row.isLabeled.values.reshape(1, -1), columns=columns)


@ray.remote
def concat_splits(row):
    return pd.concat(row, ignore_index=True)


# https://datagy.io/python-split-list-into-chunks/
def parse_response_headers(chunklist, n_chunks):
    chunked_list = [chunklist[i:i + n_chunks] for i in range(0, len(chunklist), n_chunks)]
    chunked_list2 = ray.get([concat_splits.remote(row) for row in chunked_list])
    return pd.concat(chunked_list2, ignore_index=True)


def split_data_into_headers(data):
    result = pd.json_normalize(data.response)
    result.drop(['documentId', 'documentLifecycle', 'frameId', 'timeStamp', 'parentDocumentId'], axis=1,
                inplace=True)
    return result


def split_data_into_labels(data):
    return pd.json_normalize(data.labels)


def prepare_initial_dataset(path, http_message):
    return read_json_file(path)[[f'{http_message}', 'labels']].dropna().reset_index(drop=True)


def parse_dataset(path, http_message, file_name, n_chunks):
    print(f"Prepare initial dataset:"
          f"Path: {path}, HTTP Type: {http_message}, Chunksize: {n_chunks}"
          f"Filename: {file_name}")

    response_data = prepare_initial_dataset(f'{path}', f'{http_message}')
    response_data_messages = split_data_into_headers(response_data[['response']])

    print(f"Parse HTTP {http_message} headers")
    parsed_headers = ray.get([process_rows.remote(i) for i in response_data_messages['responseHeaders']])
    final_response_headers = parse_response_headers(parsed_headers, n_chunks)

    print(f"Parse HTTP {http_message} Labels")
    parsed_labels = ray.get([process_label_rows.remote(row) for row in response_data[['labels']].labels])

    print('FOURTH PHASE')
    final_response_labels = pd.concat(parsed_labels, ignore_index=True)

    result = pd.concat([response_data_messages, final_response_labels, final_response_headers], axis=1) \
        .drop(['responseHeaders'], axis=1)
    write_to_parquet_file(result, f'{file_name}')
    print("End")


if __name__ == '__main__':
    #ray.shutdown()
    #ray.init()
    pd.set_option('display.max_columns', 500)

    # parse_dataset('json_data/test2-http.0.json', 'response', 'test4', 2000)

    # Time: 736.9540843749999 (12.4min), 536.0855764169996 (9min), 143.34907520900015 (2.4min)

    # write_to_parquet_file(res, 'test3')

    test_parquet = read_parquet_file('test4')

    print(test_parquet.info())
    print(test_parquet.memory_usage())
    print(test_parquet.nunique())

    #print(combine_datasets(['test','test2','test3']).info())

    # ddf = dd.from_pandas(result_dfs, n_partitions=32)
    # dfs2 = db.from_sequence([dd.from_pandas(x, npartitions=1) for x in dfs])
    # print(dd.concat(dfs).compute())
    # print(pd.concat(dfs))

    # test = ray.get(
    #    [process_label_rows.remote(pd.json_normalize(response_data_labels.values[np.arange(len(response_data_labels)),][i]))
    #     for i in np.arange(len(response_data_labels))])

    # print(response_data.isnull().sum())
    # Time: 0.001626124999802414
    # le = response_data[response_data['response'].isnull()]
