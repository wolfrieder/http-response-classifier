# Beyond the Request: Harnessing HTTP Response Headers for Cross-Browser Web Tracker Detection in an Imbalanced Setting

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/wolfrieder/thesis_project_v2/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

The World Wide Web’s connectivity is greatly attributed to the HTTP protocol,
with HTTP messages offering informative header fields that appeal to disciplines
like web security and privacy, especially concerning web tracking. Despite
existing research employing HTTP request messages to identify web trackers,
HTTP response headers are often overlooked. This study endeavors to design
effective machine learning classifiers for web tracker detection using
binarized HTTP response headers. Data from the Chrome, Firefox, and Brave
browsers, obtained through the traffic monitoring browser extension T.EX, serves
as our dataset. Ten supervised models were trained on Chrome data and tested
across all browsers, including a Chrome dataset from a year later. The results
demonstrated high accuracy, F1-score, precision, recall, and minimal log-loss
error for Chrome and Firefox, but subpar performance on Brave, potentially due
to its distinct data distribution and feature set. The research suggests that
these classifiers are viable for web tracker detection. However, real-world
application testing remains pending, and the distinction between tracker types
and broader label sources could be explored in future studies.

## Datasets

The data acquisition is not implemented in this repository but an existing
solution,
namely [T.EX - The Transparency EXtension](https://github.com/t-ex-tools/t.ex)
was used. In addition, the code of this project follows the data schema
of `T.EX`
and can be used on exported HTTP request and response data.
Four datasets were used for this paper: 

1. [Chrome, Firefox, and Brave Data (2022)](https://zenodo.org/record/7123945#.Y8VDEXaZPtU) --
   Size: 14.3GB (Zipped)
2. [Chrome Data (2023)](https://zenodo.org/records/11555919) -- Size: 1.1GB (
   Zipped)

[//]: # (The list of crawled websites can be found in the `tranco_list_08_12_2022.txt` file.)

## Installation Guide

To replicate the environment and set up the necessary tools for this project,
follow the step-by-step instructions outlined below:

1. Ensure that Python 3.10.11 is installed on your system (this was the used
   Python version, other version might work as well). If you do not have
   Python 3.10.11, download and install it
   from the official [Python website](https://www.python.org/downloads/),
   [asdf](https://asdf-vm.com), or another tool of your choice. Python version
   3.9.12 was also tested and works.
   However, Python version 3.10.11 is recommended due to performance
   improvements.


2. Install the [Conda](https://docs.conda.io/en/latest/miniconda.html) package
   manager, which can be utilized for
   managing dependencies and creating a virtual environment or use your IDE (
   e.g. PyCharm for venv) and PIP.


3. Create a new virtual environment with Python 3.10.11.


4. Clone the GitHub repository to your local machine (HTTP/S):
   ```
   git clone https://github.com/wolfrieder/http-response-classifier.git
   ```


5. Navigate to the cloned repository's root directory:
   ```
   cd http-response-classifier
   ```

6. Install the necessary packages from the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

7. Download the datasets (links are provided under the point `Datasets`) and
   extract the files. Follow
   the [T.EX instructions](https://github.com/t-ex-tools/t.ex?tab=readme-ov-file) (
   _Inject an existing T.EX state to fresh installation_) to download the JSON
   files (We used the Chrome browser for this step).


8. Move the JSON files to the respective `data/raw/{browser}/{date}`
   directories.

With these steps completed, you should now have a functional environment as well
as
the necessary tools for efficient model development and tracking.

## Recommended Execution

1. We use `DVC` to create a reproducible pipeline, i.e., we defined our whole
   pipeline and each step in a YAML file (`dvc.yaml`), including any
   parameters (`params.yaml`).
   To execute the pipeline, navigate to `data/raw/chrome/08_12_2022` and enter
   the following command in the terminal.

```
dvc repro
``` 

2. We recommend executing each pipeline step individually in case there are any
   problems (e.g. not enough resources which would terminate the execution)
   using the `frozen: true` attribute in the `dvc.yaml` file. True in this case
   means that the step is not executed and false or simply commenting the line
   out would execute the step. The advantage is that you do not have to repeat
   each step in case of an error (e.g., there were problems during training at
   the end, but because of the error, you would have to repeat the rest).


3. We have only tested our code on the following setup and macOS Sonoma as well
   as macOS Ventura (last version). Due to the use of an ARM CPU, the code was
   not optimized for x86 CPUs and the general available GPUs (especially
   NVIDIA). Nevertheless, we did not use any Apple Silicon specific libraries,
   so the code should work for other systems as well.

## Computer Setup

The code was only tested on an MacBook Pro with the 10-core Apple M1 Pro (ARM
CPU), 32GB
memory, and 1TB storage. It is recommended to have at least 32GB memory as tests
with
16GB memory led to memory issues while training the classifiers. Therefore, the
current requirement are 32GB of memory until further tests and optimizations are
performed.

## Known Installation Problems

If the LightGBM import throws this error message:
`AttributeError: module 'pandas.core.strings' has no attribute 'StringMethods'`
then you should update `dask` and `dependencies`. The problem is described
[here](https://github.com/microsoft/LightGBM/issues/5739) and is related to the
recent release of `pandas` version 2. -> This should not be a problem anymore as
we updated the `requirements.txt` with the new version.

The LightGBM installation might throw an error when installed on an Apple
Silicon
Macbook with `pip`. One solution was proposed
[here](https://stackoverflow.com/questions/74566704/cannot-install-lightgbm-3-3-3-on-apple-silicon)
and requires the installation of two more dependencies
via `brew install cmake libomp`.
An installation via `pip install lightgbm` will then work correctly. 