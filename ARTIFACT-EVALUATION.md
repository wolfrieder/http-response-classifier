# Artifact Appendix

Paper title: **Beyond the Request: Harnessing HTTP Response Headers for
Cross-Browser Web Tracker Classification in an Imbalanced Setting**

Artifacts HotCRP Id: **#17**

[//]: # (Requested Badge: Either **Available**, **Functional**, or **Reproduced**)

Requested Badge: **Available**

## Description

[//]: # (A short description of your artifact and how it links to your paper.)
This artifact represents our semi-automated machine learning (ML) pipeline of
our paper. It comprises the stages: (i) Initial Parsing, (ii) Data Preparation,
(iii) Data Processing, (iv) Classification, and (v) Calibration -- described in
Section 5.2 (page 7) and Figure 4 (page 8). Our models are able to detect trackers 
purely from a binarized subset of HTTP response headers. 

Our artifact allows:
- Basic data exploration of HTTP data to understand key characteristics of
the datasets.
- Classification of HTTP responses and requests as tracker or non-trackers.
- Evaluation of our classifiers across several metrics. 
- Clustering of HTTP response messages to discern similarities and differences \
across trackers and non-trackers.
- The reproduction of the results and analysis from our paper.  


### Security/Privacy Issues and Ethical Concerns (All badges)

[//]: # (If your artifact holds any risk to the security or privacy of the reviewer's)

[//]: # (machine, specify them here, e.g., if your artifact requires a specific security)

[//]: # (mechanism, like the firewall, ASLR, or another thing, to be disabled for its)

[//]: # (execution.)

[//]: # (Also, emphasize if your artifact contains malware samples, or something similar,)

[//]: # (to be analyzed.)

[//]: # (In addition, you should highlight any ethical concerns regarding your artifacts)

[//]: # (here.)

Our artifact does not require any security changes for its installation or 
execution. It does not contain any malware samples or something similar. 
The datasets were collected by T.EX in an automated fashion, i.e., data from a 
real user was not collected. 

The  currently potential known security risks are the ones reported by the Dependabot 
from our GitHub repository. These could be fixed by updating to newer versions
of certain packages. Potential risks pertain to:
- setuptools (vulnerable to Command Injection via package URL and
vulnerable to Regular Expression Denial of Service (ReDoS))
- scikit-learn (sensitive data leakage vulnerability) -> for TF-IDF, not used here
- black (vulnerable to Regular Expression Denial of Service (ReDoS))

## Basic Requirements (Only for Functional and Reproduced badges)

[//]: # (Describe the minimal hardware and software requirements of your artifact and)

[//]: # (estimate the compute time and storage required to run the artifact.)

### Hardware Requirements

[//]: # (If your artifact requires specific hardware to be executed, mention that here.)

[//]: # (Provide instructions on how a reviewer can gain access to that hardware through)

[//]: # (remote access, buying or renting, or even emulating the hardware.)

[//]: # (Make sure to preserve the anonymity of the reviewer at any time.)

**Copied from our README.md:**
The code was only tested on an MacBook Pro with the 10-core Apple M1 Pro (ARM
CPU), 32GB
memory, and 1TB storage. It is recommended to have at least 32GB memory as tests
with
16GB memory led to memory issues while training the classifiers. Therefore, the
minimal requirement are 32GB of memory until further tests and optimizations are
performed. 

### Software Requirements

[//]: # (Describe the OS and software packages required to evaluate your artifact.)

[//]: # (This description is essential if you rely on proprietary software or software)

[//]: # (that might not be easily accessible for other reasons.)

[//]: # (Describe how the reviewer can obtain and install all third-party software, data)

[//]: # (sets, and models.)

We used PyCharm 2024 and Python 3.10.11. 
We have primarily tested and developed our artifact on macOS (Sonoma and Sequoia).
For the review process we executed the artifact on a Compute VM running Ubuntu 22.04.
(Python 3.10.11, venv, and pip are needed)

The LightGBM installation might throw an error when installed on an Apple
Silicon
Macbook with `pip`. One solution was proposed
[here](https://stackoverflow.com/questions/74566704/cannot-install-lightgbm-3-3-3-on-apple-silicon)
and requires the installation of two more dependencies
via `brew install cmake libomp`.
An installation via `pip install lightgbm` will then work correctly. 

### Estimated Time and Storage Consumption

[//]: # (Provide an estimated value for the time the evaluation will take and the space)

[//]: # (on the disk it will consume.)

[//]: # (This helps reviewers to schedule the evaluation in their time plan and to see if)

[//]: # (everything is running as intended.)

[//]: # (More specifically, a reviewer, who knows that the evaluation might take 10)

[//]: # (hours, does not expect an error if, after 1 hour, the computer is still)

[//]: # (calculating things.)

Storage: around 10GB, however, if we include the crawled raw datasets than this 
number increases by around 16GB.  

Time: around 3h for executing the pipeline -- jupyter notebooks for the figures
takes ca. 1h - 2h (clustering takes the most time, followed by calibration)

**Copied from our README.md:** 
For orientation purposes, we will present the execution time for each stage 
as defined in our `dvc.yaml` file (pipeline). This does not include the times 
of our additional experiment that we added after the acceptance decision for our 
camera-ready version -- but the times are similar to the ones from the response
stages. 

1. merge_response: ~ 20.46 min
2. merge_request: ~ 15.18 min
3. parse_to_parquet_response: ~ 37.43 min
4. parse_to_parquet_request: ~ 8.46 min
5. train_test_split_response: ~ 7.30 min
6. train_test_split_request: ~ 1.42 min
7. pre_processing_train_set_response: ~ 8:26 min
8. pre_processing_train_set_request: ~ 1:08 min
9. pre_processing_other_sets_response: ~ 12:14 min
10. pre_processing_other_sets_request: ~ 2:19 min
11. feature_engineering_response: ~ 0:23 min
12. feature_engineering_request: ~ 0:12 min
13. model_training_response: ~ 46:10 min
14. model_training_request: ~ 5:07 min
15. model_evaluation_response: ~ 9:56 min

## Environment

In the following, describe how to access our artifact and all related and
necessary data and software components.
Afterward, describe how to set up everything and how to verify that everything
is set up correctly.

### Accessibility (All badges)

[//]: # (Describe how to access your artifact via persistent sources.)

[//]: # (Valid hosting options are institutional and third-party digital repositories.)

[//]: # (Do not use personal web pages.)

[//]: # (For repositories that evolve over time &#40;e.g., Git Repositories &#41;, specify a)

[//]: # (specific commit-id or tag to be evaluated.)

[//]: # (In case your repository changes during the evaluation to address the reviewer's)

[//]: # (feedback, please provide an updated link &#40;or commit-id / tag&#41; in a comment. )

Link to our GitHub repository: https://github.com/wolfrieder/http-response-classifier

Link to the two datasets: 
1. [Chrome, Firefox, and Brave Data (2022)](https://zenodo.org/record/7123945#.Y8VDEXaZPtU) --
   Size: 14.3GB (Zipped) 
2. [Chrome Data (2023)](https://zenodo.org/records/11555919) -- Size: 1.1GB (
   Zipped) -> newly crawled dataset for longitudinal analysis. 

The crawler is named T.EX [1] and is not part of our contribution. The datasets were
uploaded to Zenodo.

Link to T.EX: https://github.com/t-ex-tools/t.ex
(last commit: c547992) -- Citation:

[1] Raschke, P., Zickau, S., Kröger, J.L., Küpper, A. (2019). Towards Real-Time Web 
Tracking Detection with T.EX - The Transparency EXtension. In: Naldi, M., 
Italiano, G., Rannenberg, K., Medina, M., Bourka, A. (eds) Privacy Technologies 
and Policy. APF 2019. Lecture Notes in Computer Science(), vol 11498. Springer, 
Cham. https://doi.org/10.1007/978-3-030-21752-5_1 

**Important:** We followed the same methodology to crawl the new dataset as 
outlined by Raschke and Cory [2] -- the steps to perform a crawl and to load an 
existing dataset are described on their GitHub repository and paper. 
However, Zenodo represents the raw datasets from a crawl which are not in the 
JSON format. The Zenodo datasets have to be injected into T.EX (if one wants to 
export a dataset that was not crawled by ones machine) and then exported within
the tool to the JSON format which is part of this repository. We performed the 
latter step for the reviewers to save them time and to immediately start with 
the review. (Otherwise the reviewers would have to download T.EX, install it on 
a browser and use the browser extension to export the Zenodo datasets).  

[2] P. Raschke and T. Cory, "Presenting a Client-based Cross-browser Web Privacy 
Measurement Framework for Automated Web Tracker Detection Research," 
2022 3rd International Conference on Electrical Engineering and Informatics 
(ICon EEI), Pekanbaru, Indonesia, 2022, pp. 98-103, 
doi: 10.1109/IConEEI55709.2022.9972261.

### Set up the environment (Only for Functional and Reproduced badges)

Describe how the reviewers should set up the environment for your artifact,
including downloading and installing dependencies and the installation of the
artifact itself.
Be as specific as possible here.
If possible, use code segments to simply the workflow, e.g.,

```bash
git clone git@github.com:wolfrieder/http-response-classifier.git
```
or
```bash
git clone https://github.com/wolfrieder/http-response-classifier.git
```

if python not installed 
```bash
sudo apt install python3.11.10
```

if pip not installed:
```bash
sudo apt install python3-pip
```


Describe the expected results where it makes sense to do so.

### Testing the Environment (Only for Functional and Reproduced badges)

Describe the basic functionality tests to check if the environment is set up
correctly.
These tests could be unit tests, training an ML model on very low training data,
etc..
If these tests succeed, all required software should be functioning correctly.
Include the expected output for unambiguous outputs of tests.
Use code segments to simplify the workflow, e.g.,

```bash
python envtest.py
```

## Artifact Evaluation (Only for Functional and Reproduced badges)

This section includes all the steps required to evaluate your artifact's
functionality and validate your paper's key results and claims.
Therefore, highlight your paper's main results and claims in the first
subsection. And describe the experiments that support your claims in the
subsection after that.

### Main Results and Claims

List all your paper's results and claims that are supported by your submitted
artifacts.

#### Main Result 1: Name

Describe the results in 1 to 3 sentences.
Refer to the related sections in your paper and reference the experiments that
support this result/claim.

#### Main Result 2: Name

...

### Experiments

List each experiment the reviewer has to execute. Describe:

- How to execute it in detailed steps.
- What the expected result is.
- How long it takes and how much space it consumes on disk. (approximately)
- Which claim and results does it support, and how.

#### Experiment 1: Name

Provide a short explanation of the experiment and expected results.
Describe thoroughly the steps to perform the experiment and to collect and
organize the results as expected from your paper.
Use code segments to support the reviewers, e.g.,

```bash
python experiment_1.py
```

#### Experiment 2: Name

...

#### Experiment 3: Name

...

## Limitations (Only for Functional and Reproduced badges)

Describe which tables and results are included or are not reproducible with the
provided artifact.
Provide an argument why this is not included/possible.

## Notes on Reusability (Only for Functional and Reproduced badges)

First, this section might not apply to your artifacts.
Use it to share information on how your artifact can be used beyond your
research paper, e.g., as a general framework.
The overall goal of artifact evaluation is not only to reproduce and verify your
research but also to help other researchers to re-use and improve on your
artifacts.
Please describe how your artifacts can be adapted to other settings, e.g., more
input dimensions, other datasets, and other behavior, through replacing
individual modules and functionality or running more iterations of a specific
part.