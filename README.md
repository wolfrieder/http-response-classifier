# Machine Learning-Driven Web Privacy Analysis Using HTTP(S) Response Headers

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/wolfrieder/thesis_project_v2/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DeepSource](https://deepsource.io/gh/wolfrieder/thesis_project_v2.svg/?label=active+issues&show_trend=true&token=bI2KtH2zJhdmo15pPCscudQ9)](https://deepsource.io/gh/wolfrieder/thesis_project_v2/?ref=repository-badge)
## Introduction

This research project constitutes a significant component of my Master's thesis, 
which spans the domains of web privacy, data science, and machine learning. 
The primary objective of this endeavor is to devise and evaluate a series of machine
learning models that leverage HTTP(S) response headers and associated metadata. 
To achieve this goal, the project follows a comprehensive methodology comprising 
six major stages: (i) data acquisition, (ii) data exploration, (iii) feature 
engineering, (iv) training and testing of machine learning models, (v) hyperparameter
optimization, and (vi) model deployment. Furthermore, this project incorporates 
MLOps best practices in order to facilitate efficient model development and tracking.

## Installation Guide

To replicate the environment and set up the necessary tools for this project, 
follow the step-by-step instructions outlined below:

1. Ensure that Python 3.9.12 is installed on your system. If you do not have Python 3.9.12, download and install it
   from the official [Python website](https://www.python.org/downloads/) or use
   [asdf](https://asdf-vm.com). 

2. Install the [Conda](https://docs.conda.io/en/latest/miniconda.html) package manager, which will be utilized for
   managing dependencies and creating a virtual environment.

3. Create a new Conda environment with Python 3.9.12:
   ```
   conda create -n my_env python=3.9.12
   ```

4. Activate the newly created Conda environment:
   ```
   conda activate my_env
   ```

5. Clone the GitHub repository to your local machine:
   ```
   git clone https://github.com/wolfrieder/thesis_project_v2.git
   ```

6. Navigate to the cloned repository's root directory:
   ```
   cd thesis_project_v2
   ```

7. Install the necessary packages from the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

8. Install and set up [DVC](https://dvc.org/doc/install) for version control of your data and models:
   ```
   dvc init
   ```

9. Install and set up [MLflow](https://mlflow.org/docs/latest/quickstart.html) for experiment tracking and model management:
   ```
   TEST
   ```

With these steps completed, you should now have a functional environment for executing the research project, as well as 
the necessary tools for efficient model development and tracking.
