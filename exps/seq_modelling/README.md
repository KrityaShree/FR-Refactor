# d-BERT

# What is d-BERT?

d-BERT is our approach to improving up on the authors' work by including contextual information in the learning pipeline. We provide two types of training schedules: the standard (textual) pretraining and the code-augmented pretraining. We provide installation and running instructions for both of them below.


Note: If you are testing this on your own machine we would recommend setting this up in a virtual environment, so that it does not affect the rest of your files.

## Setup

### Setting Up a Virtual Environment

We strongly recommend setting up a virtual environment to avoid clashing with your current
environment and making setup easier. The following instructions are for Anaconda users, so if
you are not using Anaconda, try making a virtual environment in [Windows](#Windows_Env) or
[MacOS & Linux](#Mac_Linux_Env).

If using Anaconda, update your environment before installing dependencies with:

    $ conda update conda
    $ conda update --all

Check your version of Python by typing the following in terminal for MacOS & Linux and
Anaconda Prompt (in the Windows Start Menu) for Windows:

	$ python --version

Navigate to this directory in Terminal or Anaconda Prompt and create an environment 
(called env) and activate it:

	$ conda create -n env python=x.x anaconda
	$ conda activate env

Replace x.x with your Python version number.

Press y to proceed.

To leave the environment after you no longer need it:

	$ conda deactivate

After you have created your virtual environment, install the [dependencies](#Dependencies).

#### Windows Users

Without Anaconda, you can get up a virtual environment in the code directory in Command 
Prompt or Powershell like this:

    $ py -m pip install --upgrade pip
    $ py -m pip install --user virtualenv
    $ py -m venv env

To activate the virtual environment:

    $ .\env\Scripts\activate

To confirm you're in the virtual environment, check the location of your Python interpreter.

    $ where python
    .../env/bin/python.exe

To leave the environment after you no longer need it:

	$ deactivate

#### Mac/Linux Users

Without Anaconda, you can get up a virtual environment in the code directory in terminal 
like this:

    $ python3 -m pip install --user --upgrade pip
    $ python3 -m pip install --user virtualenv
    $ python3 -m venv env

To activate the virtual environment:

    $ source env/bin/activate

To confirm you're in the virtual environment, check the location of your Python interpreter.

    $ which python
    .../env/bin/python

To leave the environment after you no longer need it:

	$ deactivate

## Dependencies

All of the dependencies are listed in `requirements.txt`. To reduce the likelihood of environment
errors, install the dependencies inside a virtual environment with the following steps.

Navigate to this directory and activate the virtual environment in your terminal.
Run the following commands in terminal:

	$ pip install -r requirements.txt



Once you have setup your environment and installed all the required libraries, you will need to initialize the bert-uncased model in the Transformers library.

You can do this with

```python
model = d-BERT.from_pretrained('d-bert-bert-base-uncased-debug', config=config)
```
Currently, ```model.txt``` has a path for a code-augmented d-BERT model hosted on SageMaker.


The config file (config.json) is stored in this directory. Please move it accordingly to your prefered location.

# How to run d-BERT

You will need to specify the directory in which the FR-Refactor as ```DATA_DIR``` dataset is stored on your system. The dataset can be found one level above in the __data__ directory.

To run the standard (textual) pretraining schedule:

```bash
export DATA_DIR=/path/to/dataset

python run_standard.py \
  --model_type d-bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --train_file $DATA_DIR/train_standard.json \
  --predict_file $DATA_DIR/preds_standard_.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 1e-2 \
  --num_train_epochs 20 \
  --max_seq_length 384 \
  --output_dir /tmp/debug/standard/
  ```

To run the code-augmented pretraining schedule:

```bash
export DATA_DIR=/path/to/dataset

python run_code_augmented.py \
  --model_type d-bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --train_file $DATA_DIR/train-v1.1.json \
  --predict_file $DATA_DIR/preds-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 20 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug/code-augmented/
  ```

  # Making predictions

  The way the code is set-up, running the training script should automatically generate predictions and metrics on an unseen subset of the dataset. However, if you want to run predictions on an entirely different dataset, please follow the instructions below:

  Once you have your pretrained model, please change the path of the model in ```model.txt``` and run the following command:
  ```
  python -m make-preds.py
  ```

  Your predictions will be stored in a separate directory that will be created as the script is run.


