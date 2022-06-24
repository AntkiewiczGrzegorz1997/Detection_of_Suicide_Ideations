# Detection of Suicide Ideation on Online Forums Using Deep Learning and PsychBERT: A Pre-trained Language Model for Mental Health-related Online Posts

Master's Thesis

Author: Grzegorz Antkiewicz

Supervisor: Prof. Dr. Stefan Lessmann

Date: 28.06.2022

## Usage

The project is conducted in 2 different environments: local machine and Google Colab Pro+. The following scripts are run in the local machine: main.py, scraping.py, data_loader.py, preprocess_data.py, SMHD_preprocess.py and finetune_mental_health.py 

So that the project could be reproduced the order of running the scripts should be kept as follows:

1. To gather the data - run the scraping.py from the main.py for getting the data that will be used to train PsychBERTModel. The scraper saves the scraped content of each subreddit into a separate file. Personal credentials for running Pushlift's API are needed as well. Big part of the Data is also obtained from Google Big Query and can be later preprocessed by the script. 
2. Merge the Data - the main.py runs the data_loader.py script. It should be run twice: once for our scraped data and the second time for the data from Google Big Query. 
3. Preprocess the files - from main.py run the preprocess_data.py file that preprocesses the text and outputs a .txt file. Because of the RAM constraints it has to be done probably few times, each time on a different part of the dataset resulting in the end in few preprocessed files. 

4. With the help of Preprocess_Dataset_Right.ipynb preprocess .txt files into their final correct format that can be later used for pre-training BERT.

5. Run BERT_Pre-Training.ipynb twice: for PsychBERT1 and PsychBERT2. In the script the right datasets should be chosen.
6. For SMHD and RSDD datasets there is an addictional script to extract csv data from a specific format: SMHD_preprocess.py
7. For BERT, PsychBERT1 and PsychBERT2 extract embeddings with Embeddings.ipynb and save them in the repository. The Embeddings.ipynb as well preprocesses a bit all of the datasets
8. Run finetune_mental_health.py for each dataset. (the finetuned-BERT is needed for later)
9. Run FirstClassification.ipynb to run test a performance each of the classification model on the top of each embedding.

10. To detect the novelty data points run the NoveltyDetection.ipynb script which outputs the list with novelty indices which are going to be excluded later. 
11. To run the classification without the rows classified as novelty run FirstClassification.ipynb with setting "novelty_detection" to True and run de models as in the step 9. 

## Data availability 

For the access to the Data scraped from Reddit and used for pre-training PsychBERT please contact Grzegorz Antkiewicz (gantkiewicz97@gmail.com).
The SMHD, RSDD datasets used in the study are highly confidential and cannot be shared. The Aladag Dataset can by acquiered by contacting the author of the paper https://www.jmir.org/2018/6/e215/.  The RCSD dataset can be downloaded at https://zenodo.org/record/2667859#.YrXLZuxBxhE. 

For the purpose of providing a dataset for simulation, the dataset_simulation.csv is created. 


## Requirements

Because the RAM constraints of the normal Google Colab most of the ipynb notebooks will be run in Google Colab Pro+ which offers up to 50GB of RAM. 

The following packages requirements are only applicable to the codes that are run in python locally, not in Google Colab. 


The version of python that is required to run the code locally is python==3.8.
```bash
aiohttp==3.8.1
aiosignal==1.2.0
anyascii==0.3.0
appnope==0.1.3
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
asttokens==2.0.5
async-timeout==4.0.2
attrs==21.4.0
backcall==0.2.0
beautifulsoup4==4.10.0
bleach==5.0.0
bs4==0.0.1
certifi==2021.10.8
cffi==1.15.0
charset-normalizer==2.0.10
click==8.0.3
contractions==0.1.66
datasets==1.18.3
debugpy==1.6.0
decorator==5.1.1
defusedxml==0.7.1
dill==0.3.4
emoji==1.6.3
emot==2.1
entrypoints==0.4
et-xmlfile==1.1.0
executing==0.8.3
fastjsonschema==2.15.3
filelock==3.6.0
frozenlist==1.3.0
fsspec==2022.1.0
huggingface-hub==0.4.0
idna==3.3
importlib-resources==5.7.1
inflect==5.4.0
install==1.3.5
ipykernel==6.13.0
ipython==8.2.0
ipython-genutils==0.2.0
ipywidgets==7.7.0
jedi==0.18.1
Jinja2==3.1.1
joblib==1.1.0
json-lines==0.5.0
jsonschema==4.4.0
jupyter-client==7.3.0
jupyter-core==4.10.0
jupyterlab-pygments==0.2.2
jupyterlab-widgets==1.1.0
MarkupSafe==2.1.1
matplotlib-inline==0.1.3
mistune==0.8.4
multidict==6.0.2
multiprocess==0.70.12.2
nbclient==0.6.0
nbconvert==6.5.0
nbformat==5.3.0
nest-asyncio==1.5.5
nltk==3.7
notebook==6.4.11
numpy==1.22.0
openpyxl==3.0.9
packaging==21.3
pandas==1.3.5
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
praw==7.5.0
prawcore==2.3.0
prometheus-client==0.14.1
prompt-toolkit==3.0.29
psaw==0.1.0
psutil==5.9.0
ptyprocess==0.7.0
pure-eval==0.2.2
pyahocorasick==1.4.4
pyarrow==7.0.0
pycparser==2.21
Pygments==2.12.0
pyparsing==3.0.7
pyrsistent==0.18.1
python-dateutil==2.8.2
pytz==2021.3
PyYAML==6.0
pyzmq==22.3.0
regex==2022.1.18
requests==2.27.1
sacremoses==0.0.47
scikit-learn==1.0.2
scipy==1.8.0
Send2Trash==1.8.0
sentencepiece==0.1.96
six==1.16.0
sklearn==0.0
snowballstemmer==2.2.0
soupsieve==2.3.1
stack-data==0.2.0
terminado==0.13.3
textsearch==0.0.21
threadpoolctl==3.1.0
tinycss2==1.1.1
tokenizers==0.10.3
torch==1.11.0
tornado==6.1
tqdm==4.62.3
traitlets==5.1.1
transformers==4.11.2
typing_extensions==4.1.1
update-checker==0.18.0
urllib3==1.26.8
wcwidth==0.2.5
webencodings==0.5.1
websocket-client==1.2.3
widgetsnbextension==3.6.0
xxhash==2.0.2
yarl==1.7.2
zipp==3.8.0
```
