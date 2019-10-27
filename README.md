# LSTM GRU RF predicting SP500

This repository contains the code used for my BSC thesis with the name: [Predicting the S&P 500 with long short-term memory network, gated recurrent unit, and random forest](Till_Aczel_BSc_thesis.pdf). 

If you want to cite my thesis, please use the following citation:
```
@Thesis{aczel2019predicting,
  Title   = {Predicting the S&P 500 with long short-term memory network, gated recurrent unit, and random forest},
  Author  = {Aczél, Till},
  School  = {University of Corvinus},
  Year    = {2019},
  Type    = {Bachelor thesis},
  Url     = {https://github.com/tillaczel/LSTM-GRU-RF-predicting-SP500/blob/master/Till_Aczel_BSc_thesis.pdf},
}
```

## Prerequisites
I run the codes in a python 3.6.9 enviroment with the following packages:
```
absl-py               0.8.0
arch                  4.9.1
astor                 0.8.0
attrs                 19.1.0
backcall              0.1.0
bayesian-optimization 1.0.1
bleach                3.1.0
certifi               2019.9.11
colorama              0.4.1
cycler                0.10.0
Cython                0.29.13
decorator             4.4.0
defusedxml            0.6.0
entrypoints           0.3
gast                  0.3.2
google-pasta          0.1.7
grpcio                1.24.0
h5py                  2.10.0
ipykernel             5.1.2
ipython               7.8.0
ipython-genutils      0.2.0
ipywidgets            7.5.1
jedi                  0.15.1
Jinja2                2.10.1
joblib                0.13.2
jsonschema            3.0.2
jupyter               1.0.0
jupyter-client        5.3.3
jupyter-console       6.0.0
jupyter-core          4.5.0
Keras-Applications    1.0.8
Keras-Preprocessing   1.1.0
kiwisolver            1.1.0
Markdown              3.1.1
MarkupSafe            1.1.1
matplotlib            3.1.1
mistune               0.8.4
nbconvert             5.6.0
nbformat              4.4.0
notebook              6.0.1
numpy                 1.17.2
opt-einsum            3.0.1
pandas                0.25.1
pandocfilters         1.4.2
parso                 0.5.1
patsy                 0.5.1
pickleshare           0.7.5
pip                   19.2.3
pmdarima              1.3.0
prometheus-client     0.7.1
prompt-toolkit        2.0.9
property-cached       1.6.3
protobuf              3.9.2
Pygments              2.4.2
pyparsing             2.4.2
pyrsistent            0.15.4
python-dateutil       2.8.0
pytz                  2019.2
pywin32               225
pywinpty              0.5.5
pyzmq                 18.1.0
qtconsole             4.5.5
scikit-learn          0.21.3
scipy                 1.3.1
Send2Trash            1.5.0
setuptools            41.2.0
six                   1.12.0
statsmodels           0.10.1
tb-nightly            1.15.0a20190806
tensorflow            2.0.0rc0
termcolor             1.1.0
terminado             0.8.2
testpath              0.4.2
tf-estimator-nightly  1.14.0.dev2019080601
tornado               6.0.3
traitlets             4.3.2
wcwidth               0.1.7
webencodings          0.5.1
Werkzeug              0.16.0
wheel                 0.33.6
widgetsnbextension    3.5.1
wincertstore          0.2
wrapt                 1.11.2
```

## Running the codes

In my thesis, I analyzed the ARMA, LSTM, GRU, RF and ENS models on the minute by minute S&P 500 index dataset from 2008 to 2017. Because this dataset is not public, I also run the models on a publicly available Bitcoin dataset. The [Bitcoin dataset](https://www.kaggle.com/mczielinski/bitcoin-historical-data) was downloaded from Kaggle. The S&P 500 and Bitcoin experiments can be found in separate folders.

For training, run the [train.ipynb](train.ipynb) and for evaluating run the [evaluate.ipynb](evaluate.ipynb) notebook.
