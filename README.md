
# A*NN*aMorphose
Neural Network models for Analogy in Morphology (A*NN*aMorphose) is an approach using deep learning models for the identification and inference of analogies in morphology.
The approach presented here is described in the article "Tackling Morphological Analogies Using Deep Learning" (Anonymous, 2022).

To cite this repository, use the following reference:
```bib
@article{annonymous,
    author = {To be added if the article is accepted},
}
```

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Install Instructions](#install-instructions)
  - [Cloning the Repository](#cloning-the-repository)
  - [Installing the Dependencies](#installing-the-dependencies)
    - [Singularity](#singularity)
    - [Anaconda](#anaconda)
  - [Setting up the data (Sigmorphon2016 and Sigmorphon2019)](#setting-up-the-data-sigmorphon2016-and-sigmorphon2019)
    - [Setting-Up the Japanese Data for Sigmorphon2016](#setting-up-the-japanese-data-for-sigmorphon2016)
  - [*[Optional]* Installing the EMbedding Models for the Preliminary Experiments](#optional-installing-the-embedding-models-for-the-preliminary-experiments)
    - [GloVe](#glove)
    - [Fasttext](#fasttext)
    - [word2vec](#word2vec)
- [General Usage](#general-usage)
- [Reproducing Experiments](#reproducing-experiments)
  - [Running the Symbolic Baselines](#running-the-symbolic-baselines)
  - [Running 3CosMul and 3CosAdd Baselines](#running-3cosmul-and-3cosadd-baselines)
  - [Classification Model ANNc](#classification-model-annc)
  - [Regression Model ANNr](#regression-model-annr)
- [Files and Folders TO RE-CHECK](#files-and-folders-to-re-check)

## Install Instructions
The following installation instruction are designed for command line on Unix systems. Refer to the instructions for Git and Anaconda on your exploitation system for the corresponding instructions.

### Cloning the Repository
Clone the repository on your local machine, using the following command:

```bash
git clone https://github.com/AmandineDecker/nn-morpho-analogy.git
git submodule update --init siganalogies
```

### Installing the Dependencies
YOu can use either Singularity or 
#### Singularity
1. Install [Singularity](https://sylabs.io/guides/3.0/user-guide/installation.html)
2. Build the container by running `singularity build cuda.sif cuda.def`. Issues were reported when building the container from a non-CUDA equipped machine to run on a CUDA equiped machine.
3. Run scripts with `singularity run --nv cuda.sif python` (without `--nv` if you do not have GPU) instead of `python`.
#### Anaconda
1.  Install [Anaconda](https://www.anaconda.com/) (or miniconda to save storage space).
2.  Then, create a conda environement (for example `morpho-analogy`) and install the dependencies, using the following commands:
    ```bash
    conda env create --name morpho-analogy python=3.9 -f environment.yml
    conda install --name morpho-analogy numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six dataclasses
    ```
3.  Use one of the following, depending on whether you have a GPU available:
    ```bash
    # cuda 11.3
    conda install --name morpho-analogy pytorch cudatoolkit=11.3 -c pytorch
    # cpu
    conda install --name morpho-analogy pytorch cpuonly -c pytorch
    ```
4.  Finally, install the other required libraries:
    ```bash
    # extra conda libs
    conda install --name morpho-analogy -c conda-forge pytorch-lightning
    conda install --name morpho-analogy -c conda-forge pandas seaborn scikit-learn
    conda install -c conda-forge tabulate
    pip install pebble
    ```
5.  All the following commands assume that you have activated the environment you just created. This can be done with the following command (using our example `morpho-analogy`):
    ```bash
    conda activate morpho-analogy
    ```

### Setting up the data (Sigmorphon2016 and Sigmorphon2019)
To install the Siganalogies data, run at the root of the repository:
- `git submodule update --init siganalogies` for Siganalogies code;
- `git submodule update --init sigmorphon2016` for Sigmorphon 2016;
- `git submodule update --init sigmorphon2019` for Sigmorphon 2019.

#### Setting-Up the Japanese Data for Sigmorphon2016
The Japanese data is stored as a Sigmorphon2016-style data file `japanese-task1-train` at the root of the directory, and should be moved to `sigmorphon2016/data`, the Sigmorphon2016 data folder. This can be done by running the following command:

```bash
cp siganalogies/japanese-task1-train sigmorphon2016/data/
```

There is no test nor development set. For the training and evaluation, the file `japanese-task1-train` is split: 70\% of the analogies for the training and 30\% for the evaluation. The split is always the same for reproducibility, using random seed 42.

The Japanese data was extracted from the original [Japanese Bigger Analogy Test Set](https://vecto-data.s3-us-west-1.amazonaws.com/JBATS.zip).

### *[Optional]* Installing the EMbedding Models for the Preliminary Experiments
#### GloVe
To get the GloVe embedding model for German and set it up so that the Dataset class from `data.py`, run:
```bash
mkdir embeddings/glove
grep https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt embeddings/glove/vectors.txt
```
#### Fasttext
To get the *fasttext* embedding model for German and set it up so that the Dataset class from `embeddings/fasttext_.py`, run:
```bash
mkdir embeddings/fasttext
grep https://s3.eu-central-1.amazonaws.com/int-emb-fasttext-de-wiki/20180917/model.bin embeddings/fasttext/german.bin
```

#### word2vec
To get the *w2v* embedding model for German, run:
```bash
mkdir embeddings/w2v
grep https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt embeddings/w2v/german-vectors.txt
```

To set it up so that the Dataset class from `embeddings/w2v.py` can interpret it, run:
```bash
conda activate morpho-analogy
python embeddings/w2v.py
```
The above command will launch the `convert` function of `embeddings/w2v.py` to transform the *w2v* vector file into a ready-to-use PyTorch pickle.

## General Usage
For each of the experiments files, it is not necessary to fill the parameters when you run the code, default values are used. You can use the `--help` flag to print the help message and detail available arguments.

## Reproducing Experiments
This section explains how to reproduce step by step the experiments reported in the article.

### Running the Symbolic Baselines
To run the baselines, run `python baseline/run_baseline.py -d <dataset> -l <language> -m <algorithm>` (ex: `python baseline/run_baseline.py -l arabic -d 2016 -m kolmo` to run *Kolmo* on the 2016 version of Arabic).
This will output a summary in the command line interface as well as a CSV log file in the baseline folder (ex: `baseline/murena/2016/arabic`).

The available languages are in `siganalogies.config.SIG2016_LANGUAGES``siganalogies.config.SIG2019_HIGH`.

The available baselines are:
- `alea` for the approach of P. Langlais, F. Yvon, and P. Zweigenbaum *"Improvements in analogical learning: Application to translating multi-terms of the medical domain"*;
- `kolmo` or `kolmogorov` or `murena` for the approach of P.-A. Murena, M. Al-Ghossein, J.-L. Dessalles, and A. Cornuéjols *"Solving analogies on words based on minimal complexity transformation"*;
- `lepage` for the annalogy classifier of the toolset presented in R. Fam and Y. Lepage *"Tools for the production of analogical grids and a resource of n-gram analogical grids in 11 languages"*; the tools must be installed beforehand as described below.
  1. First, check that you have `swig` installed (`sudo apt install swig` on Ubuntu-based systems).
  2.  Then, run the following from the home directory to install Lepage's algorithm:
      ```bash
      cd baseline/lepage
      conda activate morpho-analogy
      pip install -e .
      ```

### Running 3CosMul and 3CosAdd Baselines
To run the baselines, run `python baseline/3cos.py -d <dataset> -l <language> -M <"3CosAdd" or "3CosMul"> -T <path to ANNc model or "auto">`

### Classification Model ANNc
To train a classifier and the corresponding embedding model for a language, run the following (all parameters are optional, shorthands can be seen with the `--help` flag of the command):

```
python train_clf.py -l <language> -d <dataset> -n <number of analogies in training set> -v <number of analogies in validation/development set> -t <number of analogies in test set>  --max_epochs <number of training epochs>
```
Examples: 
- `python train_clf.py -l arabic` to train on Arabic (by default on Sigmorphon 2016), using up to 50000 analogies (default) for 20 epochs (default);
- `python train_clf.py -l german -d 2019 -n 50000 --max_epochs 20` to train on German of Sigmorphon 2019, using up to 50000 analogies for 20 epochs.

### Regression Model ANNr
To train a regression model and the corresponding embedding model for a language, run the following (all parameters are optional, shorthands can be seen with the `--help` flag of the command):

```
python train_ret.py -l <language> -d <dataset> -n <number of analogies in training set> -v <number of analogies in validation/development set> -t <number of analogies in test set>  --max_epochs <number of training epochs> -C <one of "cosine embedding loss", "relative shuffle", "relative all", "all">
```
Examples: 
- `python train_clf.py -l arabic` to train on Arabic (by default on Sigmorphon 2016), using up to 50000 analogies (default) for 20 epochs (default);
- `python train_clf.py -l german -d 2019 -n 50000 --max_epochs 20` to train on German of Sigmorphon 2019, using up to 50000 analogies for 20 epochs.

## Files and Folders TO RE-CHECK
Folders in the directory:
- `baseline`: scripts to run each of the 3 baselines (`alea`, `kolmo`, and `lepage`);
- `embeddings`: scripts (and, once downloaded, model files) of the pre-trained embeddings used in our early experiments;
- `utils.py`: generic tools used throught the code;
- `logs`: files generated by the training scripts, also contains the trained models;
- `results`: some other files generated by the training scripts, contains pre-aggregated results;
- `sigmorphon2016`: data files of the Sigmorphon2016 dataset;
- `sigmorphon2019`: data files of the Sigmorphon2019 dataset;
- `snippets`: several scripts used through the development of the approach, typically for evaluation and plotting.

Files at the root of the directory:
- `analogy_clf.py`: analogy classifier model definition;
- `analogy_reg.py`: analogy regression model definition;
- `cnn_embeddings.py`: morphological embedding model definition;
- `config.py`: file centralizing the most important paths and path templates of the project;
- `environment.yml`: file containing the environment data to create the anaconda environment;
- `japanese-task1-train`: backup of the japanese data extracted from the Japanese Bigger Analogy Test Set, in the Sigmorphon2016 task 1 format; to be used if Sigmorphon2016 has to be downloaded manually;
- `README.md`: this file;
- `train_clf.py`: file to train a classification model and the corresponding embedding model on a given language;
- `train_clf_transfer.py`: file to train a classification model and the corresponding embedding model in a transfer learning setting;
- `train_ret.py`: file to train a retrieval model and the corresponding embedding model on a given language.