# Differential Testing in Hyperdimensional Computing Paradigm Codebase

**Note: If you find my work useful, please use the approproiate citations as below.**

```
(bibtex)
```

If there are any technical questions after the README, please contact:
* connectthapa84@gmail.com
* rthapa@villanova.edu


## Core Team
* Dependable, Efficient, and Intelligent Computing Lab (DETAIL)
  	* Rahul Thapa (Lead Researcher, B.S in CS)
	* Xun Jiao (Faculty Advisor, EECS)
	* Dongning Ma (Ph.D. Students, EECS)

## Table of Contents
1. [Project Overview](#Overview)
2. [Requirements](#Requirements)
3. [Dataset](#Dataset) 
4. [Training/Retraining Script](#Training)
5. [Evaluation Script](#Evaluation)
6. [Results](#Results)

## Overview

**Note: If you want to learn more about our work, checko out our full paper at [link](link)**

![overview](https://github.com/CSC4790-Fall2020-Org/senior-project-hdc-rahul-thapa/blob/master/assets/HDC_design_diagram.png)

## Requirements

The main requirements are listed below:
* Python 3.6
* Pandas
* Numpy
* OpenCV 4.2.0
* Scikit-Learn
* Matplotlib
* tqdm
* mnist

## Dataset

The link to the datasets we used in our experiments are given below. However, we only mostly focused on MNIST.

* MNIST: http://yann.lecun.com/exdb/mnist/
* Face Images: https://archive.ics.uci.edu/ml/datasets/CMU+Face+Images 

We have also provided a data folder with MNIST dataset for ease of use.  


## Training

## Steps for training
1. We provide you with the HDXplore training script, [HDXplore_MNIST.py](./HDXplore_MNIST.py)
2. To train the model:
```
python HDXplore_MNIST.py \
    --hd_dimension 10000 \
    --seeds 30,40,50 \
    --epochs 20 \
    --dataset validation \
    --method dynamic \
    --data_split 0.33
    --random_state 42 \
    --perturbation True
```
## Evaluation
1. We provide you with the HDXplore training script, [HDXplore_eval.py](./HDXplore_eval.py)
2. To evaluate the model:
```
python HDXplore_eval.py \
    --path_to_raw_model ./models/raw_models.npy \
    --path_to_retrained_model ./models/retrained_perturb_models.npy \
    --seeds 30,40,50 \
    --hd_dimension 10000
```

## Results






