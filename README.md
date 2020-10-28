# HDXplore: Differential Testing in Hyperdimensional Computing Paradigm

**Note: If you find my work useful, please use the approproiate citations as below.**

```
(bibtex)
```

If there are any technical questions after the README, please contact:
* connectthapa84@gmail.com

## Core Team
![team](https://github.com/CSC4790-Fall2020-Org/senior-project-hdc-rahul-thapa/blob/master/assets/team.jpg)

* Dependable, Efficient, and Intelligent Computing Lab (DETAIL)
  	* Rahul Thapa (Lead Researcher, B.S in CS)
	* Dongning Ma (Ph.D. Students, EECS)
	* Xun Jiao (Faculty Advisor, EECS)

## Table of Contents
1. [Project Overview](#Overview)
2. [Requirements](#Requirements)
3. [Dataset](#Dataset) 
4. [Training/Retraining Script](#Training)
5. [Evaluation Script](#Evaluation)
6. [Results](#Results)
7. [End-to-End Demo](#Demo)

## Overview

**Note: If you want to learn more about our work, check out our full paper at [link](link)**

HDC for image classification system diagram

![HDC_classification](https://github.com/CSC4790-Fall2020-Org/senior-project-hdc-rahul-thapa/blob/master/assets/HDC_classification.jpeg)

HDXplore framework highlevel diagram

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
Make sure you clone this repository and open command prompt with this project as parent directory. 

## Steps for training/retraining
1. We provide you with the HDXplore training script, [HDXplore_MNIST.py](./HDXplore_MNIST.py)
2. To train the model, run the following script. Note that all of these parameters has default value. Therefore, if you simply run the script without any parameter, it will give you a version of our result. 
```
python HDXplore_MNIST.py \
    --hd_dimension 10000 \
    --seeds 30#40#50 \
    --epochs 20 \
    --dataset validation \
    --method dynamic \
    --data_split 0.50 \
    --random_state 42 \
    --perturbation True
```
## Evaluation
1. We provide you with the HDXplore evaluation script, [HDXplore_eval.py](./HDXplore_eval.py)
2. To evaluate the model, run the following script. Make sure you use the right models and epochs, consistent with the training/retraining process above. 
```
python HDXplore_eval.py \
    --path_to_raw_model ./models/raw_models_seeds_30#40#50_epochs_20_split_50.npy \
    --path_to_retrained_model ./models/retrained_perturb_models_seeds_30#40#50_epochs_20_split_50.npy \
    --seeds 30#40#50 \
    --hd_dimension 10000
```

## Results

Our framework, HDXplore, has following contributions/results:

1. Our framework generated 3-4 discrepant images per second. These discrepancies were used for retraining our models for multiple epochs dynamically, which increased the accuracy of the models steadely.
2. We were able to increase the accuracy of Hyperdimensional computing (HD) models from ~81.5% to ~89.5.
3. The retrained model were more robust than the raw model without any retraining. On average, the retrained model has 40% less discrepancies than the raw model.

Results while Training and Validating HDC Models using HDXplore Framework on MNIST dataset

![acc_1](https://github.com/CSC4790-Fall2020-Org/senior-project-hdc-rahul-thapa/blob/master/assets/accuracy_curve.png)

![dis_1](https://github.com/CSC4790-Fall2020-Org/senior-project-hdc-rahul-thapa/blob/master/assets/discrepancies_curve.png)

Results while Testing the models on completely unseen MNIST Testing Dataset

![acc_test_1](https://github.com/CSC4790-Fall2020-Org/senior-project-hdc-rahul-thapa/blob/master/assets/accuracy_test_bar.png)

![dis_test_1](https://github.com/CSC4790-Fall2020-Org/senior-project-hdc-rahul-thapa/blob/master/assets/dis_test_bar.png)

![perturb_dis_test_1](https://github.com/CSC4790-Fall2020-Org/senior-project-hdc-rahul-thapa/blob/master/assets/perturb_dis_test_bar.png)


## Demo

**Note: If you want to see the full normal speed demo, here is the [link](https://drive.google.com/file/d/1XuxvOmO_U44uPGPP4FpCt9bfVqdzczBW/view?usp=sharing)**

![Demo](https://github.com/CSC4790-Fall2020-Org/senior-project-hdc-rahul-thapa/blob/master/assets/HDXplore_split_50_edit.gif)

