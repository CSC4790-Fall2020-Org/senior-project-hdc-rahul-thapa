import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import copy
import pandas as pd
from perturbations_MNIST import skew, rotate, noise, brightness, elastic_transform
import os, argparse
from tqdm import tqdm
import sys
from datetime import datetime

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(f"./logs/HDXplore_eval_{file_name}_logfile.log", "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def shuffle(X, y):
    permutation = np.arange(X.shape[0])
    np.random.shuffle(permutation)
    return X[permutation], y[permutation]

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.
    X_test = X_test/255.
    return X_train, labels_train, X_test, labels_test

def get_scene(img, proj):
    return np.dot(img, proj.T)

# Transform the image vectors into the hypervectors
def get_scenes(images, proj):
    return np.dot(images[:NUM_SAMPLES, :], proj.T)

def classify(images, digit_vectors):
    similarities = cosine_similarity(images, digit_vectors)
    classifications = np.argmax(similarities, axis=1)
    return classifications

def projections(X, seeds, encoding="float"):
    print("Seeds: ", seeds)
    print("Encoding: ", encoding)
    print("Generating random projection...")
    IMG_LEN = 28
    X_projs = []
    projs = []
    for seed in seeds:
        np.random.seed(seed)
        proj = np.random.rand(D, IMG_LEN * IMG_LEN)
        if encoding == "bipolar":
            proj[proj==0] = -1
        print(proj.shape)
        print("Projecting images to higher dim space...")
        X_copy = get_scenes(X, proj)
        projs.append(proj)
        X_projs.append(X_copy)        
    return X_projs, projs

def discrepancies(models, X, y, seeds):
    X_projs, projs = projections(X, seeds)
    results = []
    for i in range(len(models)):
        predictions = classify(X_projs[i], models[i])
        acc = accuracy_score(y[:X_projs[i].shape[0]], predictions)
        print(f"Seed {seeds[i]} Accuracy: ", acc)
        results.append(predictions)
    
    if len(models) == 2:
        df = pd.DataFrame({f'model_{seeds[0]}': list(results[0]),
                   f'model_{seeds[1]}': list(results[1]),
                   'y': y})
        df_discrepancies = df[(df[f"model_{seeds[0]}"] != df[f"model_{seeds[1]}"])]
        df_non_discrepancies = df[(df[f"model_{seeds[0]}"] == df[f"model_{seeds[1]}"])]
    elif len(models) == 3:
        df = pd.DataFrame({f'model_{seeds[0]}': list(results[0]),
                   f'model_{seeds[1]}': list(results[1]),
                   f'model_{seeds[2]}': list(results[2]),
                   'y': y})
        df_discrepancies = df[(df[f"model_{seeds[0]}"] != df[f"model_{seeds[1]}"]) | (df[f"model_{seeds[0]}"] != df[f"model_{seeds[2]}"])]
        df_non_discrepancies = df[(df[f"model_{seeds[0]}"] == df[f"model_{seeds[1]}"]) & (df[f"model_{seeds[0]}"] == df[f"model_{seeds[2]}"])]
    else:
        print(f"This framework does not support {len(seeds)} number of models")
    print(f"There are {len(df_discrepancies)} adversarial cases.")
    
    df_discrepancies.reset_index(inplace=True)
    df_non_discrepancies.reset_index(inplace=True)
    return df_discrepancies, df_non_discrepancies

def perturb_discrepancies(models, projs, X, df_non_discrepancies, perturbations=[skew, noise, brightness, elastic_transform]):
    n = len(df_non_discrepancies)
    new_df = df_non_discrepancies.sample(n)
    X_discrepancies, X_discrepancies_projs, y_pred, y = [], [], [], []
    pbar = tqdm(total = len(new_df))
    for row in new_df.iterrows():
        idx = row[1]["index"]
        y_true = row[1]["y"]
        for perturb in perturbations:
            old_image = X[idx].reshape(28, 28)
            new_image = perturb(old_image)
            new_image = new_image.reshape(28*28)
            predictions = []
            images = []
            for i in range(len(models)):
                proj = projs[i]
                hv = get_scene(new_image, proj).reshape(1, -1)
                predictions.append(classify(hv, models[i])[0])
                images.append(hv)
                if len(predictions) == 3:
                    if len(set(predictions)) > 1:
                        X_discrepancies.append(new_image)
                        X_discrepancies_projs.append(images)
                        y_pred.append(predictions)
                        y.append(y_true)
        pbar.update(1)
    pbar.close()

    print(f"{len(y)} perturbations found.")
    return X_discrepancies, X_discrepancies_projs, y_pred, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDXplore Evaluation')
    parser.add_argument('--path_to_raw_model', default='./models/raw_models_seeds_30#40#50_epochs_20_split_33_random_42.npy', type=str, help='Path to the model without applying HDXplore')
    parser.add_argument('--path_to_retrained_model', default='./models/retrained_perturb_models_seeds_30#40#50_epochs_20_split_33_random_42.npy', type=str, help='Path to the retrained model using HDXplore')
    parser.add_argument('--seeds', default='30#40#50', type=str, help="Three random seeds (ex: if [30, 40, 50], --seeds 30#40#50)")
    parser.add_argument('--hd_dimension', default=10000, type=int, help='Dimension of Hypervector. Default 10000')

    args = parser.parse_args()

    # Extracting name for saving log file
    file_name = args.path_to_raw_model
    temp_file = file_name.split("/")[-1]
    temp_file = temp_file.split(".")[0]
    temp_file = temp_file.split("_")[-8:]
    file_name = "_".join(temp_file)

    sys.stdout = Logger(file_name)

    _, _, X_test, y_test = load_dataset()

    NUM_SAMPLES = X_test.shape[0]
    IMG_LEN = 28
    D = args.hd_dimension

    seeds = args.seeds.split('#')
    seeds = [int(seed) for seed in seeds]

    raw_models = np.load(args.path_to_raw_model)
    retrained_models = np.load(args.path_to_retrained_model)

    print()
    print("-------------------Raw Model Evaluation-----------------------")
    print()
    df_discrepancies_raw, df_non_discrepancies_raw = discrepancies(raw_models, X_test, y_test, seeds)
    print()
    print("-------------------Retrained Model Evaluation------------------------")
    print()
    df_discrepancies, df_non_discrepancies = discrepancies(retrained_models, X_test, y_test, seeds)
    idx_retrained = set(df_non_discrepancies["index"])
    idx_raw = set(df_non_discrepancies_raw["index"])
    idx = list(idx_retrained&idx_raw)
    X_temp = X_test[idx]
    y_temp = y_test[idx]
    
    print()
    print("------------Checking to make sure there are no more discrepancies in the new dataset-----------")
    print()
    print("------------------Raw Model-----------------")
    print()
    df_discrepancies_raw, df_non_discrepancies_raw = discrepancies(raw_models, X_temp, y_temp, seeds)
    print()
    print("------------------Retrained Model-----------------")
    print()
    df_discrepancies, df_non_discrepancies = discrepancies(retrained_models, X_temp, y_temp, seeds)
    X_projs, projs = projections(X_temp, seeds)
    print()
    print("-------------------Manual Perturbations Test---------------------------")
    print("------------------Raw Model-----------------")
    print()
    _, _, _, _ = perturb_discrepancies(raw_models, projs, X_temp, df_non_discrepancies)
    print()
    print("------------------Retrained Model-----------------")
    print()
    _, _, _, _ = perturb_discrepancies(retrained_models, projs, X_temp, df_non_discrepancies)
    
    
    
    
