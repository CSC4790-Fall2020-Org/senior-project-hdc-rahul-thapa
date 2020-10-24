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
    def __init__(self, seeds, epochs, split, random_state):
        self.terminal = sys.stdout
        self.log = open(f"./logs/HDXplore_seeds_{seeds}_epochs_{epochs}_split_{int(split*100)}_random_{random_state}_logfile.log", "w")
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


def HD_classifiers(seed, encoding="float"):
    print("Seed: ", seed)
    print("Encoding: ", encoding)
    print("Generating random projection...")
    np.random.seed(seed)
    proj = np.random.rand(D, IMG_LEN * IMG_LEN)
    if encoding == "bipolar":
        proj[proj==0] = -1
    print(proj.shape)
    print("Projecting images to higher dim space...")
    X_train_copy = get_scenes(X_train, proj)
    
    digit_vectors = np.zeros((10, D))
    print("Dimension of digit vector: ", digit_vectors.shape)
    
    for i in range(NUM_SAMPLES):
        digit_vectors[y_train[i]] += X_train_copy[i]
    digit_vectors = np.array(digit_vectors)
    
    if encoding == "bipolar":
        digit_vectors[digit_vectors > 0] = 1
        digit_vectors[digit_vectors <= 0] = -1
    
    predictions = classify(X_train_copy, digit_vectors)
    acc = accuracy_score(y_train[:X_train_copy.shape[0]], predictions)
    print("Train accuracy: ", acc)
    
    X_valid_copy = get_scenes(X_valid, proj)
    predictions = classify(X_valid_copy, digit_vectors)
    acc = accuracy_score(y_valid[:X_valid_copy.shape[0]], predictions)
    print("Valid accuracy: ", acc)
        
    return digit_vectors, proj, X_train_copy, X_valid_copy


def discrepancies(models):
    results_train = []
    results_valid = []
    for i in range(len(models)):
        predictions_train = classify(X_train_projs[i], models[i])
        acc_train = accuracy_score(y_train[:X_train_projs[i].shape[0]], predictions_train)
        
        predictions_valid = classify(X_valid_projs[i], models[i])
        acc_valid = accuracy_score(y_valid[:X_valid_projs[i].shape[0]], predictions_valid)
        
        print(f"Seed {seeds[i]} Train Accuracy: ", acc_train)
        print(f"Seed {seeds[i]} Valid Accuracy: ", acc_valid)
        
        results_train.append(predictions_train)
        results_valid.append(predictions_valid)
    
    if len(models) == 2:
        df_train = pd.DataFrame({f'model_{seeds[0]}': list(results_train[0]),
                   f'model_{seeds[1]}': list(results_train[1]),
                   'y': y_train})
        df_valid = pd.DataFrame({f'model_{seeds[0]}': list(results_valid[0]),
                   f'model_{seeds[1]}': list(results_valid[1]),
                   'y': y_valid})
        df_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] != df_train[f"model_{seeds[1]}"])]
        df_discrepancies_valid = df_valid[(df_valid[f"model_{seeds[0]}"] != df_valid[f"model_{seeds[1]}"])]
        df_non_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] == df_train[f"model_{seeds[1]}"])]
        df_non_discrepancies_valid = df_valid[(df_valid[f"model_{seeds[0]}"] == df_valid[f"model_{seeds[1]}"])]
    elif len(models) == 3:
        df_train = pd.DataFrame({f'model_{seeds[0]}': list(results_train[0]),
                   f'model_{seeds[1]}': list(results_train[1]),
                   f'model_{seeds[2]}': list(results_train[2]),
                   'y': y_train})
        df_valid = pd.DataFrame({f'model_{seeds[0]}': list(results_valid[0]),
                   f'model_{seeds[1]}': list(results_valid[1]),
                   f'model_{seeds[2]}': list(results_valid[2]),
                   'y': y_valid})
        df_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] != df_train[f"model_{seeds[1]}"]) | (df_train[f"model_{seeds[0]}"] != df_train[f"model_{seeds[2]}"])]
        df_discrepancies_valid = df_valid[(df_valid[f"model_{seeds[0]}"] != df_valid[f"model_{seeds[1]}"]) | (df_valid[f"model_{seeds[0]}"] != df_valid[f"model_{seeds[2]}"])]
        df_non_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] == df_train[f"model_{seeds[1]}"]) & (df_train[f"model_{seeds[0]}"] == df_train[f"model_{seeds[2]}"])]
        df_non_discrepancies_valid = df_valid[(df_valid[f"model_{seeds[0]}"] == df_valid[f"model_{seeds[1]}"]) & (df_valid[f"model_{seeds[0]}"] == df_valid[f"model_{seeds[2]}"])]
    else:
        print(f"This framework does not support {len(seeds)} number of models")
    
    print(f"There are {len(df_discrepancies_train)} adversarial cases in training set.")
    print(f"There are {len(df_discrepancies_valid)} adversarial cases in validation set.")
    #print(f"There are {len(df_non_discrepancies_train)} non adversarial cases in training set.")
    #print(f"There are {len(df_non_discrepancies_valid)} non adversarial cases in validation set.")
    
    df_discrepancies_train.reset_index(inplace=True)
    df_discrepancies_valid.reset_index(inplace=True)
    df_non_discrepancies_train.reset_index(inplace=True)
    df_non_discrepancies_valid.reset_index(inplace=True)
    return df_discrepancies_train, df_discrepancies_valid, df_non_discrepancies_train, df_non_discrepancies_valid


def retraining(models, epochs, method="static", dataset="training"):
    print()
    df_discrepancies_train, df_discrepancies_valid, _, _ = discrepancies(models)
    print("Retraining Started...")
    print()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}: ")
        if dataset.lower() == "training":
            df_retrain = df_discrepancies_train.copy()
        elif dataset.lower() == "validation":
            df_retrain = df_discrepancies_valid.copy()

        for row in df_retrain.iterrows():
            idx = row[1]["index"]
            for i in range(len(seeds)):
                y_false = row[1][f"model_{seeds[i]}"]
                y_true = row[1]["y"]

                if dataset.lower() == "training":
                    hv = X_train_projs[i][idx]
                elif dataset.lower() == "validation":
                    hv = X_valid_projs[i][idx]
                models[i][y_false] -= hv
                models[i][y_true] += hv

        if method.lower() == "static":
            _, _, _, _ = discrepancies(models)
        elif method.lower() == "dynamic": 
            df_discrepancies_train, df_discrepancies_valid, _, _ = discrepancies(models)
    print("Retraining Stopped...")
    return models

def perturb_retraining(models, epochs=1):
    _, _, df_non_discrepancies_train, df_non_discrepanceis_valid = discrepancies(models)
    _, X_discrepancies_projs, y_pred, y = perturb_discrepancies(models, X_valid, df_non_discrepancies_valid, n=len(df_non_discrepancies_valid))
    print("Retraining with manual perturbations data started")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}: ")
        for row in range(len(y)):
            for i in range(len(models)):
                y_false = y_pred[row][i]
                y_true = y[row]
                hv = X_discrepancies_projs[row][i]
                models[i][y_false] -= hv[0]
                models[i][y_true] += hv[0]
        _, _, _, _ = discrepancies(models)
    print("Retraining Stopped ...")
    return models


def perturb_discrepancies(models, X, df_non_discrepancies, perturbations=[skew, noise, brightness, elastic_transform]):
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
    parser = argparse.ArgumentParser(description='HDXplore Framework for MNIST')
    parser.add_argument('--hd_dimension', default=10000, type=int, help='Dimension of Hypervector. Default 10000')
    parser.add_argument('--seeds', default='30#40#50', type=str, help="Three random seeds (ex: if [30, 40, 50], --seeds 30#40#50)")
    parser.add_argument('--epochs', default=20, type=int, help="Number of epochs for retraining. Defualt 20.")
    parser.add_argument('--dataset', default='validation', type=str, help="training or validation dataset to run the HDXplore algorithm on. Default validation")
    parser.add_argument('--method', default='dynamic', type=str, help="dynamic for updating the discrepant images on each epochs of retraining; static for retraining on the same initial discrepant images. Defualt dynamic.")
    parser.add_argument('--data_split', default=0.33, type=float, help="Split fraction for training and validation set. Defualt 0.33.")
    parser.add_argument('--random_state', default=42, type=int, help="random state for train validation split. Defualt 42.")
    parser.add_argument('--perturbation', default='True', type=str, help="True if you want to apply manual perturbations to the images where all models agree on. Defualt True")

    args = parser.parse_args()
    sys.stdout = Logger(args.seeds, args.epochs, args.data_split, args.random_state)

    X_train, y_train, X_test, y_test = load_dataset()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.data_split, random_state=args.random_state)

    D = args.hd_dimension
    IMG_LEN = 28
    NUM_SAMPLES = X_train.shape[0]
    seeds = args.seeds.split('#')
    seeds = [int(seed) for seed in seeds]
    epochs = args.epochs

    models = []
    projs = []
    X_train_projs = []
    X_valid_projs = []

    print()
    print("-------------------------HD Projection Module----------------------------")
    print()
    for i in range(len(seeds)):
        digit_vector, proj, X_train_copy, X_valid_copy = HD_classifiers(seeds[i])
        models.append(digit_vector)
        projs.append(proj)
        X_train_projs.append(X_train_copy)
        X_valid_projs.append(X_valid_copy)

    np.save(f"./models/raw_models_seeds_{args.seeds}_epochs_{args.epochs}_split_{int(args.data_split*100)}_random_{args.random_state}.npy", models)

    if args.perturbation == 'False':
        print()
        print("------------------------------Retraining Module--------------------------")
        print()
        models = retraining(models, epochs, method=args.method, dataset=args.dataset)
        np.save(f"./models/retrained_models_seeds_{args.seeds}_epochs_{args.epochs}_split_{int(args.data_split*100)}_random_{args.random_state}.npy", models)
    else:
        print()
        print("-----------------------------Testing Module------------------------------")
        print()
        df_discrepancies_train, df_discrepancies_valid, df_non_discrepancies_train, df_non_discrepancies_valid = discrepancies(models)
        print()
        print("--------------------------------Perturbation Module-----------------------")
        print()
        X_perturb_images, X_perturb_images_projs, y_perturb_pred, y_perturb_true = perturb_discrepancies(models, X_valid, df_non_discrepancies_valid)
        np.save(f"./perturbed_images/X_perturb_images_seeds_{args.seeds}_epochs_{args.epochs}_split_{int(args.data_split*100)}_random_{args.random_state}.npy", X_perturb_images)
        np.save(f"./perturbed_images/X_perturb_images_projs_seeds_{args.seeds}_epochs_{args.epochs}_split_{int(args.data_split*100)}_random_{args.random_state}.npy", X_perturb_images_projs)
        np.save(f"./perturbed_images/y_perturb_pred_seeds_{args.seeds}_epochs_{args.epochs}_split_{int(args.data_split*100)}_random_{args.random_state}.npy", y_perturb_pred)
        np.save(f"./perturbed_images/y_perturb_true_seeds_{args.seeds}_epochs_{args.epochs}_split_{int(args.data_split*100)}_random_{args.random_state}.npy", y_perturb_true)


        temp = []
        for i in range(len(seeds)):
            temp_image = [X_perturb_images_projs[j][i][0] for j in range(len(X_perturb_images_projs))]
            X_valid_proj = np.append(X_valid_projs[i], np.array(temp_image), axis=0)
            X_valid_projs[i] = X_valid_proj
        y_valid = np.append(np.array(y_valid), y_perturb_true, axis=0)

        print()
        print("-------------------------------------Retraining Module-----------------------------")
        print()
        models = retraining(models, epochs, method=args.method, dataset=args.dataset)
        np.save(f"./models/retrained_perturb_models_seeds_{args.seeds}_epochs_{args.epochs}_split_{int(args.data_split*100)}_random_{args.random_state}.npy", models)
