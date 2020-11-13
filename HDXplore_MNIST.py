"""
Make sure you take a look at the readme before you run/make change to this code.
"""

# all the packages we need
import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import copy
import pandas as pd
from perturbations_MNIST import skew, rotate, noise, brightness, elastic_transform
import os, argparse
from tqdm import tqdm
import sys
from datetime import datetime
import time

# for logging the output spitted out in the terminal
class Logger(object):
    def __init__(self, seeds, epochs, split, random_state):
        self.terminal = sys.stdout
        self.log = open(f"./logs/HDXplore_seeds_{seeds}_epochs_{epochs}_split_{int(split*100)}_random_{random_state}_logfile.log", "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

# an extra function in case you want to shuffle the dataset. Not so necessary as you can use the random seed
# def shuffle(X, y):
#     permutation = np.arange(X.shape[0])
#     np.random.shuffle(permutation)
#     return X[permutation], y[permutation]

# loading the MNIST dataset from the data folder
def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.
    X_test = X_test/255.
    return X_train, labels_train, X_test, labels_test

def avg(nums):
    return sum(nums)/len(nums)

# projecting an image to higher dimension
def get_scene(img, proj):
    return np.dot(img, proj.T)

# project the entire images into the hypervectors
def get_scenes(images, proj):
    return np.dot(images[:, :], proj.T)

# classifying the images using cosine similarity
def classify(images, digit_vectors):
    similarities = cosine_similarity(images, digit_vectors)
    classifications = np.argmax(similarities, axis=1)
    return classifications

# gives you HD classifiers based on the seed and encoding you provide. Note: for this research, we only used floating point encoding.
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
    
    predictions_train = classify(X_train_copy, digit_vectors)
    acc_train = accuracy_score(y_train[:X_train_copy.shape[0]], predictions_train)
    print("Train accuracy: ", acc_train)
    
    X_valid_copy = get_scenes(X_valid, proj)
    predictions_valid = classify(X_valid_copy, digit_vectors)
    acc_valid = accuracy_score(y_valid[:X_valid_copy.shape[0]], predictions_valid)
    print("Valid accuracy: ", acc_valid)

    X_test_copy = get_scenes(X_test, proj)
    predictions_test = classify(X_test_copy, digit_vectors)
    acc_test = accuracy_score(y_test[:X_test_copy.shape[0]], predictions_test)
    print("Test accuracy: ", acc_test)
        
    return digit_vectors, proj, X_train_copy, X_valid_copy, X_test_copy

# takes the model and gives out the discprepancies and non-discrepancies
def discrepancies(models):
    results_train = []
    results_valid = []
    results_test = []
    temp_train_acc = []
    temp_valid_acc = []
    temp_test_acc = []
    for i in range(len(models)):
        predictions_train = classify(X_train_projs[i], models[i])
        acc_train = accuracy_score(y_train[:X_train_projs[i].shape[0]], predictions_train)
        temp_train_acc.append(acc_train)
        
        predictions_valid = classify(X_valid_projs[i], models[i])
        acc_valid = accuracy_score(y_valid[:X_valid_projs[i].shape[0]], predictions_valid)
        temp_valid_acc.append(acc_valid)

        predictions_test = classify(X_test_projs[i], models[i])
        acc_test = accuracy_score(y_test[:X_test_projs[i].shape[0]], predictions_test)
        temp_test_acc.append(acc_test)
        
        print(f"Seed {seeds[i]} Train Accuracy: ", acc_train)
        print(f"Seed {seeds[i]} Valid Accuracy: ", acc_valid)
        print(f"Seed {seeds[i]} Test Accuracy: ", acc_test)
        
        results_train.append(predictions_train)
        results_valid.append(predictions_valid)
        results_test.append(predictions_test)
    
    global_acc_train.append(avg(temp_train_acc))
    global_acc_valid.append(avg(temp_valid_acc))
    global_acc_test.append(avg(temp_test_acc))

    
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
        df_test = pd.DataFrame({f'model_{seeds[0]}': list(results_test[0]),
                   f'model_{seeds[1]}': list(results_test[1]),
                   f'model_{seeds[2]}': list(results_test[2]),
                   'y': y_test})
        df_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] != df_train[f"model_{seeds[1]}"]) | (df_train[f"model_{seeds[0]}"] != df_train[f"model_{seeds[2]}"])]
        df_discrepancies_valid = df_valid[(df_valid[f"model_{seeds[0]}"] != df_valid[f"model_{seeds[1]}"]) | (df_valid[f"model_{seeds[0]}"] != df_valid[f"model_{seeds[2]}"])]
        df_discrepancies_test = df_test[(df_test[f"model_{seeds[0]}"] != df_test[f"model_{seeds[1]}"]) | (df_test[f"model_{seeds[0]}"] != df_test[f"model_{seeds[2]}"])]
        df_non_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] == df_train[f"model_{seeds[1]}"]) & (df_train[f"model_{seeds[0]}"] == df_train[f"model_{seeds[2]}"])]
        df_non_discrepancies_valid = df_valid[(df_valid[f"model_{seeds[0]}"] == df_valid[f"model_{seeds[1]}"]) & (df_valid[f"model_{seeds[0]}"] == df_valid[f"model_{seeds[2]}"])]
        df_non_discrepancies_test = df_test[(df_test[f"model_{seeds[0]}"] == df_test[f"model_{seeds[1]}"]) & (df_test[f"model_{seeds[0]}"] == df_test[f"model_{seeds[2]}"])]
    else:
        print(f"This framework does not support {len(seeds)} number of models")
    
    print(f"There are {len(df_discrepancies_train)} adversarial cases in training set.")
    print(f"There are {len(df_discrepancies_valid)} adversarial cases in validation set.")
    print(f"There are {len(df_discrepancies_test)} adversarial cases in testing set.")

    global_dis_train.append(len(df_discrepancies_train))
    global_dis_valid.append(len(df_discrepancies_valid))
    global_dis_test.append(len(df_discrepancies_test))
    #print(f"There are {len(df_non_discrepancies_train)} non adversarial cases in training set.")
    #print(f"There are {len(df_non_discrepancies_valid)} non adversarial cases in validation set.")
    
    df_discrepancies_train.reset_index(inplace=True)
    df_discrepancies_valid.reset_index(inplace=True)
    df_discrepancies_test.reset_index(inplace=True)
    df_non_discrepancies_train.reset_index(inplace=True)
    df_non_discrepancies_valid.reset_index(inplace=True)
    df_non_discrepancies_test.reset_index(inplace=True)
    df_discrepancies_valid.to_excel("./temp/discrepancies_valid.xlsx")
    df_non_discrepancies_valid.to_excel("./temp/non_discrepancies_valid.xlsx")
    df_discrepancies_test.to_excel("./temp/discrepancies_test.xlsx")
    df_non_discrepancies_test.to_excel("./temp/non_discrepancies_test.xlsx")
    return df_discrepancies_train, df_discrepancies_valid, df_discrepancies_test, df_non_discrepancies_train, df_non_discrepancies_valid, df_non_discrepancies_test

"""
Takes the model and retrains it for given epoch. This is perhaps the most important step.
It is using all the discrepant images (the one with and without manual perturbations) for updating digit vectors
"""
def retraining(models, epochs, method="dynamic", dataset="validation"):
    print()
    df_discrepancies_train, df_discrepancies_valid, _, _, _, _ = discrepancies(models)
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
            df_discrepancies_train, df_discrepancies_valid, _, _, _ = discrepancies(models)
    print("Retraining Stopped...")
    return models

# Just a helper function for testing. Not used in the pipeline of the framework.
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
        _, _, _, _, _, _ = discrepancies(models)
    print("Retraining Stopped ...")
    return models

"""
this function takes the model, finds the images that all model agreed on, apply 4 predefined perturbations, and saves
the discrepant perturbed images 
"""
def perturb_discrepancies(models, X, df_non_discrepancies, perturbations=[skew, noise, brightness, elastic_transform]):
    n = len(df_non_discrepancies)
    new_df = df_non_discrepancies.sample(n)
    X_discrepancies, X_discrepancies_projs, y_pred, y = [], [], [], []
    pbar = tqdm(total = len(new_df))
    start = time.time()
    perturbations_temp = []
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
    end = time.time()
    pbar.close()
    print(f"{len(perturbations)} perturbations was applied to {len(new_df)} images.")
    print(f"{len(y)} perturbations found in {end-start} seconds.")
    return X_discrepancies, X_discrepancies_projs, y_pred, y


# main function that uses function above to create the pipeline for our framework
if __name__ == '__main__':

    # parameterizing our script. You can also run the default values to replicate our result. 
    parser = argparse.ArgumentParser(description='HDXplore Framework for MNIST')
    parser.add_argument('--hd_dimension', default=10000, type=int, help='Dimension of Hypervector. Default 10000')
    parser.add_argument('--seeds', default='30#40#50', type=str, help="Three random seeds (ex: if [30, 40, 50], --seeds 30#40#50)")
    parser.add_argument('--epochs', default=20, type=int, help="Number of epochs for retraining. Defualt 20.")
    parser.add_argument('--dataset', default='validation', type=str, help="training or validation dataset to run the HDXplore algorithm on. Default validation")
    parser.add_argument('--method', default='dynamic', type=str, help="dynamic for updating the discrepant images on each epochs of retraining; static for retraining on the same initial discrepant images. Defualt dynamic.")
    parser.add_argument('--data_split', default=0.50, type=float, help="Split fraction for training and validation set. Defualt 0.50.")
    parser.add_argument('--random_state', default=42, type=int, help="random state for train validation split. Defualt 42.")
    parser.add_argument('--perturbation', default='True', type=str, help="True if you want to apply manual perturbations to the images where all models agree on. Defualt True")

    args = parser.parse_args()
    sys.stdout = Logger(args.seeds, args.epochs, args.data_split, args.random_state)

    X_train, y_train, X_test, y_test = load_dataset()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.data_split, random_state=args.random_state)
    np.save(f"./temp/X_valid.npy", X_valid)
    np.save(f"./temp/y_valid.npy", y_valid)

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
    X_test_projs = []
    global_acc_train = []
    global_acc_valid = []
    global_acc_test = []
    global_dis_train = []
    global_dis_valid = []
    global_dis_test = []

    print()
    print("-------------------------HD Projection Module----------------------------")
    print()
    for i in range(len(seeds)):
        digit_vector, proj, X_train_copy, X_valid_copy, X_test_copy = HD_classifiers(seeds[i])
        models.append(digit_vector)
        projs.append(proj)
        X_train_projs.append(X_train_copy)
        X_valid_projs.append(X_valid_copy)
        X_test_projs.append(X_test_copy)

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
        df_discrepancies_train, df_discrepancies_valid, _, df_non_discrepancies_train, df_non_discrepancies_valid, _ = discrepancies(models)
        print()
        print("--------------------------------Perturbation Module-----------------------")
        print()
        X_perturb_images, X_perturb_images_projs, y_perturb_pred, y_perturb_true = perturb_discrepancies(models, X_valid, df_non_discrepancies_valid)
        exit()
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
        
        # X_valid_projs[0], X_valid_projs[1], X_valid_projs[2], y_valid = shuffle(X_valid_projs[0], X_valid_projs[1], X_valid_projs[2], y_valid, random_state=0)

        print()
        print("-------------------------------------Retraining Module-----------------------------")
        print()
        models = retraining(models, epochs, method=args.method, dataset=args.dataset)
        np.save(f"./models/retrained_perturb_models_seeds_{args.seeds}_epochs_{args.epochs}_split_{int(args.data_split*100)}_random_{args.random_state}.npy", models)

    np.save(f"./temp/perturb_{args.perturbation}_dis_train.npy", global_dis_train)
    np.save(f"./temp/perturb_{args.perturbation}_dis_valid.npy", global_dis_valid)
    np.save(f"./temp/perturb_{args.perturbation}_dis_test.npy", global_dis_test)
    np.save(f"./temp/perturb_{args.perturbation}_acc_train.npy", global_acc_train)
    np.save(f"./temp/perturb_{args.perturbation}_acc_valid.npy", global_acc_valid)
    np.save(f"./temp/perturb_{args.perturbation}_acc_test.npy", global_acc_test)

