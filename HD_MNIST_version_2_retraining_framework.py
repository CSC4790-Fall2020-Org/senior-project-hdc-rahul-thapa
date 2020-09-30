#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import copy
import pandas as pd
from perturbations_MNIST import skew, rotate, noise, brightness, elastic_transform

# In[ ]:


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


# In[ ]:


def HD_classifiers(seed, encoding="float"):
    # print("Generating random projection...")
    # proj = np.random.rand(D, IMG_LEN * IMG_LEN)
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
    
    X_test_copy = get_scenes(X_test, proj)
    predictions = classify(X_test_copy, digit_vectors)
    acc = accuracy_score(y_test[:X_test_copy.shape[0]], predictions)
    print("Test accuracy: ", acc)
        
    return digit_vectors, proj, X_train_copy, X_test_copy


# In[ ]:


def discrepancies(models):
    results_train = []
    results_test = []
#     model_names = []
#     for seed, model in zip(seeds, models):
    for i in range(len(models)):
#         np.random.seed(seed)
#         model_names.append(f"model_{seed}")
#         proj = np.random.rand(D, IMG_LEN * IMG_LEN)
#         X_train_copy = get_scenes(X_train, proj)
#         X_test_copy = get_scenes(X_test, proj)
        
        predictions_train = classify(X_train_projs[i], models[i])
        acc_train = accuracy_score(y_train[:X_train_projs[i].shape[0]], predictions_train)
        
        predictions_test = classify(X_test_projs[i], models[i])
        acc_test = accuracy_score(y_test[:X_test_projs[i].shape[0]], predictions_test)
        
        print(f"Seed {seeds[i]} Train Accuracy: ", acc_train)
        print(f"Seed {seeds[i]} Test Accuracy: ", acc_test)
        
        results_train.append(predictions_train)
        results_test.append(predictions_test)
    
    if len(models) == 2:
        df_train = pd.DataFrame({f'model_{seeds[0]}': list(results_train[0]),
                   f'model_{seeds[1]}': list(results_train[1]),
                   'y': y_train})
        df_test = pd.DataFrame({f'model_{seeds[0]}': list(results_test[0]),
                   f'model_{seeds[1]}': list(results_test[1]),
                   'y': y_test})
        df_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] != df_train[f"model_{seeds[1]}"])]
        df_discrepancies_test = df_test[(df_test[f"model_{seeds[0]}"] != df_test[f"model_{seeds[1]}"])]
        df_non_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] == df_train[f"model_{seeds[1]}"])]
        df_non_discrepancies_test = df_test[(df_test[f"model_{seeds[0]}"] == df_test[f"model_{seeds[1]}"])]
    elif len(models) == 3:
        df_train = pd.DataFrame({f'model_{seeds[0]}': list(results_train[0]),
                   f'model_{seeds[1]}': list(results_train[1]),
                   f'model_{seeds[2]}': list(results_train[2]),
                   'y': y_train})
        df_test = pd.DataFrame({f'model_{seeds[0]}': list(results_test[0]),
                   f'model_{seeds[1]}': list(results_test[1]),
                   f'model_{seeds[2]}': list(results_test[2]),
                   'y': y_test})
        df_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] != df_train[f"model_{seeds[1]}"]) | (df_train[f"model_{seeds[0]}"] != df_train[f"model_{seeds[2]}"])]
        df_discrepancies_test = df_test[(df_test[f"model_{seeds[0]}"] != df_test[f"model_{seeds[1]}"]) | (df_test[f"model_{seeds[0]}"] != df_test[f"model_{seeds[2]}"])]
        df_non_discrepancies_train = df_train[(df_train[f"model_{seeds[0]}"] == df_train[f"model_{seeds[1]}"]) & (df_train[f"model_{seeds[0]}"] == df_train[f"model_{seeds[2]}"])]
        df_non_discrepancies_test = df_test[(df_test[f"model_{seeds[0]}"] == df_test[f"model_{seeds[1]}"]) & (df_test[f"model_{seeds[0]}"] == df_test[f"model_{seeds[2]}"])]
    else:
        print(f"This framework does not support {len(seeds)} number of models")
    
    print(f"There are {len(df_discrepancies_train)} adversarial cases in training set.")
    print(f"There are {len(df_discrepancies_test)} adversarial cases in testing set.")
    
    df_discrepancies_train.reset_index(inplace=True)
    df_discrepancies_test.reset_index(inplace=True)
    df_non_discrepancies_train.reset_index(inplace=True)
    df_non_discrepancies_test.reset_index(inplace=True)
    return df_discrepancies_train, df_discrepancies_test, df_non_discrepancies_train, df_non_discrepancies_test


# In[ ]:


def retraining(models, epochs, method="partial", dataset="training"):
#     models = []
#     projs = []
#     X_train_projs = []
#     X_test_projs = []
#     for i in range(len(seeds)):
#         digit_vector, proj, X_train_copy, X_test_copy = HD_classifiers(seeds[i])
#         models.append(digit_vector)
#         projs.append(proj)
#         X_train_projs.append(X_train_copy)
#         X_test_projs.append(X_test_copy)
    df_discrepancies_train, df_discrepancies_test, _, _ = discrepancies(models)
    print("Retraining Started")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}: ")

        if dataset.lower() == "training":
            df_retrain = df_discrepancies_train.copy()
        elif dataset.lower() == "testing":
            df_retrain = df_discrepancies_test.copy()

        for row in df_retrain.iterrows():
            idx = row[1]["index"]
            for i in range(len(seeds)):
                y_false = row[1][f"model_{seeds[i]}"]
                y_true = row[1]["y"]

                if dataset.lower() == "training":
                    hv = X_train_projs[i][idx]
                elif dataset.lower() == "testing":
                    hv = X_test_projs[i][idx]
#                 np.random.seed(seeds[i])
#                 proj = np.random.rand(D, IMG_LEN * IMG_LEN)
#                 hv = get_scene(X_train[idx].reshape((1, -1)), proj)
                models[i][y_false] -= hv
                models[i][y_true] += hv

        if method.lower() == "partial":
            _, _, _, _ = discrepancies(models)
        elif method.lower() == "full": 
            df_discrepancies_train, df_discrepancies_test, _, _ = discrepancies(models)
    print("Retraining Stopped...")
    return models

def perturb_retraining(models, epochs=1):
    _, _, df_non_discrepancies_train, df_non_discrepanceis_test = discrepancies(models)
    _, X_discrepancies_projs, y_pred, y = perturb_discrepancies(models, X_test, df_non_discrepancies_test, n=len(df_non_discrepancies_test))
    print("Retraining with manual perturbations data started")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}: ")
        for row in range(len(y)):
            for i in range(len(models)):
                y_false = y_pred[row][i]
                y_true = y[row]

                hv = X_discrepancies_projs[row]
                models[i][y_false] -= hv[0]
                models[i][y_true] += hv[0]
        _, _, _, _ = discrepancies(models)
    print("Retraining Stopped ...")
    return models


def perturb_discrepancies(models, X, df_non_discrepancies, n=4000, perturbations=[skew, noise, brightness, elastic_transform]):
    new_df = df_non_discrepancies.sample(n)
    X_discrepancies, X_discrepancies_projs, y_pred, y = [], [], [], []
    for row in new_df.iterrows():
        idx = row[1]["index"]
        y_true = row[1]["y"]
        for perturb in perturbations:
            old_image = X[idx].reshape(28, 28)
            new_image = perturb(old_image)
            new_image = new_image.reshape(28*28)
            predictions = []
            for i in range(len(models)):
                proj = projs[i]
                hv = get_scene(new_image, proj).reshape(1, -1)
                predictions.append(classify(hv, models[i])[0])
            if len(set(predictions)) > 1:
                print("True")
                X_discrepancies.append(new_image)
                X_discrepancies_projs.append(hv)
                y_pred.append(predictions)
                y.append(y_true)
    print(f"{len(y)} perturbations found.")
    return X_discrepancies, X_discrepancies_projs, y_pred, y

def retraining_test_direct():
    return

def retraining_train():
    return

def retraining_test():
    return

# In[ ]:


X_train, labels_train, _, _ = load_dataset()
# X_train, labels_train = shuffle(X_train, labels_train)
X_train, X_test, y_train, y_test = train_test_split(X_train, labels_train, test_size=0.33, random_state=42)


# In[ ]:


D = 10000 # dimensions in random space
IMG_LEN = 28
NUM_SAMPLES = X_train.shape[0]
seeds = [30, 40, 50]
epochs = 10

models = []
projs = []
X_train_projs = []
X_test_projs = []
for i in range(len(seeds)):
    digit_vector, proj, X_train_copy, X_test_copy = HD_classifiers(seeds[i])
    models.append(digit_vector)
    projs.append(proj)
    X_train_projs.append(X_train_copy)
    X_test_projs.append(X_test_copy)


# In[ ]:

#df_discrepancies_train, df_discrepancies_test, df_non_discrepancies_train, df_non_discrepancies_test = discrepancies(models)
#print(len(df_discrepancies_test))
#print(len(df_non_discrepancies_test))
#print(len(df_discrepancies_train))
#print(len(df_non_discrepancies_train))
models = retraining(models, epochs, method="full", dataset="testing")

# perturb_discrepancies(models, X_test, df_non_discrepancies_test, n=len(df_non_discrepancies_test))
# In[ ]:
#models = perturb_retraining(models)




# discrepancies(models, [30, 40, 50])

# digit_vector, proj, _, _ = HD_classifiers(40)

# def retraining(seeds, epochs, retrain_on='train', method="direct"):
#     models = []
#     projs = []
#     X_train_projs = []
#     X_test_projs = []
#     for i in range(len(seeds)):
#         digit_vector, proj, X_train_copy, X_test_copy = HD_classifiers(seeds[i])
#         models.append(digit_vector)
#         projs.append(proj)
#         X_train_projs.append(X_train_copy)
#         X_test_projs.append(X_test_copy)
#         
#     for epoch in range(epochs):
#         print("Retraining Started: ")
#         print(f"Epoch {epoch+1}")
#         results_test = []
#         results_train = []
#         for i in range(len(seeds)):
#             predictions_train = classify(X_train_projs[i], models[i])
#             predictions_test = classify(X_test_projs[i], models[i])
#             results_train.append(predictions_train)
#             results_test.append(predictions_test)
#         
#         results_train_dict = {}
#         results_test_dict = {}
#         model_names = []
#         for i in range(len(seeds)):
#             model_names.append(f"model_{seeds[i]}")
#             results_train_dict[f"model_{seeds[i]}"] = list(results_train[i])
#             results_test_dict[f"model_{seeds[i]}"] = list(results_test[i])
#         
#         df_train = pd.DataFrame(results_train_dict)
#         df_train["y"] = y_train
#         df_test = pd.DataFrame(results_test_dict)
#         df_test["y"] = y_test
#         
#         if retrain_on.lower() == "train":
#             for i in range(len(model_names)):
#                 mask = (df_train[model_names[i]] != df_train[model_names[i+1]])
#                 for j in range(i+2, len(model_names)):
#                     mask = mask + (df_train[model_names[i]] != df_train[model_names[j]])
#                 break
#             df_discrepencies_train = df_train[mask]
#             df_discrepencies_train.reset_index(inplace=True)
#             
#             if method == "direct":
#                 print("Retraining on training set...")
#                 for epoch in range(epochs):
#                     for row in df_discrepencies_train.iterrows():
#                         idx = row[1]["index"]
#                         for i in range(len(model_names)):
#                             y_false = row[1][model_names[i]]
#                             y_true = row[1]["y"]
#                             hv = get_scene(X_train[idx].reshape((1, -1)), projs[i])
#                             models[i][y_false] -= hv[0]
#                             models[i][y_true] += hv[0]
#                     
#                     print(f"Epoch {epoch+1}")
#                 for i in range(len(seeds)):
#                     predictions = classify(X_train_projs[i], models[i])
#                     acc = accuracy_score(y_train[:X_train_projs[i].shape[0]], predictions)
#                     print(model_names[i] + ": " + str(acc))
#                 print("Retraining Stopped...")
#                 return models
#         if retrain_on.lower() == "test":
#             for i in range(len(model_names)):
#                 mask = (df_test[model_names[i]] != df_test[model_names[i+1]])
#                 for j in range(i+2, len(model_names)):
#                     mask = mask + (df_test[model_names[i]] != df_test[model_names[j]])
#             df_discrepencies_test = df_train[mask]
#             
#         if retrain_on.lower() == 'both':
#             for i in range(len(model_names)):
#                 mask_test = (df_test[model_names[i]] != df_test[model_names[i+1]])
#                 mask_train = (df_train[model_names[i]] != df_train[model_names[i+1]])
#                 for j in range(i+2, len(model_names)):
#                     mask_test = mask_test + (df_test[model_names[i]] != df_test[model_names[j]])
#                     mask_train = mask_train + (df_train[model_names[i]] != df_train[model_names[j]])
#             df_discrepencies_test = df_train[mask_test]
#             df_discrepencies_train = df_train[mask_train]

# models = retraining([30, 40, 50], 10)

# len(models)

# model_30, model_40, model_50 = models

# import copy
# seeds = [30, 40, 50]
# results = []
# for seed, model in zip(seeds, models):
#     np.random.seed(seed)
#     proj = np.random.rand(D, IMG_LEN * IMG_LEN)
#     X_train_copy = get_scenes(X_train, proj)
#     predictions = classify(X_train_copy, model)
#     results.append(predictions)
#     print("here")

# import pandas as pd
# df = pd.DataFrame({'model_30': list(results[0]),
#                    'model_40': list(results[1]),
#                    'model_50': list(results[2]),
#                    'y': y_train})

# df_discrepencies = df[(df["model_30"] != df["model_40"]) | (df["model_30"] != df["model_50"]) | (df["model_40"] != df["model_50"])]

# len(df_discrepencies)

# np.random.seed(40)
# proj = np.random.rand(D, IMG_LEN * IMG_LEN)
# X_train_copy = copy.deepcopy(X_train)
# X_train_copy = get_scenes(X_train_copy, proj)
# predictions = classify(X_train_copy, model_40)
# acc = accuracy_score(y_train[:X_train.shape[0]], predictions)
# print(acc)
