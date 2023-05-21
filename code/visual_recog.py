import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

from visual_words import * 


def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    # K=10

    # ----- TODO -----
    hist, bins= np.histogram(wordmap, bins = range(K+1))

    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    K = opts.K
    L = opts.L #For the beginning, 2 is recommended
    # ----- TODO -----
    # L=2
    # K=10
    
    smallest_cell_hight = wordmap.shape[0]//2**L 
    smallest_cell_width = wordmap.shape[1]//2**L 

    last_layer =  np.empty([2**L, 2**L, K], dtype = wordmap.dtype)
    last_layer_hist_vector =  np.empty([0], dtype = wordmap.dtype)

    hist_all =  np.empty([0], dtype = wordmap.dtype)

    for i in range(last_layer.shape[0]):
        for j in range(last_layer.shape[1]):
            last_layer[i,j,:] = get_feature_from_wordmap(opts,\
                                wordmap[smallest_cell_hight*i:smallest_cell_hight*(i+1),\
                                        smallest_cell_width*(j):smallest_cell_width*(j+1)])
            last_layer_hist_vector = np.concatenate([last_layer_hist_vector, last_layer[i,j,:]])

    if L<=1:
        last_layer_hist_vector *= 2**(-L)
    else:
        last_layer_hist_vector *= 2**(-1)

    for i in range(L):
        group = 2**(L-i) #num of the smallest cell for upper layers on 
        for m in range((last_layer.shape[0]//group)):
            for n in range((last_layer.shape[1]//group)):
                upper_cell_hist = last_layer[m*group:(m+1)*group, n*group:(n+1)*group, :].sum(axis=(0,1)) #sum of each cell

                if i<=1:
                    upper_cell_hist /= 2**(L)
                else:
                    upper_cell_hist /= 2**(-i+L+1)

                hist_all = np.concatenate([hist_all, upper_cell_hist])
#                 print(upper_cell_hist)
    hist_all = np.concatenate([hist_all, last_layer_hist_vector])
    hist_all = hist_all/np.linalg.norm(hist_all, ord=1)

    return hist_all

def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """
    # ----- TODO -----

    img = Image.open(img_path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255
    
    # smaller_side = min(img.shape[0:2])
    # bigger_side = max(img.shape[0:2])
    # side_indx = img.shape.index(smaller_side)
    
    # smaller_side = min(img.shape[0:2])
    # bigger_side = max(img.shape[0:2])
    # side_indx = img.shape.index(smaller_side)

    # if smaller_side == bigger_side:
    #     img_crop = img
    # else:
    #     if side_indx == 0:
    #         img_crop = img[:,(bigger_side-smaller_side)//2:-(bigger_side-smaller_side)//2]
    #     else:
    #         img_crop = img[(bigger_side-smaller_side)//2:-(bigger_side-smaller_side)//2,:]

    #     if img_crop.ndim == 2:
    #         img_crop = np.repeat(img_crop[:, :, np.newaxis], 3, axis=2)

    # img = resize(img_crop, (512, 512))

    wordmap = get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    
    return feature

def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    # data_dir = '../data'
    # out_dir = '.'
    # SPM_layer_num = 2


    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))

    # ----- TODO -----
    wordmap_list = []
    features = []
    for i in range(len(train_files)):
        img_file = train_files[i]
        img_path = join(data_dir, img_file)
        # img = Image.open(img_path)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255
        ##my idea
        # smaller_side = min(img.shape[0:2])
        # bigger_side = max(img.shape[0:2])
        # side_indx = img.shape.index(smaller_side)

        # if smaller_side == bigger_side:
        #     img_crop = img
        # else:
        #     if side_indx == 0:
        #         img_crop = img[:,(bigger_side-smaller_side)//2:-(bigger_side-smaller_side)//2]
        #     else:
        #         img_crop = img[(bigger_side-smaller_side)//2:-(bigger_side-smaller_side)//2,:]

        #     if img_crop.ndim == 2:
        #         img_crop = np.repeat(img_crop[:, :, np.newaxis], 3, axis=2)

        # img = resize(img_crop, (512, 512))
        feature = get_image_feature(opts,img_path, dictionary) #Extracts the spatial pyramid matching feature.
        features.append(feature)
        if i%100 == 0:
            print('Building system: {}/{} files were done'.format(i, len(train_files)))
    features = np.vstack(features)

    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    
    return features


def similarity_to_set(word_hist, histograms):
    """
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """

    # ----- TODO -----
    minima = np.minimum(word_hist, histograms)
    sim = np.sum(minima, axis=1)
    sim_norm = np.true_divide(np.sum(minima, axis=1), np.sum(histograms, axis=1))
    dist = 1-sim_norm

    return dist

def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    # K = dictionary.shape[0]
    # L = trained_system["SPM_layer_num"]
    
    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)
    train_labels = trained_system["labels"]
    features = trained_system["features"]

    # ----- TODO -----
    conf =  np.zeros([8, 8])

    for i in range(len(test_files)):
    # for i in range(100):
        img_file = test_files[i]
        img_path = join(data_dir, img_file)
        # img = Image.open(img_path)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255
        
            ##my idea
        # smaller_side = min(img.shape[0:2])
        # bigger_side = max(img.shape[0:2])
        # side_indx = img.shape.index(smaller_side)

        # if smaller_side == bigger_side:
        #     img_crop = img
        # else:
        #     if side_indx == 0:
        #         img_crop = img[:,(bigger_side-smaller_side)//2:-(bigger_side-smaller_side)//2]
        #     else:
        #         img_crop = img[(bigger_side-smaller_side)//2:-(bigger_side-smaller_side)//2,:]

        #     if img_crop.ndim == 2:
        #         img_crop = np.repeat(img_crop[:, :, np.newaxis], 3, axis=2)

        # img = resize(img_crop, (512, 512))
        feature_test = get_image_feature(test_opts, img_path, dictionary)
        dist_list = similarity_to_set(feature_test, features)
        pred = train_labels[dist_list.argmin()]
        conf[test_labels[i], pred] +=1
        if i%100 == 0:
            print('Inference: {}/{} files were done'.format(i, len(test_files)))

    accuracy = np.trace(conf)/conf.sum()
    return conf, accuracy

def compute_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass

def evaluate_recognition_System_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass