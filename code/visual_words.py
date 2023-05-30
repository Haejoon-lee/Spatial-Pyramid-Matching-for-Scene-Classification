import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans

from scipy.ndimage import gaussian_filter, gaussian_laplace
from scipy.spatial import distance #1.3


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    filter_scales = opts.filter_scales
    
    # ----- TODO -----
    if img.ndim == 2:     #Gray image
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    img = skimage.color.rgb2lab(img)
    img_min = img.min(axis=(0, 1), keepdims=True)
    img_max = img.max(axis=(0, 1), keepdims=True)
    img = (img - img_min) / (img_max - img_min)

    n_scale = len(filter_scales)
    H = img.shape[0]
    W = img.shape[1]
    filter_responses = np.empty([H, W, 3*4*n_scale], dtype = img.dtype)

    for j in range(n_scale):   #F = n_scale * 4
        for k in range(3):
            filter_responses[:, :, 4*j * 3 + k] = gaussian_filter(img[:,:,k], sigma=filter_scales[j])
        for k in range(3):
            filter_responses[:, :, (4*j+1) * 3 + k] = gaussian_laplace(img[:,:,k], sigma=filter_scales[j])
        for k in range(3):
            filter_responses[:, :, (4*j+2) * 3 + k] = gaussian_filter(img[:,:,k], sigma=filter_scales[j], order = (0, 1))
        for k in range(3):
            filter_responses[:, :, (4*j+3) * 3 + k] = gaussian_filter(img[:,:,k], sigma=filter_scales[j], order = (1, 0))
    
    return filter_responses

    # pass

def compute_dictionary_one_image(filter_responses, alpha):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """

    # ----- TODO -----
    filter_responses_sampled = filter_responses.reshape(-1, filter_responses.shape[2]) #Reshape to (H*W, 3*4*n_scale) 
    
    #Sample alpha random pixels
    filter_responses_sampled = filter_responses_sampled[np.random.randint(filter_responses_sampled.shape[0], size=alpha), :]

    return filter_responses_sampled

def compute_dictionary(opts, n_worker=1): 

    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha
    filter_scales = opts.filter_scales
    
    train_files = open(join(data_dir, "train_files.txt")).read().splitlines() #image list
    # ----- TODO -----

    #Load a image
    filter_responses_sampled_all = []

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
        
        filter_responses = extract_filter_responses(opts, img)
        # filter_responses = extract_filter_responses(filter_scales, img)
        filter_responses_sampled = compute_dictionary_one_image(filter_responses, alpha)
        filter_responses_sampled_all.append(filter_responses_sampled)
        if i%100 == 0:
            print('Computing Dictionary: {}/{} files were done'.format(i, len(train_files)))


    filter_responses_sampled_all = np.vstack(filter_responses_sampled_all)
    
    kmeans = KMeans(n_clusters=K).fit(filter_responses_sampled_all)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    
    return filter_responses_sampled_all, dictionary

def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img's each pixel using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    # ----- TODO -----
    filter_scales = opts.filter_scales
    # filter_scales = [1, 2, 4, 8] #should be opt.~
    filter_responses = extract_filter_responses(opts, img)
    H = img.shape[0]
    W = img.shape[1]
    wordmap =  np.empty([H, W], dtype = img.dtype)
    
    for i in range(H):
        for j in range(W):
            wordmap[i,j] = distance.cdist(np.array([filter_responses[i,j,:]]), dictionary, metric='euclidean').argmin()

    return wordmap
