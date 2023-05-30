from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts

import math

def main():
    opts = get_opts()
    print(opts)
    # # Q1.1
    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img = Image.open(img_path)
    img = Image.open(img_path).convert("RGB")  #Discard the last channel of RGBA to RGB  
    img = np.array(img).astype(np.float32) / 255

    ##===My idea for squaring every image into the same size(512, 512) for performance improvement===##
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


    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)

    # # Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)

    # Q1.3

    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'dictionary-copy1.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    util.visualize_wordmap(wordmap)

    # # Q2.1-2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    print(conf)
    print(accuracy)

    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
