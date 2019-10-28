import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import random

np.random.seed(123)
random.seed(123)


flip_rate = 0.2     # prob to invert pixel value, MAX_BURNS and MAX_SAMPLES should be adjusted on this
eta = 1.            # coeficient of loss between y_ij and x_ij
beta = 2.           # coeficient of loss between y_ij and its four neighboring pixel values in y
MAX_BURNS = 15      # number of burn-ins
MAX_SAMPLES = 5     # number of samplings
max_value = 255


def gibbs_sampling(X, Y):
    height, width = Y.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            markov_blanket = [Y[i-1][j], Y[i][j-1], Y[i][j+1], Y[i+1][j], X[i][j]]
            prob = 1. / (1. + np.exp(-2.*eta*markov_blanket[4] - 2.*beta*sum(markov_blanket[:4])))
            if np.random.rand() < prob:
                Y[i][j] = 1
            else: Y[i][j] = -1
    # indeces = [idx for idx,_ in np.ndenumerate(np.zeros((height-2, width-2)))]
    # random.shuffle(indeces)
    # for i,j in indeces:
    #     markov_blanket = [Y[i-1][j], Y[i][j-1], Y[i][j+1], Y[i+1][j], X[i][j]]
    #     prob = 1. / (1. + np.exp(-2.*eta*markov_blanket[4] - 2.*beta*sum(markov_blanket[:4])))
    #     if np.random.rand() < prob:
    #         Y[i][j] = 1
    #     else: Y[i][j] = -1


def denoise(noisy_img):
    Y = copy.deepcopy(noisy_img)
    height, width = Y.shape
    # burn in
    for b in range(1, MAX_BURNS+1):
        print('burn in', b)
        gibbs_sampling(noisy_img, Y)
    # sampling
    posterior = np.zeros((height, width))
    for s in range(1, MAX_SAMPLES+1):
        print('sampling', s)
        gibbs_sampling(noisy_img, Y)
        posterior[np.where(np.array(Y)==1)] += 1
    denoised = np.ones(posterior.shape)
    denoised[np.where(posterior<.5)] = -1
    return denoised


def load_img(filename):
    orig_img = plt.imread(filename)
    height, width = orig_img.shape
    X = np.ones([height+2, width+2])
    for i in range(height):
        for j in range(width):
            if orig_img[i,j] == 0:
                X[i+1,j+1] = -1
            # if np.sum(orig_img[i,j]) < 2.:
            #     X[i+1,j+1] = -1
    return X


def add_noise(orig_img, flip_rate):
    height, width = orig_img.shape
    Y = copy.deepcopy(orig_img)
    for i in range(1, height-1):
        for j in range(1, width-1):
            if np.random.rand() < flip_rate:
                Y[i,j] = -Y[i,j]
    return Y


def save_as_png(imgi, title):
    # plt.imshow(imgi, cmap=plt.cm.gray)
    # plt.savefig(title + '_.png')
    img = copy.deepcopy(imgi)
    img[img > 1] = 1
    img[img < -1] = -1
    # ret1,th1 = cv2.threshold(img,0,1,cv2.THRESH_BINARY)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == -1: img[i,j] = 0
            else: img[i,j] = max_value
    cv2.imwrite(title + '.png', img)


def get_error(denoised_img, orig_img):
    height, width = denoised_img.shape
    diff = denoised_img[1:height-1, 1:width-1] - orig_img[1:height-1, 1:width-1]
    return 1.*np.count_nonzero(diff)/(height-2)/(width-2)


if __name__ == "__main__":
    img_name = 'einstein_equation'
    orig_img = load_img(img_name+'.png')
    #orig_img = cv2.imread(img_name+'.png', cv2.THRESH_BINARY)
    #save_as_png(orig_img, img_name+'_orig')
    #print ((orig_img*max_value+max_value)/2)
    save_as_png(orig_img, img_name+'_orig')
    noisy_img = add_noise(orig_img, flip_rate)
    save_as_png(noisy_img, img_name+'_noisy_'+str(flip_rate))
    denoised_img = denoise(noisy_img)
    print('flip rate = ' + str(flip_rate) + '; denoised error: {:1.3f} %'.format(100*get_error(denoised_img, orig_img)))
    save_as_png(denoised_img, img_name+'_denoised_'+str(flip_rate))

