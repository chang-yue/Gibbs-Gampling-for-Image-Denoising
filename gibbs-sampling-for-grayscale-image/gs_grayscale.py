import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import random
from sklearn.decomposition import PCA

np.random.seed(123)
random.seed(123)


sigma = 50      # Gaussian noise level
eta = 1.        # ratio of loss between y_ij and x_ij, where loss ratio between y_ij and its Markov blanket is 1
gamma = 10.     # coeficient of pixel loss (L1 or L2)
beta11 = 2.     # coeficient of gradient loss when using L1 for pixel_loss and L1 for sparse_grad_loss
beta12 = 0.025  # coeficient of gradient loss when using L1 for pixel_loss and L2 for sparse_grad_loss
beta22 = 7.5    # coeficient of gradient loss when using L2 for pixel_loss and L2 for sparse_grad_loss
beta21 = 500.   # coeficient of gradient loss when using L2 for pixel_loss and L1 for sparse_grad_loss
# lam = 1.

alpha = -200.   # coeficient of window-wise product loss 
momentum = 1.   # tradeoff between sampling and keeping, from 0 to 1. 0 means no sampling and 1 means all sampling
MAX_BURNS = 4   # number of burn-ins
MAX_SAMPLES = 2 # number of samplings
RANGE = 256     # range of pixel values
V_y = np.array(range(RANGE))


# phase 1, pixel loss
##################################################################################################
def gibbs_sampling_gaussian(X, Y, num_nei, l=2):
    height, width = Y.shape
    if num_nei == 25:
        for i in range(2, height-2):
            for j in range(2, width-2):
                markov_blanket = np.array(Y[i-2:i+3, j-2:j+3])
                #markov_blanket[2][2] = X[i][j]
                markov_blanket = np.repeat(markov_blanket.reshape([1, num_nei]), RANGE, axis=0)
                markov_blanket[:, num_nei/2] = np.arange(RANGE).T   # TODO: optimization needed, do not check whole range
                diff = markov_blanket - np.repeat(np.arange(RANGE).reshape([RANGE, 1]), num_nei, axis=1)
                l1_loss = np.sum(np.abs(diff), axis=1) + eta*np.abs(V_y - X[i][j])
                l2_loss = np.sum(np.square(diff), axis=1) + eta*np.square(V_y - X[i][j])

                if l==1: log_prob = -gamma*l1_loss
                else: log_prob = -gamma*l2_loss
                prob = np.exp(log_prob - np.max(log_prob))
                Y[i][j] = np.random.choice(RANGE, p=prob/np.sum(prob))
        return

    for i in range(1, height-1):
        for j in range(1, width-1):
            if num_nei == 5:
                markov_blanket = np.array([Y[i-1][j], Y[i][j-1], X[i][j], Y[i][j+1], Y[i+1][j]])
            elif num_nei == 9:
                markov_blanket = np.array(Y[i-1:i+2, j-1:j+2])
                #markov_blanket[1][1] = X[i][j]

            markov_blanket = np.repeat(markov_blanket.reshape([1, num_nei]), RANGE, axis=0)
            markov_blanket[:, num_nei/2] = np.arange(RANGE).T
            diff = markov_blanket - np.repeat(np.arange(RANGE).reshape([RANGE, 1]), num_nei, axis=1)
            l1_loss = np.sum(np.abs(diff), axis=1) + eta*np.abs(V_y - X[i][j])
            l2_loss = np.sum(np.square(diff), axis=1) + eta*np.square(V_y - X[i][j])
            #print np.mean(l1_loss), np.mean(l2_loss)
            if l==1: log_prob = -gamma*l1_loss
            else: log_prob = -gamma*l2_loss
            prob = np.exp(log_prob - np.max(log_prob))
            Y[i][j] = np.random.choice(RANGE, p=prob/np.sum(prob))


def denoise_gaussian_helper(noisy_img, orig_img, num_nei, l):
    Y = copy.deepcopy(noisy_img)
    height, width = Y.shape
    # burn in
    for b in range(1, MAX_BURNS+1):
        gibbs_sampling_gaussian(noisy_img, Y, num_nei, l)
        print 'burn in', b, ', ', psnr_mse(Y, orig_img)
    # sampling
    posterior = np.zeros((height, width))
    for s in range(1, MAX_SAMPLES+1):
        gibbs_sampling_gaussian(noisy_img, Y, num_nei, l)
        print 'sampling', s, ', ', psnr_mse(Y, orig_img)
        posterior += Y
    denoised = posterior / MAX_SAMPLES
    return denoised

def denoise_gaussian(noisy_img, orig_img, num_nei, l):
    h, w = orig_img.shape
    denoised = np.zeros([h,w])
    denoised = denoise_gaussian_helper(noisy_img, orig_img, num_nei, l)
    return denoised
##################################################################################################



# phase 2, sparse grad prior
##################################################################################################
def denoise_sparse_grad_prior_helper(noisy_img, orig_img, l_loss, l_grad, filter_size=3):
    Y = copy.deepcopy(noisy_img)
    height, width = Y.shape
    radius = (filter_size - 1) / 2

    def gibbs_sampling_sparse_grad_prior(X, Y, l_loss, l_grad):
        height, width = Y.shape
        for i in range(radius, height-radius):
            for j in range(radius, width-radius):
                markov_blanket_ = np.array(Y[i-radius:i+radius+1, j-radius:j+radius+1]).reshape([1, filter_size**2])
                markov_blanket = np.repeat(markov_blanket_, RANGE, axis=0)  # TODO: optimization needed, do not check whole range
                markov_blanket[:, filter_size**2/2] = np.arange(RANGE).T
                diff = markov_blanket - np.repeat(np.arange(RANGE).reshape([RANGE, 1]), filter_size**2, axis=1)

                if l_loss==1 and l_grad==1:
                    l1_loss = np.sum(np.abs(diff), axis=1) + eta*np.abs(V_y - X[i][j])
                    l1_grad = np.mean(np.abs(np.divide(diff, markov_blanket_+1.)), axis=1)
                    log_prob = -gamma*l1_loss - beta11*l1_grad

                if l_loss==2 and l_grad==2:
                    # l2_loss = np.sum(np.square(diff), axis=1) + eta*np.square(V_y - X[i][j])
                    # l2_grad = np.mean(np.square(np.divide(diff, markov_blanket_+1.)), axis=1)
                    # log_prob = -gamma*l2_loss - beta22*l2_grad
                    l2_loss = np.sum(np.square(diff), axis=1) + eta*np.square(V_y - X[i][j])
                    l2_grad = np.sum(np.log(1. + 0.5*np.square(diff/10.)), axis=1)
                    log_prob = -gamma*l2_loss - beta22*500*l2_grad

                if l_loss==2 and l_grad==1:
                    l2_loss = np.sum(np.square(diff), axis=1) + eta*np.square(V_y - X[i][j])
                    l1_grad = np.mean(np.abs(np.divide(diff, markov_blanket_+1.)), axis=1)
                    log_prob = -gamma*l2_loss - beta21*l1_grad
                    #log_prob = -np.multiply(l2_loss, l1_grad)
                    #print np.mean(gamma*l2_loss), np.mean(beta21*l1_grad)

                if l_loss==1 and l_grad==2:
                    l1_loss = np.sum(np.abs(diff), axis=1) + eta*np.abs(V_y - X[i][j])
                    l2_grad = np.mean(np.square(np.divide(diff, markov_blanket_+1.)), axis=1)
                    log_prob = -gamma*l1_loss - beta12*l2_grad
                    #print np.mean(gamma*l1_loss), np.mean(beta12*l2_grad)

                prob = np.exp(log_prob - np.max(log_prob))
                Y[i][j] = np.random.choice(RANGE, p=prob/np.sum(prob))

    # burn in
    for b in range(1, MAX_BURNS+1):
        gibbs_sampling_sparse_grad_prior(noisy_img, Y, l_loss, l_grad)
        print 'burn in', b, ', ', psnr_mse(Y, orig_img)
    # sampling
    posterior = np.zeros((height, width))
    for s in range(1, MAX_SAMPLES+1):
        gibbs_sampling_sparse_grad_prior(noisy_img, Y, l_loss, l_grad)
        print 'sampling', s, ', ', psnr_mse(Y, orig_img)
        posterior += Y
    denoised = posterior / MAX_SAMPLES
    return denoised

def denoise_sparse_grad_prior(noisy_img, orig_img, l_loss, l_grad, filter_size=3):
    h, w = orig_img.shape
    denoised = np.zeros([h,w])
    denoised = denoise_sparse_grad_prior_helper(noisy_img, orig_img, l_loss, l_grad, filter_size)
    return denoised
##################################################################################################



# phase 3, window-wise product (PCA)
##################################################################################################
def denoise_ensemble_helper(noisy_img, orig_img, sigma, filter_num=3, filter_size=3, smooth=True):
    Y = copy.deepcopy(noisy_img)
    height, width = Y.shape
    radius = (filter_size - 1) / 2
    if smooth: noisy_img_blur = cv2.GaussianBlur(noisy_img, (filter_size,filter_size), sigma)
    else: noisy_img_blur = copy.deepcopy(noisy_img)

    conv = np.zeros([(height-2*radius)*(width-2*radius), filter_size**2])
    count = 0
    for i in range(radius, height-radius):
        for j in range(radius, width-radius):
            conv[count,:] = noisy_img_blur[i-radius:i+radius+1, j-radius:j+radius+1].reshape([1, filter_size**2])
            count += 1
    pca = PCA(n_components=filter_num)
    pca.fit(conv)
    filters = pca.components_   # (filter_num, filter_size**2)
    importance = np.sqrt(pca.explained_variance_ratio_.reshape([1,filter_num]))
    importance /= np.sum(importance)    # (1, filter_num)

    def gibbs_sampling_ensemble(X, Y):
        height, width = Y.shape
        for i in range(radius, height-radius):
            for j in range(radius, width-radius):
                markov_blanket_ = np.array(Y[i-radius:i+radius+1, j-radius:j+radius+1]).reshape([1, filter_size**2])
                markov_blanket = np.repeat(markov_blanket_, RANGE, axis=0)  # TODO: optimization needed, do not check whole range
                markov_blanket[:, filter_size**2/2] = np.arange(RANGE).T
                diff = markov_blanket - np.repeat(np.arange(RANGE).reshape([RANGE, 1]), filter_size**2, axis=1)
                l2_loss = np.sum(np.square(diff), axis=1) + eta*np.square(V_y - X[i][j])
                l2_grad = np.sum(np.log(1. + 0.5*np.square(diff/10.)), axis=1)
                log_prob = -gamma*l2_loss - beta22*500*l2_grad

                markov_blanket = np.array(Y[i-radius:i+radius+1, j-radius:j+radius+1])
                markov_blanket[radius][radius] = 0
                markov_blanket = markov_blanket.reshape([1, filter_size**2])
                temp1 = np.sum(np.multiply(filters, markov_blanket), axis=1)
                temp1 = np.array([temp1.T]*RANGE)   # (RANGE, filter_num)
                temp2 = 1 + 0.5*np.square(temp1 + np.outer(V_y, filters[:,filter_size**2/2]))
                temp2 = np.sum(np.multiply(np.log(temp2), importance), axis=1)

                markov_blanket = np.array(Y[i-radius:i+radius+1, j-radius:j+radius+1])
                markov_blanket[radius][radius] = X[i][j]
                log_prob += alpha*temp2 #-lam*(V_y - np.mean(markov_blanket))**2 - gamma*compute_l1_grad(markov_blanket, filter_size)
                prob = np.exp(log_prob - np.max(log_prob))
                Y[i][j] = (1.-momentum)*Y[i][j] + momentum*np.random.choice(RANGE, p=prob/np.sum(prob))

    # burn in
    for b in range(1, MAX_BURNS+1):
        gibbs_sampling_ensemble(noisy_img, Y)
        print 'burn in', b, ', ', psnr_mse(Y, orig_img)
    # sampling
    posterior = np.zeros((height, width))
    for s in range(1, MAX_SAMPLES+1):
        gibbs_sampling_ensemble(noisy_img, Y)
        print 'sampling', s, ', ', psnr_mse(Y, orig_img)
        posterior += Y
    denoised = posterior / MAX_SAMPLES
    return denoised

def denoise_ensemble(noisy_img, orig_img, sigma, filter_num=3, filter_size=3, smooth=True):
    h, w = orig_img.shape
    denoised = np.zeros([h,w])
    denoised = denoise_ensemble_helper(noisy_img, orig_img, sigma, filter_num, filter_size, smooth)
    return denoised
##################################################################################################



def add_noise(orig_img, sigma):
    h, w = orig_img.shape
    noisy_img = np.zeros([h,w])
    noisy_img = orig_img + np.random.normal(0, sigma, [h,w])
    cv2.imwrite(img_name+'_noisy_'+str(sigma)+'.png', noisy_img)
    return noisy_img

def psnr_mse(noisy_img, orig_img):
    h, w = orig_img.shape
    mse = np.sum(np.square(noisy_img[1:h-1,1:w-1]-orig_img[1:h-1,1:w-1]))/(h-2)/(w-2)
    psnr = 10.*np.log10(np.amax(orig_img[1:h-1,1:w-1])**2 / mse)
    return psnr, mse



##################################################################################################
if __name__ == "__main__":
    # load original image
    img_name = 'birds'
    orig_img = cv2.imread(img_name+'.png', cv2.IMREAD_GRAYSCALE)
    height, width = orig_img.shape

    # add noise
    noisy_img = add_noise(orig_img, sigma)
    noisy_img[noisy_img<0] = 0
    noisy_img[noisy_img>255] = 255
    print '(PSNR, MSE) of noisy image:', psnr_mse(noisy_img, orig_img)
    noisy_img_blur = cv2.GaussianBlur(noisy_img, (3,3), sigma)
    cv2.imwrite(img_name+'_gaussian_blur_'+str(sigma)+'.png', noisy_img_blur)
    print '(PSNR, MSE) after (3,3) Gaussian blur:', psnr_mse(noisy_img_blur, orig_img)

    noisy_img_nlm = cv2.fastNlMeansDenoising(np.array(noisy_img, dtype='uint8'))
    noisy_img_nlm[noisy_img_nlm<0] = 0
    noisy_img_nlm[noisy_img_nlm>255] = 255
    cv2.imwrite(img_name+'_nlm_'+str(sigma)+'.png', noisy_img_nlm)
    print '(PSNR, MSE) after NLM:', psnr_mse(noisy_img_nlm, orig_img)

    # # try different neighbor size and norm, 9 neighbor one is the best
    # denoised_gaussian_5_1_img = denoise_gaussian(noisy_img, orig_img, 5, 1)
    # cv2.imwrite(img_name+'_denoised_gaussian_5_1_'+str(sigma)+'.png', denoised_gaussian_5_1_img)
    # print '(PSNR, MSE) of gaussian with 5 neighbors and l1-norm:', psnr_mse(denoised_gaussian_5_1_img, orig_img)

    # denoised_gaussian_9_1_img = denoise_gaussian(noisy_img, orig_img, 9, 1)
    # cv2.imwrite(img_name+'_denoised_gaussian_9_1_'+str(sigma)+'.png', denoised_gaussian_9_1_img)
    # print '(PSNR, MSE) of gaussian with 9 neighbors and l1-norm:', psnr_mse(denoised_gaussian_9_1_img, orig_img)

    # denoised_gaussian_25_1_img = denoise_gaussian(noisy_img, orig_img, 25, 1)
    # cv2.imwrite(img_name+'_denoised_gaussian_25_1_'+str(sigma)+'.png', denoised_gaussian_25_1_img)
    # print '(PSNR, MSE) of gaussian with 25 neighbors and l1-norm:', psnr_mse(denoised_gaussian_25_1_img[1:height-1,1:width-1,:], orig_img[1:height-1,1:width-1,:])

    # denoised_gaussian_5_2_img = denoise_gaussian(noisy_img, orig_img, 5, 2)
    # cv2.imwrite(img_name+'_denoised_gaussian_5_2_'+str(sigma)+'.png', denoised_gaussian_5_2_img)
    # print '(PSNR, MSE) of gaussian with 5 neighbors and l2-norm:', psnr_mse(denoised_gaussian_5_2_img, orig_img)

    # denoised_gaussian_9_2_img = denoise_gaussian(noisy_img, orig_img, 9, 2)
    # cv2.imwrite(img_name+'_denoised_gaussian_9_2_'+str(sigma)+'.png', denoised_gaussian_9_2_img)
    # print '(PSNR, MSE) of gaussian with 9 neighbors and l2-norm:', psnr_mse(denoised_gaussian_9_2_img, orig_img)

    # denoised_gaussian_25_2_img = denoise_gaussian(noisy_img, orig_img, 25, 2)
    # cv2.imwrite(img_name+'_denoised_gaussian_25_2_'+str(sigma)+'.png', denoised_gaussian_25_2_img)
    # print '(PSNR, MSE) of gaussian with 25 neighbors and l2-norm:', psnr_mse(denoised_gaussian_25_2_img[1:height-1,1:width-1,:], orig_img[1:height-1,1:width-1,:])

    # # with sparse gradient prior
    # denoised_sparse_grad_prior_1_1_img = denoise_sparse_grad_prior(noisy_img, orig_img, 1, 1, 3)
    # cv2.imwrite(img_name+'_denoised_sparse_grad_prior_1_1_'+str(sigma)+'.png', denoised_sparse_grad_prior_1_1_img)
    # print '(PSNR, MSE) of l1-loss and l1-grad:', psnr_mse(denoised_sparse_grad_prior_1_1_img, orig_img)

    # denoised_sparse_grad_prior_2_2_img = denoise_sparse_grad_prior(noisy_img, orig_img, 2, 2, 3)
    # cv2.imwrite(img_name+'_denoised_sparse_grad_prior_2_2_'+str(sigma)+'.png', denoised_sparse_grad_prior_2_2_img)
    # print '(PSNR, MSE) of l2-loss and l2-grad:', psnr_mse(denoised_sparse_grad_prior_2_2_img, orig_img)

    # denoised_sparse_grad_prior_2_1_img = denoise_sparse_grad_prior(noisy_img, orig_img, 2, 1, 3)
    # cv2.imwrite(img_name+'_denoised_sparse_grad_prior_2_1_'+str(sigma)+'.png', denoised_sparse_grad_prior_2_1_img)
    # print '(PSNR, MSE) of l2-loss and l1-grad:', psnr_mse(denoised_sparse_grad_prior_2_1_img, orig_img)

    # denoised_sparse_grad_prior_1_2_img = denoise_sparse_grad_prior(noisy_img, orig_img, 1, 2, 3)
    # cv2.imwrite(img_name+'_denoised_sparse_grad_prior_1_2_'+str(sigma)+'.png', denoised_sparse_grad_prior_1_2_img)
    # print '(PSNR, MSE) of l1-loss and l2-grad:', psnr_mse(denoised_sparse_grad_prior_1_2_img, orig_img)

    # # with pca filter prior
    # denoised_filter_pca_3_img = denoise_filter_pca(noisy_img, orig_img, sigma, 3)
    # cv2.imwrite(img_name+'_denoised_filter_pca_3_'+str(sigma)+'.png', denoised_filter_pca_3_img)
    # print '(PSNR, MSE) of 3 pca filter:', psnr_mse(denoised_filter_pca_3_img, orig_img)

    # # ensemble
    denoised_ensemble_img = denoise_ensemble(noisy_img, orig_img, sigma, 3)
    cv2.imwrite(img_name+'_denoised_ensemble_'+str(sigma)+'.png', denoised_ensemble_img)
    print '(PSNR, MSE) of ensemble:', psnr_mse(denoised_ensemble_img, orig_img)

