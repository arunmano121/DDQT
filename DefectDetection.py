#!/usr/bin/env python3

# IO modules to help with reading in Matlab data
import scipy.io
# plotting tools
import matplotlib.pyplot as plt
# numerical python library to perform array operations
import numpy as np
# perform statistical operations using scipy
import scipy.stats as sp
# used to pre-filter data using wavelets
import pywt
# scitkit-learn library tools to perform machine learning operations
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
# Point class object and function to evaluate if point lies within polygon
from point_in_convex_polygon import Point, is_within_polygon


def annotate_plots(ax, defs):
    '''\nAnnotate charts with locations of defects'''

    for i in range(1, len(defs)+1):
        ax.plot(defs[i]['s1_vertices'], defs[i]['s2_vertices'], color='red')
        ax.annotate(defs[i]['name'], xy=(np.mean(defs[i]['s1_vertices']),
                                         np.mean(defs[i]['s2_vertices'])))

    return


def defect_detection_metrics(mat, mah, iso, s1_2d, s2_2d,
                             defs, t_stamps, t, units, plot=True):
    '''\nQuantification of defect detection'''

    # total number of defects
    num_defs = len(defs)

    # y_truth holds the true labels of the defective areas
    # location of defects are labeled with 1
    y_truth = np.zeros(mat.shape)

    for i in range(1, num_defs+1):
        y_truth[:, defs[i]['s1_sel'], defs[i]['s2_sel']] = 1

    # counter holds the defect number and t_idx contains the time index
    # of the defect
    for counter, t_idx in enumerate(t_stamps):

        # Compute ROC curve and ROC area for raw data
        fpr_mat, tpr_mat, _ = \
            roc_curve(
                y_truth[t_idx, :, :].ravel(), mat[t_idx, :, :].ravel())
        auc_mat = auc(fpr_mat, tpr_mat)

        # index where fpr is closest to 0.02 (2%)
        fpr_p2_mat_idx = abs(np.array(fpr_mat - 0.02)).argmin()
        # index where fpr is closest to 0.05 (5%)
        fpr_p5_mat_idx = abs(np.array(fpr_mat - 0.05)).argmin()
        # index where fpr is closest to 0.10 (10%)
        fpr_p10_mat_idx = abs(np.array(fpr_mat - 0.10)).argmin()

        # find tpr at those indices
        tpr_p2_mat = tpr_mat[fpr_p2_mat_idx]
        tpr_p5_mat = tpr_mat[fpr_p5_mat_idx]
        tpr_p10_mat = tpr_mat[fpr_p10_mat_idx]

        # Compute ROC curve and ROC area for outliers computed using
        # Mahalanobis distance
        fpr_mah, tpr_mah, _ = \
            roc_curve(
                y_truth[t_idx, :, :].ravel(), mah[t_idx, :, :].ravel())
        auc_mah = auc(fpr_mah, tpr_mah)

        # index where fpr is closest to 0.02 (2%)
        fpr_p2_mah_idx = abs(np.array(fpr_mah - 0.02)).argmin()
        # index where fpr is closest to 0.05 (5%)
        fpr_p5_mah_idx = abs(np.array(fpr_mah - 0.05)).argmin()
        # index where fpr is closest to 0.10 (10%)
        fpr_p10_mah_idx = abs(np.array(fpr_mah - 0.10)).argmin()

        # find tpr at those indices
        tpr_p2_mah = tpr_mah[fpr_p2_mah_idx]
        tpr_p5_mah = tpr_mah[fpr_p5_mah_idx]
        tpr_p10_mah = tpr_mah[fpr_p10_mah_idx]

        # Compute ROC curve and ROC area for outliers computed using
        # Isolation Forest
        fpr_iso, tpr_iso, _ = \
            roc_curve(
                y_truth[t_idx, :, :].ravel(), iso[t_idx, :, :].ravel())
        auc_iso = auc(fpr_iso, tpr_iso)

        # index where fpr is closest to 0.02 (2%)
        fpr_p2_iso_idx = abs(np.array(fpr_iso - 0.02)).argmin()
        # index where fpr is closest to 0.05 (5%)
        fpr_p5_iso_idx = abs(np.array(fpr_iso - 0.05)).argmin()
        # index where fpr is closest to 0.10 (10%)
        fpr_p10_iso_idx = abs(np.array(fpr_iso - 0.10)).argmin()

        # find tpr at those indices
        tpr_p2_iso = tpr_iso[fpr_p2_iso_idx]
        tpr_p5_iso = tpr_iso[fpr_p5_iso_idx]
        tpr_p10_iso = tpr_iso[fpr_p10_iso_idx]

        # Print results
        print('Time Index: %d' % (t_idx))
        print('FPR range:0.00-1.00  AUC Raw: %0.2f'
              '  AUC Mah: %0.2f  AUC Iso: %0.2f'
              % (auc_mat, auc_mah, auc_iso))

        print('FPR:0.02  TPR Raw: %0.2f  TPR Mah: %0.2f  TPR Iso: %0.2f'
              % (tpr_p2_mat, tpr_p2_mah, tpr_p2_iso))
        print('FPR:0.05  TPR Raw: %0.2f  TPR Mah: %0.2f  TPR Iso: %0.2f'
              % (tpr_p5_mat, tpr_p5_mah, tpr_p5_iso))
        print('FPR:0.10  TPR Raw: %0.2f  TPR Mah: %0.2f  TPR Iso: %0.2f'
              % (tpr_p10_mat, tpr_p10_mah, tpr_p10_iso))

        # plot results based on plot boolean flag
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=3)

            fig.suptitle('Result of Outlier analysis at %0.1f%s'
                         % (t[t_idx], units['t_units']))

            # raw data
            cs = ax[0].contourf(s1_2d, s2_2d, mat[t_idx, :, :])
            ax[0].plot(0, 0)
            ax[0].set(title='Raw data')
            annotate_plots(ax[0], defs)
            fig.colorbar(cs, ax=ax[0], shrink=0.33)

            # outliers computed using Mahalanobis distance
            cs = ax[1].contourf(s1_2d, s2_2d, mah[t_idx, :, :])
            ax[1].plot(0, 0)
            ax[1].set(title='Outliers using Mahalanobis distance')
            annotate_plots(ax[1], defs)
            fig.colorbar(cs, ax=ax[1], shrink=0.33)

            # outliers computed using Isolation Forest
            cs = ax[2].contourf(s1_2d, s2_2d, iso[t_idx, :, :])
            ax[2].plot(0, 0)
            ax[2].set(title='Isolation Forest')
            annotate_plots(ax[2], defs)
            fig.colorbar(cs, ax=ax[2], shrink=0.33)

            plt.tight_layout()
            plt.show()

            # plots of ROC curves
            fig, ax = plt.subplots(nrows=1, ncols=2)
            fig.suptitle('ROC curves')

            # FPR range 0.00-1.00
            ax[0].plot(fpr_mat, tpr_mat, color='darkorange',
                       lw=2, label='Raw ROC curve (area = %0.2f)'
                       % (auc_mat))
            ax[0].plot(fpr_mah, tpr_mah, color='green',
                       lw=2, label='Mah ROC curve (area = %0.2f)'
                       % (auc_mah))
            ax[0].plot(fpr_iso, tpr_iso, color='red',
                       lw=2, label='Iso ROC curve (area = %0.2f)'
                       % (auc_iso))
            ax[0].plot([0, 1], [0, 1], color='navy', lw=2,
                       linestyle='--')
            ax[0].set_xlim(0.0, 1.0)
            ax[0].set_ylim(0.0, 1.05)
            ax[0].legend(loc='lower right')
            ax[0].set_xlabel('False Positive Rate')
            ax[0].set_ylabel('True Positive Rate')

            # FPR range 0.00-0.10
            ax[1].plot(fpr_mat, tpr_mat, color='darkorange',
                       lw=2, label='Raw ROC curve')
            ax[1].plot(fpr_mah, tpr_mah, color='green',
                       lw=2, label='Mah ROC curve')
            ax[1].plot(fpr_iso, tpr_iso, color='red',
                       lw=2, label='Iso ROC curve')
            ax[1].plot([0, 1], [0, 1], color='navy', lw=2,
                       linestyle='--')
            ax[1].set_xlim(0.0, 0.10)
            ax[1].legend(loc='lower right')
            ax[1].set_ylim(0.0, 1.05)
            ax[1].set_xlabel('False Positive Rate')
            ax[1].set_ylabel('True Positive Rate')

            plt.title('ROC curve - Time: %0.1f%s'
                      % (t[t_idx], units['t_units']))
            plt.show()

    return


def scale_frames(arr, t_stamps):
    '''\nScale frames between 0-1 for better interpretability'''

    outarr = np.zeros(arr.shape)
    # iterate through time stamps
    for t_idx in t_stamps:
        # slice of data at time stamp
        j_mat = arr[t_idx, :, :]

        # min and max of slice to help with scaling
        j_min = np.min(j_mat)
        j_max = np.max(j_mat)

        # scale data between 0-1
        outarr[t_idx, :, :] = (j_mat - j_min)/(j_max - j_min)

    return outarr


def fit_isolationforest_model(features, t_stamps, pca_var):
    '''\nFit Isolation Forest model'''

    # initialize the model with 15% outliers
    clf = IsolationForest(contamination=0.15)

    # create an empty numpy array to hold results
    shape = features[list(features.keys())[0]].shape
    iso = np.zeros(shape)

    # iterate through the time stamps
    for t_idx in t_stamps:
        # create an empty numpy array to hold input features data
        X = np.zeros([shape[1] * shape[2], len(features)])

        # fill the array with features at different columns
        for counter, feature in enumerate(features):
            X[:, counter] = np.ravel(features[feature][t_idx, :, :])

        # if PCA is required the parameter will be less than 1
        if pca_var < 1.0:
            # Standardizing the features before performing PCA
            X = StandardScaler().fit_transform(X)
            # perform PCA to reduce dimensionality
            pca = PCA(n_components=pca_var)
            X = pca.fit_transform(X)

            # if PCA is performed
            # number of components needed to explain variance
            print('PCA: Explained variance: %0.2f%% \nfeatures reqd: %d' %
                  (pca_var*100, X.shape[1]))

        # fit the Isolation Forest model
        clf.fit(X)

        # predict outliers
        # multiply by -1 to flip labels and be consistent with mah and mat
        # does not affect any other quantitative results
        val = -1 * clf.decision_function(X)

        iso[t_idx, :, :] = val.reshape((shape[1], shape[2]))

    return iso


def outlier_mah(features, t_stamps, pca_var):
    '''\nMahalanobis distance to identify outliers'''

    # create an empty numpy array to hold results
    shape = features[list(features.keys())[0]].shape

    mah = np.zeros(shape)

    # iterate through the time stamps
    for t_idx in t_stamps:
        # create an empty numpy array to hold input features data
        X = np.zeros([shape[1] * shape[2], len(features)])

        # fill the array with features at different columns
        for counter, feature in enumerate(features):
            X[:, counter] = np.ravel(features[feature][t_idx, :, :])

        # if PCA is required the parameter will be less than 1
        if pca_var < 1.0:
            # Standardizing the features before performing PCA
            X = StandardScaler().fit_transform(X)
            # perform PCA to reduce dimensionality
            pca = PCA(n_components=pca_var)
            X = pca.fit_transform(X)

            # if PCA is performed
            # number of components needed to explain variance
            print('PCA: Explained variance: %0.2f%% \nfeatures reqd: %d' %
                  (pca_var*100, X.shape[1]))

        # compute mean, covariance and covariance inverse
        mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov_inv = np.linalg.inv(cov)

        # regular method - but computationally intensive
        # t = np.matmul(np.matmul((X_slice - mu), cov_inv),
        #               np.transpose((X_slice - mu)))
        # val = np.sqrt(np.diag(t))

        # alternate method to speed-up significantly ~ 150x
        # source - https://stackoverflow.com/questions/27686240/
        # calculate-mahalanobis-distance-using-numpy-only

        delta = X - mu
        val = np.sqrt(np.einsum('nj,jk,nk->n', delta, cov_inv, delta))

        # store the Mahalanobis distance
        mah[t_idx, :, :] = val.reshape((shape[1], shape[2]))

    return mah


def normalize_features(features, t_stamps):
    '''\nNormalize features'''

    # iterate through the features
    for feature in features:
        # iterate through the time stamps
        for t_idx in t_stamps:
            # extract feature at timestamp
            j_feature = features[feature][t_idx, :, :]

            # compute the 1st and 99th percentile
            p1 = np.percentile(j_feature, 1)
            p99 = np.percentile(j_feature, 99)

            # capping at 1st and 99th percentile to avoid extreme values
            j_feature[j_feature <= p1] = p1
            j_feature[j_feature >= p99] = p99

            # calculate min and max of the features to scale
            j_min = np.min(j_feature)
            j_max = np.max(j_feature)

            features[feature][t_idx, :, :] = \
                (j_feature - j_min) / (j_max - j_min)

    return features


def combine_features(feature_list):
    '''\nCombine all features from different methods into one single dict'''

    # placeholder dictionary to hold all the features
    features = {}

    for i in range(len(feature_list)):
        features = {**feature_list[i], **features}

    return features


def visualize_features(mat, features, s1_2d, s2_2d, feature,
                       t_idx, t, units):
    '''\nVisualize computed features'''

    fig, ax = plt.subplots(nrows=1, ncols=2)

    fig.suptitle('Comparison of raw signal and feature at %0.1f%s'
                 % (t[t_idx], units['t_units']))

    # contour plot of raw signal
    ax[0].contourf(s1_2d, s2_2d, mat[t_idx, :, :])
    ax[0].set(title='Raw signal')

    # contour plot of feature
    ax[1].contourf(s1_2d, s2_2d, features[feature][t_idx, :, :])
    ax[1].set(title='Feature - ' + feature)

    plt.tight_layout()
    plt.show()

    return


def compute_features_wav(mat, t_stamps):
    '''\nCalculates wavelet transformed features at every location'''

    # initialize an empty dictionary to hold and return all features
    features_wav = {}

    # ll is reconstructed from approximation
    # lh is reconstructed from radial detail
    # hl is reconstructed from theta detail
    # hh is reconstructed from diagonal detail

    features_wav['ll'] = np.zeros(mat.shape)
    features_wav['lh'] = np.zeros(mat.shape)
    features_wav['hl'] = np.zeros(mat.shape)
    features_wav['hh'] = np.zeros(mat.shape)

    # iterating through the time stamps
    for t_idx in t_stamps:
        # wavelet decomposition
        LL, (LH, HL, HH) = pywt.dwt2(mat[t_idx, :, :], 'bior1.3')

        # inverse wavelet transform
        features_wav['ll'][t_idx, :, :] = \
            pywt.idwt2((LL, (None, None, None)), 'bior1.3',
                       mode='symmetric')[1:, :]

        features_wav['lh'][t_idx, :, :] = \
            pywt.idwt2((None, (LH, None, None)), 'bior1.3',
                       mode='symmetric')[1:, :]

        features_wav['hl'][t_idx, :, :] = \
            pywt.idwt2((None, (None, HL, None)), 'bior1.3',
                       mode='symmetric')[1:, :]

        features_wav['hh'][t_idx, :, :] = \
            pywt.idwt2((None, (None, None, HH)), 'bior1.3',
                       mode='symmetric')[1:, :]

    return features_wav


def compute_features_td(mat, t_stamps):
    '''\nCalculates temporal features at every spatial location'''

    # all these time domain features are calculated at every spatial point
    # in the time domain

    # initialize an empty dictionary to hold and return all features
    features_td = {}

    # a small constant to prevent divide by zeros
    eps = 1e-7

    # z val compares the distance of current obs from the mean of
    # 10 prior or 100 prior obs
    features_td['val_z10'] = np.zeros(mat.shape)
    features_td['val_z100'] = np.zeros(mat.shape)

    # ratio of current obs compared to median of 10 or 100 prior obs
    features_td['val_med10'] = np.zeros(mat.shape)
    features_td['val_med100'] = np.zeros(mat.shape)

    # ratio of current obs compared to mean of 10 or 100 prior obs
    features_td['val_mean10'] = np.zeros(mat.shape)
    features_td['val_mean100'] = np.zeros(mat.shape)

    # ratio of mean of 10 obs to mean of 100 obs
    features_td['rat_mean_10_100'] = np.zeros(mat.shape)

    # ratio of stdev of 10 obs to stdev of 100 obs
    features_td['rat_std_10_100'] = np.zeros(mat.shape)

    # ratio of median of 10 obs to median of 100 obs
    features_td['rat_med_10_100'] = np.zeros(mat.shape)

    # ratio of skew of 10 obs to skew of 100 obs
    features_td['rat_skew_10_100'] = np.zeros(mat.shape)

    # ratio of kurtosis of 10 obs to kurtosis of 100 obs
    features_td['rat_kurtosis_10_100'] = np.zeros(mat.shape)

    # ratio of median of 10 obs to mean of 10 obs
    features_td['rat_med_mean_10'] = np.zeros(mat.shape)

    # ratio of median of 100 obs to mean of 100 obs
    features_td['rat_med_mean_100'] = np.zeros(mat.shape)

    # ratio of skew of 10 obs to kurtosis of 10 obs
    features_td['rat_skew_kurtosis_10'] = np.zeros(mat.shape)

    # ratio of skew of 100 obs to kurtosis of 100 obs
    features_td['rat_skew_kurtosis_100'] = np.zeros(mat.shape)

    # uniform average value of the 5 most recent obs
    features_td['smooth_uni_5'] = np.zeros(mat.shape)

    # weighted verage value of the 5 most recent obs
    features_td['smooth_wt_5'] = np.zeros(mat.shape)

    # uniform average value of the 25 most recent obs
    features_td['smooth_uni_25'] = np.zeros(mat.shape)

    # iterating through the other time stamps
    for t_idx in t_stamps:
        j_slice = mat[t_idx, :, :]  # current time slice
        j_mean10 = np.mean(mat[t_idx-10:t_idx, :, :], axis=0)
        j_std10 = np.std(mat[t_idx-10:t_idx, :, :], axis=0)
        j_med10 = np.median(mat[t_idx-10:t_idx, :, :], axis=0)
        j_mean100 = np.mean(mat[t_idx-100:t_idx, :, :], axis=0)
        j_std100 = np.std(mat[t_idx-100:t_idx, :, :], axis=0)
        j_med100 = np.median(mat[t_idx-100:t_idx, :, :], axis=0)
        j_skew10 = sp.skew(mat[t_idx-10:t_idx, :, :], axis=0)
        j_skew100 = sp.skew(mat[t_idx-100:t_idx, :, :], axis=0)
        j_kurtosis10 = sp.kurtosis(mat[t_idx-10:t_idx, :, :], axis=0)
        j_kurtosis100 = sp.kurtosis(mat[t_idx-100:t_idx, :, :], axis=0)

        # adding eps to avoid division by 0
        j_mean10 += eps
        j_std10 += eps
        j_med10 += eps
        j_mean100 += eps
        j_std100 += eps
        j_med100 += eps
        j_skew10 += eps
        j_skew100 += eps
        j_kurtosis10 += eps
        j_kurtosis100 += eps

        features_td['val_z10'][t_idx, :, :] = (j_slice - j_mean10) / j_std10
        features_td['val_z100'][t_idx, :, :] = \
            (j_slice - j_mean100) / j_std100
        features_td['val_med10'][t_idx, :, :] = j_slice / j_med10
        features_td['val_med100'][t_idx, :, :] = j_slice / j_med100
        features_td['val_mean10'][t_idx, :, :] = j_slice / j_mean10
        features_td['val_mean100'][t_idx, :, :] = j_slice / j_mean100

        features_td['rat_mean_10_100'][t_idx, :, :] = j_mean10 / j_mean100
        features_td['rat_std_10_100'][t_idx, :, :] = j_std10 / j_std100
        features_td['rat_med_10_100'][t_idx, :, :] = j_med10 / j_med100
        features_td['rat_skew_10_100'][t_idx, :, :] = j_skew10 / j_skew100
        features_td['rat_kurtosis_10_100'][t_idx, :, :] = \
            j_kurtosis10 / j_kurtosis100

        features_td['rat_med_mean_10'][t_idx, :, :] = j_med10 / j_mean10
        features_td['rat_med_mean_100'][t_idx, :, :] = j_med100 / j_mean100
        features_td['rat_skew_kurtosis_10'][t_idx, :, :] = \
            j_skew10 / j_kurtosis10
        features_td['rat_skew_kurtosis_100'][t_idx, :, :] = \
            j_skew100 / j_kurtosis100

        features_td['smooth_uni_5'][t_idx, :, :] = \
            np.mean(mat[t_idx-5:t_idx, :, :], axis=0)
        features_td['smooth_wt_5'][t_idx, :, :] = \
            0.4*mat[t_idx, :, :] + 0.25*mat[t_idx-1, :, :] + \
            0.15*mat[t_idx-2, :, :] + 0.1*mat[t_idx-3, :, :] + \
            + 0.1*mat[t_idx-4, :, :]
        features_td['smooth_uni_25'][t_idx, :, :] = \
            np.mean(mat[t_idx-25:t_idx, :, :], axis=0)

    return features_td


def compute_features_sd(mat, t_stamps):
    '''\nCalculates spatial features at every location and time stamp'''

    # all these features are calculated in the spatial domain at every point

    # initialize an empty dictionary to hold and return all features
    features_sd = {}

    # z val calculates the distance of obs from the mean of other obs
    # present at same radius
    features_sd['val_z_r'] = np.zeros(mat.shape)

    # z val calculates the distance of obs from the mean of other obs
    # present at same theta
    features_sd['val_z_o'] = np.zeros(mat.shape)

    # median of obs from in a surround 3x3 grid
    features_sd['med_3x3'] = np.zeros(mat.shape)

    # median of obs from in a surround 5x5 grid
    features_sd['med_5x5'] = np.zeros(mat.shape)

    # iterate through different time stamps
    for t_idx in t_stamps:
        j_mean_r = np.mean(mat[t_idx, :, :], axis=1)
        j_mean_o = np.mean(mat[t_idx, :, :], axis=0)
        j_std_r = np.std(mat[t_idx, :, :], axis=1)
        j_std_o = np.std(mat[t_idx, :, :], axis=0)

        # Add an extra dimension in the last axis
        j_mean_r = np.expand_dims(j_mean_r, axis=1)
        j_mean_o = np.expand_dims(j_mean_o, axis=0)
        j_std_r = np.expand_dims(j_std_r, axis=1)
        j_std_o = np.expand_dims(j_std_o, axis=0)

        features_sd['val_z_r'][t_idx, :, :] = \
            (mat[t_idx, :, :] - j_mean_r) / j_std_r

        features_sd['val_z_o'][t_idx, :, :] = \
            (mat[t_idx, :, :] - j_mean_o) / j_std_o

        for r_idx in range(1, mat.shape[1]-1):
            for o_idx in range(1, mat.shape[2]-1):
                features_sd['med_3x3'][t_idx, r_idx, o_idx] =\
                    np.median(mat[t_idx, r_idx-1:r_idx+2, o_idx-1:o_idx+2])

        for r_idx in range(2, mat.shape[1]-2):
            for o_idx in range(2, mat.shape[2]-2):
                features_sd['med_5x5'][t_idx, r_idx, o_idx] =\
                    np.median(mat[t_idx, r_idx-2:r_idx+3, o_idx-2:o_idx+5])

    return features_sd


def compute_features_grad(mat):
    '''\nCalculates spatial and temporal gradients'''

    # initialize an empty dictionary to hold and return all features
    features_grad = {}

    # first derivative
    features_grad['t_grad'] = np.roll(mat, -1, axis=0) - mat
    features_grad['s1_grad'] = np.roll(mat, -1, axis=1) - mat
    features_grad['s2_grad'] = np.roll(mat, -1, axis=2) - mat

    # second derivative
    features_grad['t_grad2'] = \
        np.roll(features_grad['t_grad'], -1, axis=0)\
        - features_grad['t_grad']
    features_grad['s1_grad2'] = np.roll(features_grad['s1_grad'], -1,
                                        axis=1)\
        - features_grad['s1_grad']
    features_grad['s2_grad2'] = \
        np.roll(features_grad['s2_grad'], -1, axis=2) - \
        features_grad['s2_grad']

    return features_grad


def define_defects(s1, s2, defs_coord, def_names):
    '''\nDefine coordinates of defects'''

    # initialize empty dictionary to hold defect coordinates
    defs = {}

    # as many defects can be setup within the defects dictionary.
    # each defect key will be an unique identifier

    # defs[n:{...}] contains the parameters of defect 'n'

    # defs[n:{'polygon':}] contains the defect's Point object definition
    # defs[n:{'s1_sel':}] contains the selected s1 coordinates of defect
    # defs[n:{'s2_sel':}] contains the selected s2 coordinates of defect
    # defs[n:{'s1_vertices':}] contains the s1 vertices of defect
    # defs[n:{'s2_vertices':}] contains the s2 vertices of defect
    # defs[n:{'name':}] contains the names of defect

    # iterate through the defects
    for i in range(1, len(defs_coord) + 1):
        defs[i] = {}

        defs[i]['polygon'] = [Point(x, y) for x, y in defs_coord[i-1]]

        defs[i]['s1_sel'] = []
        defs[i]['s2_sel'] = []

        # extract vertices of defects in s1 and s2 coordinates
        defs[i]['s1_vertices'], defs[i]['s2_vertices'] = \
            map(list, zip(*defs_coord[i-1]))

        # add the first element to the tail of dictionary to aid with plots
        defs[i]['s1_vertices'].append(defs[i]['s1_vertices'][0])
        defs[i]['s2_vertices'].append(defs[i]['s2_vertices'][0])

        defs[i]['name'] = def_names[i-1]

    # iterate through the spatial coordinates to identify which pixels
    # belong to defecs
    for s1_idx in range(len(s1)):
        for s2_idx in range(len(s2)):
            p = Point(s1[s1_idx], s2[s2_idx])

            # iterate through the defects
            for i in range(1, len(defs)+1):
                if is_within_polygon(defs[i]['polygon'], p):
                    defs[i]['s1_sel'].append(s1_idx)
                    defs[i]['s2_sel'].append(s2_idx)

    return defs


def visualize_spatial_data(mat, t, s1_2d, s2_2d,
                           t_min_idx, t_max_idx, del_t_idx, units):
    '''\nVisualize spatial slices of data at certain time stamps'''

    # visualize data between t_min_idx - t_max_idx in steps of del_t_idx
    for t_idx in range(t_min_idx, t_max_idx, del_t_idx):
        fig, ax = plt.subplots(nrows=1, ncols=1)

        fig.suptitle('Raw signal')

        # iterating across time stamps
        cs = ax.contourf(s1_2d, s2_2d, mat[t_idx, :, :])
        ax.plot(0, 0)
        ax.set(title='Time stamp: %0.1f%s' % (t[t_idx], units['t_units']))
        ax.set(xlabel='%s' % (units['s1_units']))
        ax.set(ylabel='%s' % (units['s2_units']))
        fig.colorbar(cs, ax=ax, shrink=0.5)

        plt.tight_layout()
        plt.show()

    return


def visualize_time_series(data, t, s1, s2, units):
    '''\nPick 4 random spatial coordinates and chart the time-series'''

    # pick 4 random spatial coordinates along s1 and s2 axes
    s1_coord = np.random.randint(low=0, high=len(s1), size=4)
    s2_coord = np.random.randint(low=0, high=len(s2), size=4)

    # extract time series signals at the four random coordinates
    sig1 = data[:, s1_coord[0], s2_coord[0]]
    sig2 = data[:, s1_coord[1], s2_coord[1]]
    sig3 = data[:, s1_coord[2], s2_coord[2]]
    sig4 = data[:, s1_coord[3], s2_coord[3]]

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Time series')

    ax[0, 0].plot(t, sig1)
    ax[0, 0].set(xlabel='Time[%s]' % (units['t_units']), ylabel='Signal',
                 title='s1=%0.1f%s, s2=%0.1f%s' %
                 (s1[s1_coord[0]], units['s1_units'],
                  s2[s2_coord[0]], units['s2_units']))
    ax[0, 0].grid()

    ax[1, 0].plot(t, sig2)
    ax[1, 0].set(xlabel='Time[%s]' % (units['t_units']), ylabel='Signal',
                 title='s1=%0.1f%s, s2=%0.1f%s' %
                 (s1[s1_coord[1]], units['s1_units'],
                  s2[s2_coord[1]], units['s2_units']))
    ax[1, 0].grid()

    ax[0, 1].plot(t, sig3)
    ax[0, 1].set(xlabel='Time[%s]' % (units['t_units']), ylabel='Signal',
                 title='s1=%0.1f%s, s2=%0.1f%s' %
                 (s1[s1_coord[2]], units['s1_units'],
                  s2[s2_coord[2]], units['s2_units']))
    ax[0, 1].grid()

    ax[1, 1].plot(t, sig4)
    ax[1, 1].set(xlabel='Time[%s]' % (units['t_units']), ylabel='Signal',
                 title='s1=%0.1f%s, s2=%0.1f%s' %
                 (s1[s1_coord[3]], units['s1_units'],
                  s2[s2_coord[3]], units['s2_units']))
    ax[1, 1].grid()

    plt.tight_layout()
    plt.show()

    return


def read_matlab_data(dataset, table):
    '''\nReads in raw matlab data using scipy IO modules'''

    # use scipy IO modules to read in matlab data
    mat = scipy.io.loadmat(dataset)[table]
    return mat


def main():
    '''\nAll the subroutines will be called from here'''

    # load data
    print(read_matlab_data.__doc__)
    # Example - this assumes a matlab dataset named defect.mat and the
    # table named rawData inside the dataset
    dataset = 'sample.mat'
    tablename = 'rawData1'
    mat = read_matlab_data(dataset=dataset, table=tablename)

    # describe data
    # the input dataset is assumed to contain a time axis and two spatial
    # coordinates - s1 and s2. Rearrange axes as necessary.
    [t_max, s1_max, s2_max] = mat.shape
    print('Shape of the data matrix')
    print('t_max: %d  s1_max: %d s2_max: %d' % (t_max, s1_max, s2_max))

    # scanning parameters
    # in this sample code, time axes ranges from t_lb to t_ub over t_max
    t_lb = 0*1e-1
    t_ub = t_max*1e-1
    t = np.linspace(t_lb, t_ub, t_max)

    # s1 axis range from s1_lb to s1_ub divided over s1_max steps
    s1_lb = 0
    s1_ub = 200
    s1 = np.linspace(s1_lb, s1_ub, s1_max)

    # s2 axis range from s2_lb to s2_ub divided over s2_max steps
    s2_lb = 0
    s2_ub = 360
    s2 = np.linspace(s2_lb, s2_ub, s2_max)

    # dictionary object to hold the units along the different axis
    units = {'t_units': '$\\mu$S', 's1_units': 'mm', 's2_units': 'mm'}

    # meshgrid conversion in 2D
    s1_2d, s2_2d = np.meshgrid(s1, s2, indexing='ij')

    # raw data visualization
    print(visualize_time_series.__doc__)
    visualize_time_series(mat, t, s1, s2, units)

    print(visualize_spatial_data.__doc__)
    t_min_idx = 450
    t_max_idx = 500
    del_t_idx = 25
    visualize_spatial_data(mat, t, s1_2d, s2_2d,
                           t_min_idx, t_max_idx, del_t_idx, units)

    # define defects
    print(define_defects.__doc__)
    # define as many defects as needed
    # each defect should contain the coordiantes of the vertices
    # the structure is list of tuples
    def1 = [(20, 20), (50, 10), (30, 40), (20, 30)]
    def2 = [(120, 120), (180, 120), (150, 180)]
    def3 = [(60, 60), (80, 60), (80, 80), (60, 80)]

    # list contains all the defects
    defs_coord = [def1, def2, def3]
    def_names = ['D1', 'D2', 'D3']  # names of defects
    defs = define_defects(s1, s2, defs_coord, def_names)

    # sample time indices where computationally intentionally features
    # will be calculated.
    t_stamps = [500, 550, 600]

    # identity features
    features_id = {}
    features_id['id'] = mat

    # compute gradient features
    print(compute_features_grad.__doc__)
    features_grad = {}
    features_grad = compute_features_grad(mat)

    # compute spatial domain features
    print(compute_features_sd.__doc__)
    features_sd = {}
    features_sd = compute_features_sd(mat, t_stamps)

    # compute time domain features
    print(compute_features_td.__doc__)
    features_td = {}
    features_td = compute_features_td(mat, t_stamps)

    # compute wavelet decomposition features
    print(compute_features_wav.__doc__)
    features_wav = {}
    features_wav = compute_features_wav(mat, t_stamps)

    # visualize feature
    print(visualize_features.__doc__)
    t_idx = 650
    visualize_features(mat, features_grad, s1_2d, s2_2d, 's1_grad',
                       t_idx, t, units)
    visualize_features(mat, features_grad, s1_2d, s2_2d, 's2_grad',
                       t_idx, t, units)

    # combine features
    # print(combine_features.__doc__)
    feature_list = [features_id, features_grad, features_sd,
                    features_td, features_wav]
    features = {}
    features = combine_features(feature_list)
    print('Total number of features is %d' % (len(features)))

    # normalize features
    print(normalize_features.__doc__)
    features = normalize_features(features, t_stamps)

    # Outlier analysis using Mahalanobis distance
    # if PCA is required to trim features, set pca_var to the desired
    # explained varaince level - in this example, 90% variance is desired
    print(outlier_mah.__doc__)
    mah = {}
    mah = outlier_mah(features, t_stamps, pca_var=0.9)

    # fit Isolation Forest model
    # if PCA is required to trim features, set pca_var to the desired
    # explained varaince level - in this example, 90% variance is desired
    print(fit_isolationforest_model.__doc__)
    iso = {}
    iso = fit_isolationforest_model(features, t_stamps, pca_var=0.9)

    # scale frames between 0-1
    print(scale_frames.__doc__)
    mat = scale_frames(mat, t_stamps)
    mah = scale_frames(mah, t_stamps)
    iso = scale_frames(iso, t_stamps)

    # Defect detection metrics
    print(defect_detection_metrics.__doc__)
    defect_detection_metrics(mat, mah, iso, s1_2d, s2_2d,
                             defs, t_stamps, t, units, plot=True)


if __name__ == '__main__':
    '''\n
    Defect Detection and Quantification Toolbox (DDQT)

    A Python toolbox for -
      Reading in Matlab data
      Visualizing data
      Creating features in the time and spatial domain
      Feature reduction using PCA
      Identifying defects using Mahalanobis distance and Outlier Forest
      Quantifying results using ROC curves
      Visualizing outcomes

    todo:
    '''

    main()
