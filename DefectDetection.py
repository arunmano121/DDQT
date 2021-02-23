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


def feature_perf(iso, t_stamps, d1, d2, d3):
    '''\nQuantification of defect detection of features'''

    # mult_arr holds the multiplier factor to add computations only in the
    # quadrant containing the defect
    mult_arr = np.zeros((len(t_stamps), iso.shape[1], iso.shape[2]))

    # boundaries of different quadrants
    o_0 = int(394/360*0)
    o_90 = int(394/360*90)
    o_180 = int(394/360*180)
    o_270 = int(394/360*270)
    o_360 = int(394/360*360)

    mult_arr[0, :, o_90:o_180] = 1  # D1 is in quad 2 between 90 and 180
    mult_arr[1, :, o_270:o_360] = 1  # D2 is in quad 4 between 270 and 360
    mult_arr[2, :, o_0:o_90] = 1  # D3 is in quad 1 between 0 and 90

    # y_truth holds the true labels of the defective areas
    # location of defects are labeled with 1
    y_truth = np.zeros(iso.shape)
    # location of D1
    y_truth[:, d1['r_sel'], d1['o_sel']] = 1
    # location of D2
    y_truth[:, d2['r_sel'], d2['o_sel']] = 1
    # location of D3
    y_truth[:, d3['r_sel'], d3['o_sel']] = 1

    # cumulative sum of performance across the three defects
    j_perf = 0

    # counter holds the defect number and t_idx contains the time index
    # of the defect
    # selecting features based on performance over the 3 defects in total
    for counter, t_idx in enumerate(t_stamps):
        # Compute ROC curve and ROC area for outliers computed using
        # Isolation Forest
        fpr_iso, tpr_iso, _ = \
            roc_curve(
                (y_truth[t_idx, :, :] * mult_arr[counter, :, :]).ravel(),
                (iso[t_idx, :, :] * mult_arr[counter, :, :]).ravel())
        auc_iso = auc(fpr_iso, tpr_iso)

        j_perf += auc_iso

    # selecting features based on performance over individual defect
    # counter ranges from 0-2 corresponding to D1-D3
    # counter = 2
    # t_idx = t_stamps[counter]
    # # Compute ROC curve and ROC area for outliers computed using
    # # Isolation Forest
    # fpr_iso, tpr_iso, _ = \
    #     roc_curve(
    #         (y_truth[t_idx, :, :] * mult_arr[counter, :, :]).ravel(),
    #         (iso[t_idx, :, :] * mult_arr[counter, :, :]).ravel())
    # auc_iso = auc(fpr_iso, tpr_iso)

    # j_perf += auc_iso

    return j_perf


def feature_selection(features, t_stamps, mat, d1, d2, d3):
    '''\nSelect features based on greedy forward selection'''

    # feature performance measurement is performed by measuring the ROC of
    # feature across the three defects d1, d2 and d3. At each stage, the
    # best feature is added to the selection pool and the incremental
    # benefit of adding additional features is verified

    # a temporary place-holder dictionary to hold the current feature along
    # with the selected features
    temp = {}

    # selected features
    sel_features = {}

    # maximum performance of feature within one round
    max_perf_round = 0.0
    max_prev_perf_round = 0.0

    for i in range(len(features)):
        # iterating through the different features
        for feature in features.keys():
            if feature not in sel_features.keys():
                j_feature = {}
                j_feature[feature] = features[feature]
                temp = combine_features([sel_features, j_feature])

                iso = fit_isolationforest_model(temp, t_stamps, pca=False,
                                                pca_var=0.95)
                iso = scale_frames(iso, t_stamps)
                j_perf = feature_perf(iso, t_stamps, d1, d2, d3)

                # if current feature exceeds performance of current maximum
                if j_perf > max_perf_round:
                    # set max_perf to j_perf
                    max_perf_round = j_perf
                    j_best_feature = feature

        # only if the performance of the current round exceeds the previous
        # round, update the features else break out of feature selection
        if max_perf_round > max_prev_perf_round:
            sel_features[j_best_feature] = features[j_best_feature]

            print("Round %d: Selected Features: %s"
                  % (i, sel_features.keys()))

            max_prev_perf_round = max_perf_round
        else:
            break

    return sel_features


def annotate_plots(ax, d, title):
    '''\nAnnotate charts with locations of defects'''

    ax.plot([d['o1'], d['o1'], d['o2'], d['o2'], d['o1']],
            [d['r1'], d['r2'], d['r2'], d['r1'], d['r1']], color='red')
    ax.annotate(title, xy=(d['o1'], int(0.5*d['r1'] + 0.5*d['r2'])))

    return


def defect_detection_metrics(mat, mah, iso, r2, o2, t_stamps, d1, d2, d3,
                             plot=True):
    '''\nQuantification of defect detection'''

    # mult_arr holds the multiplier factor to add computations only in the
    # quadrant containing the defect
    mult_arr = np.zeros((len(t_stamps), r2.shape[0], r2.shape[1]))

    # boundaries of different quadrants
    o_0 = int(394/360*0)
    o_90 = int(394/360*90)
    o_180 = int(394/360*180)
    o_270 = int(394/360*270)
    o_360 = int(394/360*360)

    mult_arr[0, :, o_90:o_180] = 1  # D1 is in quad 2 between 90 and 180
    mult_arr[1, :, o_270:o_360] = 1  # D2 is in quad 4 between 270 and 360
    mult_arr[2, :, o_0:o_90] = 1  # D3 is in quad 1 between 0 and 90

    # y_truth holds the true labels of the defective areas
    # location of defects are labeled with 1
    y_truth = np.zeros(mat.shape)
    # location of D1
    y_truth[:, d1['r_sel'], d1['o_sel']] = 1
    # location of D2
    y_truth[:, d2['r_sel'], d2['o_sel']] = 1
    # location of D3
    y_truth[:, d3['r_sel'], d3['o_sel']] = 1

    # counter holds the defect number and t_idx contains the time index
    # of the defect
    for counter, t_idx in enumerate(t_stamps):

        # Compute ROC curve and ROC area for raw data
        fpr_mat, tpr_mat, _ = \
            roc_curve(
                (y_truth[t_idx, :, :] * mult_arr[counter, :, :]).ravel(),
                (mat[t_idx, :, :] * mult_arr[counter, :, :]).ravel())
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
                (y_truth[t_idx, :, :] * mult_arr[counter, :, :]).ravel(),
                (mah[t_idx, :, :] * mult_arr[counter, :, :]).ravel())
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
                (y_truth[t_idx, :, :] * mult_arr[counter, :, :]).ravel(),
                (iso[t_idx, :, :] * mult_arr[counter, :, :]).ravel())
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
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),
                                   nrows=1, ncols=3)

            fig.suptitle('Result of Outlier analysis at %d$\\mu$S'
                         % (t_idx*0.1))

            # raw data
            cs = ax[0].contourf(o2, r2, mat[t_idx, :, :])
            ax[0].plot(0, 0)
            ax[0].set(title='Raw data')
            annotate_plots(ax[0], d1, 'D1')
            annotate_plots(ax[0], d2, 'D2')
            annotate_plots(ax[0], d3, 'D3')
            fig.colorbar(cs, ax=ax[0], shrink=0.33)

            # outliers computed using Mahalanobis distance
            cs = ax[1].contourf(o2, r2, mah[t_idx, :, :])
            ax[1].plot(0, 0)
            ax[1].set(title='Outliers using Mahalanobis distance')
            annotate_plots(ax[1], d1, 'D1')
            annotate_plots(ax[1], d2, 'D2')
            annotate_plots(ax[1], d3, 'D3')
            fig.colorbar(cs, ax=ax[1], shrink=0.33)

            # outliers computed using Isolation Forest
            cs = ax[2].contourf(o2, r2, iso[t_idx, :, :])
            ax[2].plot(0, 0)
            ax[2].set(title='Isolation Forest')
            annotate_plots(ax[2], d1, 'D1')
            annotate_plots(ax[2], d2, 'D2')
            annotate_plots(ax[2], d3, 'D3')
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

            plt.title('ROC curve - Time: %d$\\mu$S' % (t_idx*0.1))
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


def fit_isolationforest_model(features, t_stamps, pca, pca_var):
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

        # if PCA is required the flag should be set to True
        if pca:
            # Standardizing the features before performing PCA
            X = StandardScaler().fit_transform(X)
            # perform PCA to reduce dimensionality
            pca = PCA(n_components=pca_var)
            X = pca.fit_transform(X)

        # fit the Isolation Forest model
        clf.fit(X)

        # predict outliers
        # multiply by -1 to flip labels and be consistent with mah and mat
        # does not affect any other quantitative results
        val = -1 * clf.decision_function(X)

        iso[t_idx, :, :] = val.reshape((shape[1], shape[2]))

    return iso


def outlier_mah(features, t_stamps, pca, pca_var):
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

        # if PCA is required the flag should be set to True
        if pca:
            # Standardizing the features before performing PCA
            X = StandardScaler().fit_transform(X)
            # perform PCA to reduce dimensionality
            pca = PCA(n_components=pca_var)
            X = pca.fit_transform(X)

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


def visualize_features(mat, features, r2, o2, feature, t_idx):
    '''\nVisualize computed features'''

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),
                           nrows=1, ncols=2)

    fig.suptitle('Comparison of raw signal and feature at %d$\\mu$S'
                 % (t_idx*0.1))

    # contour plot of raw signal
    ax[0].contourf(o2, r2, mat[t_idx, :, :])
    ax[0].plot(0, 0)
    ax[0].set(title='Raw signal')

    # contour plot of feature
    ax[1].contourf(o2, r2, features[feature][t_idx, :, :])
    ax[1].plot(0, 0)
    ax[1].set(title='Feature - ' + feature)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Radial scan')

    o = 300
    o_idx = int(394/360*o)

    # radial scan of raw signal
    ax[0].plot(r2[:, o_idx], mat[t_idx, :, o_idx])
    ax[0].set(xlabel='Radius [mm]', ylabel='Signal',
              title='Theta=%0.1f$^\\circ$' % (o))
    ax[0].grid()

    # radial scan of feature
    ax[1].plot(r2[:, o_idx], features[feature]
               [t_idx, :, o_idx])
    ax[1].set(xlabel='Radius [mm]', ylabel='Feature',
              title='Theta=%0.1f$^\\circ$' % (o))
    ax[1].grid()

    plt.tight_layout()
    plt.show()

    return


def compute_features_wav(mat, t_stamps):
    '''\nCalculates wavelet transformed and reconstructed features
    at every location'''

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


def id_timestamps(features, d1, d2, d3):
    '''Calculates time stamps of defects using correlation among features'''

    f1_d1 = np.mean(features['t_grad'][:, d1['r_sel'], d1['o_sel']],
                    axis=(1))
    f2_d1 = np.mean(features['r_grad'][:, d1['r_sel'], d1['o_sel']],
                    axis=(1))
    cor_d1 = np.convolve(f1_d1, f2_d1, 'same')

    f1_d2 = np.mean(features['t_grad'][:, d2['r_sel'], d2['o_sel']],
                    axis=(1))
    f2_d2 = np.mean(features['r_grad'][:, d2['r_sel'], d2['o_sel']],
                    axis=(1))
    cor_d2 = np.convolve(f1_d2, f2_d2, 'same')

    f1_d3 = np.mean(features['t_grad'][:, d3['r_sel'], d3['o_sel']],
                    axis=(1))
    f2_d3 = np.mean(features['r_grad'][:, d3['r_sel'], d3['o_sel']],
                    axis=(1))
    cor_d3 = np.convolve(f1_d3, f2_d3, 'same')

    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # fig.suptitle('Correlation for features at defects')

    # ax[0].plot(cor_d1)
    # ax[0].set(xlabel='Coordinate', ylabel='Correlation', title='D1')
    # ax[0].grid()

    # ax[1].plot(cor_d2)
    # ax[1].set(xlabel='Coordinate', ylabel='Correlation', title='D2')
    # ax[1].grid()

    # ax[2].plot(cor_d3)
    # ax[2].set(xlabel='Coordinate', ylabel='Correlation', title='D3')
    # ax[2].grid()

    # plt.tight_layout()
    # plt.show()

    t_stamps = [np.argmax(cor_d1), np.argmax(cor_d2), np.argmax(cor_d3)]

    print('Coordinates of max correlation at defect D1: %d, D2: %d, D3: %d'
          % (t_stamps[0], t_stamps[1], t_stamps[2]))

    return t_stamps


def compute_features_grad(mat):
    '''\nCalculates spatial and temporal gradients'''

    # initialize an empty dictionary to hold and return all features
    features_grad = {}

    # first derivative
    features_grad['t_grad'] = np.roll(mat, -1, axis=0) - mat
    features_grad['r_grad'] = np.roll(mat, -1, axis=1) - mat
    features_grad['o_grad'] = np.roll(mat, -1, axis=2) - mat

    # second derivative
    features_grad['t2_grad'] = \
        np.roll(features_grad['t_grad'], -1, axis=0)\
        - features_grad['t_grad']
    features_grad['r2_grad'] = np.roll(features_grad['r_grad'], -1, axis=1)\
        - features_grad['r_grad']
    features_grad['o2_grad'] = \
        np.roll(features_grad['o_grad'], -1, axis=2) - \
        features_grad['o_grad']

    return features_grad


def define_defects(r, o):
    '''\nDefine coordinates of defects'''

    # initialize empty dictionary to hold the theta and radial coordinates
    d1 = {}
    d2 = {}
    d3 = {}

    # d['o_idx'] contains the angular extent of the defect
    # d['r_idx'] contains the radial extent of the defect
    # d['polygon'] contains the Point object definition of the defect
    # d['r_sel'] contains the selected radius coordinates of the defect
    # d['o_sel'] contains the selected angular coordinates of the defect

    # D1: 186-196mm 125-135deg
    d1['o_idx1'] = int(394/360*125)
    d1['o_idx2'] = int(394/360*135)
    d1['r_idx1'] = 186 - 20
    d1['r_idx2'] = 196 - 20
    d1['o1'] = o[d1['o_idx1']]
    d1['o2'] = o[d1['o_idx2']]
    d1['r1'] = r[d1['r_idx1']]
    d1['r2'] = r[d1['r_idx2']]

    d1['polygon'] = [Point(r*np.cos(theta), r*np.sin(theta))
                     for (r, theta) in
                     [(d1['r1'], d1['o1']), (d1['r1'], d1['o2']),
                      (d1['r2'], d1['o2']), (d1['r2'], d1['o1'])]]

    d1['r_sel'] = []
    d1['o_sel'] = []

    # D2: 150-160mm 295-305deg
    d2['o_idx1'] = int(394/360*295)
    d2['o_idx2'] = int(394/360*305)
    d2['r_idx1'] = 150 - 20
    d2['r_idx2'] = 160 - 20
    d2['o1'] = o[d2['o_idx1']]
    d2['o2'] = o[d2['o_idx2']]
    d2['r1'] = r[d2['r_idx1']]
    d2['r2'] = r[d2['r_idx2']]

    d2['polygon'] = [Point(r*np.cos(theta), r*np.sin(theta))
                     for (r, theta) in
                     [(d2['r1'], d2['o1']), (d2['r1'], d2['o2']),
                      (d2['r2'], d2['o2']), (d2['r2'], d2['o1'])]]

    d2['r_sel'] = []
    d2['o_sel'] = []

    # D3: 120-130mm 40-50deg
    d3['o_idx1'] = int(394/360*40)
    d3['o_idx2'] = int(394/360*50)
    d3['r_idx1'] = 120 - 20
    d3['r_idx2'] = 130 - 20
    d3['o1'] = o[d3['o_idx1']]
    d3['o2'] = o[d3['o_idx2']]
    d3['r1'] = r[d3['r_idx1']]
    d3['r2'] = r[d3['r_idx2']]

    d3['polygon'] = [Point(r*np.cos(theta), r*np.sin(theta))
                     for (r, theta) in
                     [(d3['r1'], d3['o1']), (d3['r1'], d3['o2']),
                      (d3['r2'], d3['o2']), (d3['r2'], d3['o1'])]]

    d3['r_sel'] = []
    d3['o_sel'] = []

    for r_idx in range(len(r)):
        for o_idx in range(len(o)):
            p = Point(r[r_idx]*np.cos(o[o_idx]), r[r_idx]*np.sin(o[o_idx]))

            if is_within_polygon(d1['polygon'], p):
                d1['r_sel'].append(r_idx)
                d1['o_sel'].append(o_idx)
            elif is_within_polygon(d2['polygon'], p):
                d2['r_sel'].append(r_idx)
                d2['o_sel'].append(o_idx)
            elif is_within_polygon(d3['polygon'], p):
                d3['r_sel'].append(r_idx)
                d3['o_sel'].append(o_idx)

    return d1, d2, d3


def visualize_spatial_data(mat, t, r2, o2):
    '''\nVisualize data in polar coordinates'''

    # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),
    #                        nrows=2, ncols=3)

    # fig.suptitle('Raw signal')

    # for counter, t in enumerate(range(400, 1000, 100)):
    #     # iterating across 6 time stamps in steps of 100 micro seconds
    #     r, c = divmod(counter, 3)
    #     cs = ax[r, c].contourf(o2, r2, mat[t, :, :])
    #     ax[r, c].plot(0, 0)
    #     ax[r, c].set(title='Time stamp: %d$\\mu$S' % (t*0.1))
    #     fig.colorbar(cs, ax=ax[r, c], shrink=0.5)

    # plt.tight_layout()
    # plt.show()

    # visualize data between 40 - 90 micro S in steps of 5 micro S
    for counter, t in enumerate(range(400, 900, 25)):
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),
                               nrows=1, ncols=1, figsize=(8, 8))

        fig.suptitle('Raw signal')

        # iterating across 16 time stamps
        cs = ax.contourf(o2, r2, mat[t, :, :])
        ax.plot(0, 0)
        ax.set(title='Time stamp: %d$\\mu$S' % (t*0.1))
        fig.colorbar(cs, ax=ax, shrink=0.5)

        plt.tight_layout()
        plt.show()

    return


def visualize_time_series(data, t, r, o):
    '''\nPick 4 random spatial coordinates and chart the time-series'''

    # pick 4 random spatial and theta coordinates
    o_coord = np.random.randint(low=0, high=len(o), size=4)
    r_coord = np.random.randint(low=0, high=len(r), size=4)

    # extract time series signals at the four random coordinates
    s1 = data[:, r_coord[0], o_coord[0]]
    s2 = data[:, r_coord[1], o_coord[1]]
    s3 = data[:, r_coord[2], o_coord[2]]
    s4 = data[:, r_coord[3], o_coord[3]]

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Time series')

    ax[0, 0].plot(t*(1e6), s1)
    ax[0, 0].set(xlabel='Time [$\\mu$S]', ylabel='Signal',
                 title='Coordinates: r=%0.1f mm, theta=%0.1f$^\\circ$' %
                 (r[r_coord[0]], np.degrees(o[o_coord[0]])))
    ax[0, 0].grid()

    ax[1, 0].plot(t*(1e6), s2)
    ax[1, 0].set(xlabel='Time [$\\mu$S]', ylabel='Signal',
                 title='Coordinates: r=%0.1f mm, theta=%0.1f$^\\circ$' %
                 (r[r_coord[1]], np.degrees(o[o_coord[1]])))
    ax[1, 0].grid()

    ax[0, 1].plot(t*(1e6), s3)
    ax[0, 1].set(xlabel='Time [$\\mu$S]', ylabel='Signal',
                 title='Coordinates: r=%0.1f mm, theta=%0.1f$^\\circ$' %
                 (r[r_coord[2]], np.degrees(o[o_coord[2]])))
    ax[0, 1].grid()

    ax[1, 1].plot(t*(1e6), s4)
    ax[1, 1].set(xlabel='Time [$\\mu$S]', ylabel='Signal',
                 title='Coordinates: r=%0.1f mm, theta=%0.1f$^\\circ$' %
                 (r[r_coord[3]], np.degrees(o[o_coord[3]])))
    ax[1, 1].grid()

    plt.tight_layout()
    plt.show()


def read_matlab_data(dataset, table):
    '''\nReads in raw matlab data using scipy IO modules'''

    # use scipy IO modules to read in matlab data
    mat = scipy.io.loadmat(dataset)[table]
    return mat


def main():
    '''\nAll the subroutines will be called from here'''

    # load data
    # print(read_matlab_data.__doc__)
    mat = read_matlab_data(dataset='rawData1.mat', table='rawData1')

    # trimming data
    mat = mat[0:1000, :, :]

    # describe data
    [t_max, r_max, o_max] = mat.shape
    print('Shape of the data matrix -  t_max: %d  r_max: %d  theta_max: %d'
          % (t_max, r_max, o_max))

    # scanning parameters
    t = np.linspace(1, t_max, t_max)*(1e-7)  # in micro seconds
    r = np.linspace(20, 220, r_max)  # in mm
    o = np.radians(np.linspace(0, 360, o_max))  # in radians

    # polar meshgrid conversion in 2D
    r2, o2 = np.meshgrid(r, o, indexing='ij')

    # raw data visualization
    # print(visualize_time_series.__doc__)
    visualize_time_series(mat, t, r, o)
    # print(visualize_spatial_data.__doc__)
    visualize_spatial_data(mat, t, r2, o2)

    # define defects
    # print(define_defects.__doc__)
    d1, d2, d3 = define_defects(r, o)

    # identity features
    features_id = {}
    features_id['id'] = mat

    # compute gradient features
    # print(compute_features_grad.__doc__)
    features_grad = {}
    features_grad = compute_features_grad(mat)

    # identify time stamps of defects using feature correlation analysis
    # print(id_timestamps.__doc__)
    # time stamps where computationally intensive features will
    # be calculated
    # the following time stamps are the instants when A0 wave
    # encounters the defects - this was determined by performing the
    # step above
    t_stamps = id_timestamps(features_grad, d1, d2, d3)

    # delta time before or after the A0 wave encounters defect
    del_t = 0
    t_stamps = list(np.array(t_stamps) + del_t)

    # setting based on email
    t_stamps = [690, 550, 500]

    # compute spatial domain features
    # print(compute_features_sd.__doc__)
    features_sd = {}
    features_sd = compute_features_sd(mat, t_stamps)

    # compute time domain features
    # print(compute_features_td.__doc__)
    features_td = {}
    features_td = compute_features_td(mat, t_stamps)

    # compute wavelet decomposition features
    # print(compute_features_wav.__doc__)
    features_wav = {}
    features_wav = compute_features_wav(mat, t_stamps)

    # visualize feature
    # print(visualize_features.__doc__)
    visualize_features(mat, features_grad, r2, o2, 'r_grad', 650)
    visualize_features(mat, features_grad, r2, o2, 'o_grad', 650)

    # combine features
    # print(combine_features.__doc__)
    feature_list = [features_id, features_grad, features_sd,
                    features_td, features_wav]
    features = {}
    features = combine_features(feature_list)
    print('Total number of features is %d' % (len(features)))

    # normalize features
    # print(normalize_features.__doc__)
    features = normalize_features(features, t_stamps)

    # selecting features based on greedy forward selection
    # print(feature_selection.__doc__)
    # features = feature_selection(features, t_stamps, mat, d1, d2, d3)
    selected_features = {}
    for feature in ['rat_skew_10_100', 'o_grad', 'rat_med_mean_10']:
        selected_features[feature] = features[feature]

    features = selected_features
    print('Total number of features after greedy feature '
          'selection is %d' % (len(features)))

    # Outlier analysis using Mahalanobis distance
    # print(outlier_mah.__doc__)
    mah = {}
    mah = outlier_mah(features, t_stamps, pca=False, pca_var=0.95)

    # fit Isolation Forest model
    # print(fit_isolationforest_model.__doc__)
    iso = {}
    iso = fit_isolationforest_model(features, t_stamps,
                                    pca=False, pca_var=0.95)

    # scale frames between 0-1
    # print(scale_frames.__doc__)
    mat = scale_frames(mat, t_stamps)
    mah = scale_frames(mah, t_stamps)
    iso = scale_frames(iso, t_stamps)

    # Defect detection metrics
    # print(defect_detection_metrics.__doc__)
    defect_detection_metrics(mat, mah, iso, r2, o2, t_stamps,
                             d1, d2, d3, plot=True)


if __name__ == '__main__':
    '''\n
    Defect Detection and Quantification Toolbox using Python

    A Python toolbox for -
      Reading in Matlab data
      Visualizing data
      Creating features in the time and spatial domain
      Greedy feature selection
      Identifying defects using Mahalanobis distance and Outlier Forest
      Quantifying results using ROC curves
      Visualizing outcomes

    TODO:
    -
    '''

    main()
