import os
import cv2
import json
import numpy as np
from collections import Counter
import config_lgrcs as config


def get_distance(vec_a, vec_b, axis):
    return np.linalg.norm(vec_a - vec_b, axis=axis)


def get_ratio(a, b, epsilon=1e-5):
    b = np.add(b, epsilon)
    return np.divide(a, b)


def get_angle(u, v):
    # alpha = arcos( u @ v / (|u| * |v|) )
    if (u == v).all():
        return 0.0

    u_ = u / np.linalg.norm(u)
    v_ = v / np.linalg.norm(v)
    cos_a = u_ @ v_

    alpha = np.arccos(cos_a)
    alpha = np.degrees(alpha)
    return alpha


def calculate_area(points):
    mask = np.zeros((config.PIXELS_SIZE, config.PIXELS_SIZE))
    hull = cv2.convexHull(points)
    draw = cv2.drawContours(mask, [hull], -1, 1)
    area = cv2.countNonZero(draw)

    return float(area)


def image_enhancements(gray_img):
    """
    This tool developed in order to use a NoIR camera(a cheap one),
    for keeping the 'face predictor' stable when the user
    is in changing his lighting environment.
    """
    hist, bins = np.histogram(gray_img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255. / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    gray_img = cdf[gray_img]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    gray_img = clahe.apply(gray_img)

    return gray_img


def feature_extraction(point):

    inner_idx = config.LANDMARKS_INDEXES['Inner_lips']
    outer_idx = config.LANDMARKS_INDEXES['Outer_lips']

    inner = point[inner_idx[0]: inner_idx[1]]
    outer = point[outer_idx[0]: outer_idx[1]]

    inner_area = calculate_area(inner) # contains 8 landmarks
    outer_area = calculate_area(outer) # contains 12 landmarks
    in2out_ratio = get_ratio(inner_area, outer_area)

    width = get_distance(outer[0], outer[6], axis=0)  # landmarks: #48 & #54
    height = get_distance(outer[3], outer[9], axis=0)  # landmarks: #51 & #57
    h2w_ratio = get_ratio(height, width)

    vector_u = inner[4] - inner[3]
    vector_v = inner[4] - inner[5]
    angle = get_angle(vector_u, vector_v)

    row = np.array([height, width, h2w_ratio, inner_area, outer_area, in2out_ratio, angle])

    return row


def detect_outliers(df, n_appears, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    for col in features:
        q1 = np.percentile(df[col], 25)
        q3 = np.percentile(df[col], 75)

        # Inter-quartile range (IQR)
        iqr = q3 - q1
        outlier_step = 1.2 * iqr
        outlier_list_col = df[(df[col] < q1 - outlier_step) | (df[col] > q3 + outlier_step )].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n_appears)

    return multiple_outliers


def export_model_details(model, path, name=None, cr=None):
    """
    export the final summary report
    """
    if name is not None:
        path = os.path.join(path, name)

    hash = '#'*30
    model_conf = model.get_config()
    opt_conf = model.optimizer.get_config()

    with open(path, 'w') as f:
        if cr is not None:
            f.write('\n  {}\t  CLASSIFICATION REPORT \t {}\n\n\n'.format(hash, hash))
            f.write('\n' + cr + '\n')

        f.write('\n  {}\t  SUMMARY \t {}\n\n\n'.format(hash, hash))
        model.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write('\n  {}\t  OPTIMIZER CONFIGURATION \t {}\n\n\n'.format(hash, hash))
        f.write(json.dumps(str(opt_conf), indent=4, sort_keys=True))

        f.write('\n  {}\t  LAYERS CONFIGURATION \t {}\n\n\n'.format(hash, hash))
        f.write(json.dumps(model_conf, indent=4, sort_keys=True))

    print('[INFO] {} saved.'.format(path))