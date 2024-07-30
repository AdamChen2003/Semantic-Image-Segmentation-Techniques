import cv2
import numpy as np

import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern

def extract_combined_features(image):
    # Extract all features
    color_features = extract_color_features(image)
    lbp_features = extract_lbp_features(image)
    hog_features = extract_hog_features(image)
    
    # Combine all features
    combined_features = np.concatenate([color_features, lbp_features, hog_features])
    
    return combined_features

def extract_color_features(img):
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Compute color histograms
    hist_rgb = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist_lab = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Flatten and normalize histograms
    hist_rgb = hist_rgb.flatten() / hist_rgb.sum()
    hist_hsv = hist_hsv.flatten() / hist_hsv.sum()
    hist_lab = hist_lab.flatten() / hist_lab.sum()
    
    # Compute mean and std for each channel
    means_rgb = img.mean(axis=(0, 1))
    stds_rgb = img.std(axis=(0, 1))
    
    return np.concatenate([hist_rgb, hist_hsv, hist_lab, means_rgb, stds_rgb])

def extract_lbp_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
    return features

def extract_sift_features(image, num_features=100):
    sift = cv2.SIFT_create(nfeatures=num_features)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return np.zeros((num_features, 128))  # SIFT descriptor is 128-dimensional
    if descriptors.shape[0] < num_features:
        padding = np.zeros((num_features - descriptors.shape[0], 128))
        descriptors = np.vstack((descriptors, padding))
    return descriptors[:num_features].flatten()

def extract_surf_features(image, num_features=100, hessian_threshold=400):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold, nOctaves=4, nOctaveLayers=3, extended=False, upright=True)
    keypoints, descriptors = surf.detectAndCompute(image, None)
    if descriptors is None:
        return np.zeros((num_features, 64))  # SURF descriptor is 64-dimensional
    if descriptors.shape[0] < num_features:
        padding = np.zeros((num_features - descriptors.shape[0], 64))
        descriptors = np.vstack((descriptors, padding))
    return descriptors[:num_features].flatten()

def extract_features(image, method):
    # Using the method prescribed in the paper: https://arxiv.org/pdf/1901.07828

    # HOG
    if method == 'HOG':
        return extract_hog_features(image)
    
    # SIFT
    if method == 'SIFT':
        return extract_sift_features(image)
        
    # SURF
    if method == 'SURF':
        return extract_surf_features(image)
    
    if method == 'Combined':
        return extract_combined_features(image)


def create_pixel_features(image, mask, method):
    X = extract_features(image, method)
    y = np.apply_along_axis(lambda x: x[0], 1, mask.reshape(-1,3))
    return X, y

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

pathname = 'Dataset-small/data/WildScenes/WildScenes2d/V-01/'
sample_size = 100
target_size = (256, 256)

def resize_image_and_mask(image, mask, target_size):
    image_resized = cv2.resize(image, target_size)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized

X_all = []
y_all = []

image_filenames = [img_name for img_name in os.listdir(f'{pathname}image') if ':' not in img_name]
np.random.shuffle(image_filenames)
selected_filenames = image_filenames[:sample_size]

for img_name in selected_filenames:
    img = np.asarray(cv2.imread(f'{pathname}/image/{img_name}'))
    mask = np.asarray(cv2.imread(f'{pathname}/indexLabel/{img_name}'))
    img, mask = resize_image_and_mask(img, mask, target_size)
    img, mask = create_pixel_features(img, mask, method='Combined')
    X_all.append(img)
    y_all.append(mask)

X = np.array(X_all)
y = np.array(y_all)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import jaccard_score, accuracy_score
y_pred = np.concatenate(y_pred)
y_test = np.concatenate(y_test)
iou = jaccard_score(y_pred, y_test, average=None)
print(f'IoU for each class: {iou}')
print(f'Mean IoU: {np.mean(iou)}')
print(f'Accuracy: {accuracy_score(y_pred, y_test)}')