import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

list_fragments = ["7", "326", "24", "4", "33", "57", "55", "167", "262", "306"]
algorithms = ["SIFT", "ORB", "FAST"]


def kp_and_desc_from(algorithm_name: str, image: np.ndarray):
    key_points, descriptors = None, None

    if algorithm_name.upper() == "SIFT":
        sift = cv2.SIFT_create()
        key_points, descriptors = sift.detectAndCompute(image, None)

    elif algorithm_name.upper() == "ORB":
        orb = cv2.ORB_create()
        key_points, descriptors = orb.detectAndCompute(image, None)

    elif algorithm_name.upper() == "FAST":
        fast = cv2.FastFeatureDetector_create()
        brisk = cv2.BRISK_create()
        # fast.setNonmaxSuppression(0)
        key_points = fast.detect(image, None)
        key_points, descriptors = brisk.compute(image, key_points)

    return key_points, descriptors


def process_fragment_plot(fragment_path, target_path, algorithm):
    print("Frag ", fragment_path, " with ", algorithm)
    fragment_original = cv2.imread(fragment_path)
    original_img = cv2.imread(target_path)

    fragment = cv2.imread(fragment_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    f_kp, f_desc = kp_and_desc_from(algorithm, fragment)
    t_kp, t_desc = kp_and_desc_from(algorithm, target)
    matcher = None
    if algorithm in ["SIFT"]:
        f_desc = np.float32(f_desc)
        t_desc = np.float32(t_desc)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(f_desc, t_desc)
    matches = sorted(matches, key=lambda x: x.distance)[:15]

    match_img = cv2.drawMatches(fragment_original, f_kp, original_img, t_kp, matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # matplotlib doesn't handle colors the same way as opencv
    match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    return match_img_rgb


for fragment_index in list_fragments:
    frag_path = f'frag_eroded/frag_eroded_{fragment_index}.png'
    image_path = 'Michelangelo_ThecreationofAdam_1707x775.jpg'

    fig = plt.figure( figsize=(10, 10))
    rows = 3
    columns = 1
    for i, algo in enumerate(algorithms):
        result_img = process_fragment_plot(frag_path, image_path, algo)
        fig.add_subplot(rows, columns, i+1)
        plt.title(f"Fragment {fragment_index} - {algo}")
        plt.imshow(result_img)

    print(f"Save fragment {fragment_index}")
    if not os.path.exists("./exo1_imgs"):
        os.makedirs("./exo1_imgs")
    plt.savefig(f"./exo1_imgs/fragment_{fragment_index}.png")
