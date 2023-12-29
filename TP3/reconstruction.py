import os
import sys

import cv2
import numpy as np


def recontruct_image(original_image, fragments_dir: str, result_file: str, output_image: str,
                     print_index: bool = False):
    img_width = cv2.imread(original_image, cv2.IMREAD_UNCHANGED).shape[1]
    img_height = cv2.imread(original_image, cv2.IMREAD_UNCHANGED).shape[0]
    reconstructed_image = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    reconstructed_image.fill(255)

    with open(result_file, "r") as frag_direction_file:
        for line in frag_direction_file:
            line: str = line.strip()
            parts: list[str] = line.split(" ")

            number: int = int(float(parts[0]))
            x: int = int(float(parts[1]))
            y: int = int(float(parts[2]))
            angle: float = float(parts[3])
            frag_location: str = os.path.join(fragments_dir, "frag_eroded_" + parts[0] + ".png")
            fragment = cv2.imread(frag_location, cv2.IMREAD_UNCHANGED)
            h = fragment.shape[0]
            w = fragment.shape[1]

            # print("fragment ", fragment.shape)
            M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=1)
            fragment = cv2.warpAffine(fragment, M, (w, h))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = .5
            font_color = (0, 255, 0)  # BGR color (green in this case)
            thickness = 1

            # Use the putText function to add text to the image

            x_pos = x - w // 2
            y_pos = y - h // 2
            try:
                for y_frag in range(h):
                    for x_frag in range(w):
                        alpha = fragment[y_frag, x_frag][3]
                        if alpha > 0:
                            reconstructed_image[y_pos + y_frag, x_pos + x_frag, :3] = fragment[y_frag, x_frag][:3]
                            # to debug more easily
                            if print_index:
                                cv2.putText(reconstructed_image, str(number), (x, y), font, font_scale, font_color,
                                            thickness)


            except:
                print("error ", frag_location)

    cv2.imwrite(output_image, reconstructed_image)

    cv2.imshow("image", reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kp_and_desc_from(algorithm_name: str, image: np.ndarray):
    """
    Fonction permettant de calculer les points d'intérêt et les descripteurs d'une image selon un algorithme donné
    :param algorithm_name: Nom de l'algorithme à utiliser
    :param image: Descripteur de fichier OpenCV vers l'image
    :return: Points d'intérêt et descripteurs
    """
    key_points, descriptors = None, None

    if algorithm_name.upper() == "SIFT":
        sift = cv2.SIFT_create()
        key_points, descriptors = sift.detectAndCompute(image, None)

    # elif algorithm_name.upper() == "SURF":
    #     surf = cv2.SURF_create()
    #     key_points, descriptors = surf.detectAndCompute(image, None)
    #
    # elif algorithm_name.upper() == "ORB":
    #     orb = cv2.ORB_create()
    #     key_points, descriptors = orb.detectAndCompute(image, None)
    #
    elif algorithm_name.upper() == "FAST":
        fast = cv2.FastFeatureDetector_create()
        brisk = cv2.BRISK_create()
        # fast.setNonmaxSuppression(0)
        key_points = fast.detect(image, None)
        key_points, descriptors = brisk.compute(image, key_points)

    return key_points, descriptors


def print_soluce_for_fragment(fragment_index: str):
    with open("solution_fragments.txt", "r") as result_file:
        for line in result_file:
            line: str = line.strip()
            parts: list[str] = line.split(" ")
            if parts[0] == fragment_index:
                print(f"fragment {fragment_index} : x = {parts[1]}, y = {parts[2]}, angle = {parts[3]}")
                break


def get_closest_vote(votes_dict, tx, ty, angle_deg, range_threshold=2):
    for (voted_tx, voted_ty, voted_angle), count in votes_dict.items():
        if (
                # abs(voted_tx - tx) <= range_threshold and
                # abs(voted_ty - ty) <= range_threshold and
                abs(voted_angle - angle_deg) <= range_threshold):
            return (voted_tx, voted_ty, voted_angle)
    return None


def main():
    # Vérification des arguments
    if len(sys.argv) != 3:
        print("Usage : python exercice_1.py <image_cible> <dossier_fragments>", file=sys.stderr)
        sys.exit(1)

    ALGORITHM_NAME = "FAST"
    THRESHOLD = 100
    # Le tableau
    target_image = sys.argv[1]
    # Dossier fragments
    fragments_dir = sys.argv[2]

    original_img = cv2.imread(
        'C:\\Users\ddbtn\PycharmProjects\APP5_traitement-images\TP3\Michelangelo_ThecreationofAdam_1707x775.jpg')
    target_image_opencv = cv2.imread(target_image, cv2.IMREAD_GRAYSCALE)

    # On fait récupère l'index et le nom de chaque fragment
    fragments = []

    for filename in os.listdir(fragments_dir):
        if os.path.isfile(os.path.join(fragments_dir, filename)):  # and filename.find("_24.") != -1:
            index = filename.split(".")[-2].split("_")[-1]
            fragments.append((filename, index))

    fragments.sort(key=lambda x: int(x[1]))

    # On récupère les points d'intérêt et descripteurs de l'image cible
    target_kp, target_desc = kp_and_desc_from(ALGORITHM_NAME, target_image_opencv)
    # Liste des résultats
    results: list = []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for fragment in fragments:
        fragment_path = os.path.join(fragments_dir, fragment[0])
        fragment_index = fragment[1]
        # if fragment_index == index_to_test:
        #     print("fragment index", fragment_index)
        print(f"{int(fragment_index) + 1}/{len(fragments)}")

        original_fragment = cv2.imread(fragment_path)

        fragment_image_opencv = cv2.imread(fragment_path, cv2.IMREAD_GRAYSCALE)
        fragment_kp, fragment_desc = kp_and_desc_from(ALGORITHM_NAME, fragment_image_opencv)

        matches = matcher.match(fragment_desc, target_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        print("matches ", len(matches))

        good_matches: list[cv2.DMatch] = []
        # filter to have the best matches | or just matches[:20]
        # for m in matches:
        #     if m.distance < THRESHOLD:
        #         good_matches.append(m)

        print("good_matches ", len(good_matches))

        print("Index ", fragment_index)
        print_soluce_for_fragment(fragment_index)
        # if fragment_index == index_to_test:
        # iii = cv2.drawMatches(original_fragment, fragment_kp,
        #                       original_img, target_kp, good_matches, None,
        #                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow("matches", iii)
        # cv2.waitKey(0)
        votes_angle = {}


        i = 0
        max_iter = 10000

        if len(matches) < 2:
            continue

        while i < max_iter:
            m, m1 = np.random.choice(matches, 2, replace=True)

            x1, y1 = fragment_kp[m1.queryIdx].pt
            x2, y2 = fragment_kp[m.queryIdx].pt

            x3, y3 = target_kp[m1.trainIdx].pt
            x4, y4 = target_kp[m.trainIdx].pt

            angle_frag = np.arctan2(y1 - y2, x1 - x2)
            angle_targ = np.arctan2(y3 - y4, x3 - x4)

            angle_diff = angle_frag - angle_targ
            angle_deg = int(np.degrees(angle_diff))

            size = original_fragment.shape[0]  # pareil pour width
            # coordonate based on the middle of the fragment
            x_frag = x1 - size // 2
            y_frag = y1 - size // 2

            tx = int(float(x3 - (x_frag * np.cos(angle_diff) - x_frag * np.sin(angle_diff))))
            ty = int(float(y3 - (y_frag * np.sin(angle_diff) + y_frag * np.cos(angle_diff))))

            # if votes_angle.get((tx, ty, angle_deg)) is not None:
            #     votes_angle[(tx, ty, angle_deg)] += 1
            # else:
            #     votes_angle[(tx, ty, angle_deg)] = 1

            i += 1
            closest_vote = get_closest_vote(votes_angle, tx, ty, angle_deg)
            if closest_vote:
                votes_angle[closest_vote] += 1
            else:
                votes_angle[(tx, ty, angle_deg)] = 1

        if len(votes_angle) == 0:
            continue

        chosen_angle = max(votes_angle, key=votes_angle.get)

        results.append((fragment_index,
                        round(chosen_angle[0]),
                        round(chosen_angle[1]),
                        round(chosen_angle[2])
                        )
                       )
        votes_angle = sorted(votes_angle.items(), key=lambda x: x[1], reverse=True)
        print("votes_angle ", votes_angle[:10])
        print("chooooosen ", chosen_angle)
        print("shape frag", original_fragment.shape)
        print("----------------------------------")

    results.sort(key=lambda x: int(x[0]))
    with open("reconstruction_results.txt", "w") as result_file:
        for result in results:
            result_file.write("{} {} {} {}\n".format(result[0], result[1], result[2], result[3]))


if __name__ == '__main__':
    main()
    recontruct_image(sys.argv[1], sys.argv[2],
                     "reconstruction_results.txt", "result.png", True)
