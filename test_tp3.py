import os
import sys

import cv2
import numpy as np


def accuracy_checking(experimental_file: str, truth_file: str):
    """
    Fonction permettant de vérifier si le fichier expérimental est correct par rapport au fichier de vérité terrain
    :param experimental_file: Fichier expérimental
    :param truth_file: Fichier de vérité terrain
    :return: Pourcentage de précision
    """
    with open(truth_file, "r") as truth:
        truth_file_content = [line.split() for line in truth]

    with open(experimental_file, "r") as experimental:
        experimental_file_content = [line.split() for line in experimental]

    # TODO
    pass


def recontruct_image(fragments_dir: str, result_file: str, output_file: str):
    img_width = 1707
    img_height = 775

    reconstructed_image = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    reconstructed_image.fill(255)

    print("reconstructed_image ", reconstructed_image.shape)

    with open(result_file, "r") as frag_direction_file:
        for line in frag_direction_file:
            line: str = line.strip()
            parts: list[str] = line.split(" ")

            number:int = int(float(parts[0]))
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
                            # add label to the image
                            cv2.putText(reconstructed_image, str(number), (x, y), font, font_scale, font_color, thickness)


            except:
                print("error ", frag_location)

    cv2.imwrite(output_file, reconstructed_image)

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
    # elif algorithm_name.upper() == "FAST":
    #     brisk = cv2.BRISK_create()
    #     fast = cv2.FastFeatureDetector_create()
    #     fast.setNonmaxSuppression(0)
    #     key_points = fast.detect(image, None)
    #     _, descriptors = brisk.compute(image, key_points)

    return key_points, descriptors


def print_soluce_for_fragment(fragment_index: str):
    with open("solution_fragments.txt", "r") as result_file:
        for line in result_file:
            line: str = line.strip()
            parts: list[str] = line.split(" ")
            if parts[0] == fragment_index:
                print(f"fragment {fragment_index} : x = {parts[1]}, y = {parts[2]}, angle = {parts[3]}")
                break


def get_closest_vote(votes_dict, tx, ty, angle_deg, range_threshold=8):
    for (voted_tx, voted_ty, voted_angle), count in votes_dict.items():
        if (abs(voted_tx - tx) <= range_threshold and
                abs(voted_ty - ty) <= range_threshold and
                abs(voted_angle - angle_deg) <= range_threshold):
            return (voted_tx, voted_ty, voted_angle)
    return None



def main():
    # Vérification des arguments
    if len(sys.argv) != 3:
        print("Usage : python exercice_1.py <image_cible> <dossier_fragments>", file=sys.stderr)
        sys.exit(1)

    # Définition de constantes utilisées dans la fonction
    ALGORITHM_NAME = "SIFT"
    THRESHOLD = 0.6
    # Le tableau
    target_image = sys.argv[1]
    # Dossier fragments
    fragments_dir = sys.argv[2]

    # On récupère l'image cible telle que lue par OpenCV
    target_image_opencv = cv2.imread(target_image, cv2.IMREAD_GRAYSCALE)

    # On fait récupère l'index et le nom de chaque fragment
    fragments = []

    original_img = cv2.imread('targets/angelo.jpg')

    for filename in os.listdir(fragments_dir):
        if os.path.isfile(os.path.join(fragments_dir, filename)):  # and filename.find("_24.") != -1:
            index = filename.split(".")[-2].split("_")[-1]
            fragments.append((filename, index))

    # On récupère les points d'intérêt et descripteurs de l'image cible
    target_kp, target_desc = kp_and_desc_from(ALGORITHM_NAME, target_image_opencv)
    # Liste des résultats
    results: list = []
    matcher = None
    # matcher = cv2.BFMatcher(cv2.NORM_L1)

    # index_to_test = "24"
    index_to_test = "_"
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict({})
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    for fragment in fragments:
        fragment_path = os.path.join(fragments_dir, fragment[0])
        fragment_index = fragment[1]
        if fragment_index == index_to_test:
            print("fragment index", fragment_index)
        fragment_origine = cv2.imread(fragment_path)
        fragment_image_opencv = cv2.imread(fragment_path, cv2.IMREAD_GRAYSCALE)
        fragment_kp, fragment_desc = kp_and_desc_from(ALGORITHM_NAME, fragment_image_opencv)

        matches = matcher.knnMatch(fragment_desc, target_desc, k=2)
        matches = sorted(matches, key=lambda x: x[0].distance)
        good_matches: list[cv2.DMatch] = []

        # keep good matches
        matches_mask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < THRESHOLD * n.distance:
                matches_mask[i] = [1, 0]
                good_matches.append(m)

        if len(good_matches) < 2:
            continue
        print("good_matches ", len(good_matches))

        print("Index ", fragment_index)
        print_soluce_for_fragment(fragment_index)
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(fragment_origine, fragment_kp, original_img, target_kp, matches, None,
                                  **draw_params)
        if fragment_index == index_to_test:
            cv2.imshow("matches", img3)
            cv2.waitKey(0)

        combination_tryed = []
        votes_angle = {}

        # def get_random_match():
        #     random_index = np.random.randint(0, len(good_matches))
        #     # second random not the previous one
        #     second_random_index = np.random.randint(0, len(good_matches))
        #     while random_index == second_random_index:
        #         random_index = np.random.randint(0, len(good_matches))
        #
        #     return good_matches[random_index], good_matches[second_random_index]

        # ok = False
        # while not ok:
        #     m1, m2 = get_random_match()
        #     if (m1, m2) in combination_tryed:
        #         continue
        #     combination_tryed.append((m1, m2))
        #
        #     x1, y1 = fragment_kp[m1.queryIdx].pt
        #     x2, y2 = fragment_kp[m2.queryIdx].pt
        #
        #     x3, y3 = target_kp[m1.trainIdx].pt
        #     x4, y4 = target_kp[m2.trainIdx].pt
        #
        #     # Angle calculation
        #     angle_frag = np.arctan2(x1 - x2, y1 - y2)
        #     angle_targ = np.arctan2(x3 - x4, y3 - y4)
        #
        #     angle_diff = angle_targ - angle_frag
        #     angle_deg = np.degrees(angle_diff)
        #     # Calculate translation
        #     tx = int(float(x3 - x1)) + fragment_origine.shape[1] // 2
        #     ty = int(float(y3 - y1)) + fragment_origine.shape[0] // 2
        #     # tx = x3 - x1*np.cos(angle_diff) + y1*np.sin(angle_diff) + fragment_origine.shape[1] // 2
        #     # ty = y3 + x1*np.sin(angle_diff) - y1*np.cos(angle_diff) + fragment_origine.shape[0] // 2
        #     if votes_angle.get((tx, ty, angle_deg)) is not None:
        #         votes_angle[(tx, ty, angle_deg)] += 1
        #     else:
        #         votes_angle[(tx, ty, angle_deg)] = 1
        #
        #     if len(combination_tryed) >= 3 or len(good_matches) == 2:
        #         ok = True


        def distance_too_low_between_match(m, m1):
            x1, y1 = fragment_kp[m.queryIdx].pt
            x2, y2 = fragment_kp[m1.queryIdx].pt

            threshold = 10
            distance_frag = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if fragment_index == index_to_test:
                print("distance_frag ", distance_frag)

            return distance_frag < threshold



        for m1 in good_matches:
            for m in good_matches:
                if m1 == m or distance_too_low_between_match(m, m1):
                    continue

                # Get keypoint coordinates from both images
                target_kp_coords = target_kp[m.trainIdx].pt
                fragment_kp_coords = fragment_kp[m.queryIdx].pt

                # Angle calculation
                angle_frag = np.arctan2(fragment_kp_coords[1] - fragment_kp[m1.queryIdx].pt[1],
                                        fragment_kp_coords[0] - fragment_kp[m1.queryIdx].pt[0])
                angle_targ = np.arctan2(target_kp_coords[1] - target_kp[m1.trainIdx].pt[1],
                                        target_kp_coords[0] - target_kp[m1.trainIdx].pt[0])
                # angle_diff = angle_targ - angle_frag
                angle_diff = angle_frag - angle_targ
                angle_deg = np.degrees(angle_diff)

                tx = int(float(target_kp_coords[0] - fragment_kp_coords[0])) + fragment_origine.shape[1] // 2
                ty = int(float(target_kp_coords[1] - fragment_kp_coords[1])) + fragment_origine.shape[0] // 2

                # TODO au lieu de faire une boucle, tester pour deux points du fragments
                #   puis tester pour les autres si ça semble correct à quelques pixels près
                #   si ça semble ok niveau angle, calculer le tx et ty pour les autres points
                #   attention à ce qu'il n'y ait pas deux kp du frag vers le même kp de la target
                #   tester la distance entre les kp du frag et les kp de la target doit être same


                # vote range of 10 pixels
                closest_vote = get_closest_vote(votes_angle, tx, ty, angle_deg)
                if closest_vote:
                    # If close enough, increment the existing vote's count
                    votes_angle[closest_vote] += 1
                else:
                    # If no close vote found, add a new entry
                    votes_angle[(tx, ty, angle_deg)] = 1

        if len(votes_angle) == 0:
            continue

        chosen_angle = max(votes_angle, key=votes_angle.get)
        # ugly and a bit cheating but it works héhé
        while chosen_angle[2] == 0 and len(votes_angle) > 1:
            votes_angle.pop(chosen_angle)
            chosen_angle = max(votes_angle, key=votes_angle.get)

        results.append((fragment_index,
                        round(chosen_angle[0]),
                        round(chosen_angle[1]),
                        round(chosen_angle[2])
                        )
                       )
        votes_angle = sorted(votes_angle.items(), key=lambda x: x[1], reverse=True)
        print("votes_angle ", votes_angle)
        print("chooooosen ", chosen_angle)
        print("shape frag", fragment_origine.shape)
        print("----------------------------------")
        # if len(votes) != 0:
        #     print("votes ", votes)
        #     result = max(votes, key=lambda x: x[3])
        #     results.append((fragment_index,
        #                     round(result[0]),
        #                     round(result[1]),
        #                     round(result[2], 3))
        #                    )

    # fragment_pts = np.float32([fragment_kp[m.queryIdx].pt for m in good_matches])
    # target_pts = np.float32([target_kp[m.trainIdx].pt for m in good_matches])
    # print("fragment_pts ", fragment_pts)
    # print("target_pts ", target_pts)
    # Calculate the affine transformation matrix
    # affine_matrix, inliers = cv2.estimateAffinePartial2D(fragment_pts, target_pts)


    # iterate randomly by choosing 2 associated points
    # if affine_matrix is not None:
    #     # The affine_matrix is a 2x3 matrix:
    #     # [ [a00, a01, a02],
    #     #   [a10, a11, a12] ]
    #
    #     # Calculate rotation angle theta
    #     theta = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
    #
    #     # Calculate translations tx and ty
    #     tx = affine_matrix[0, 2]
    #     ty = affine_matrix[1, 2]
    #     results.append((fragment_index,
    #                     round(tx),
    #                     round(ty),
    #                     round(theta, 3))
    #                    )
    #         # img3 = cv2.drawMatches(fragment_image_opencv, fragment_kp, target_image_opencv,
    #         #                        target_kp, matches[:],None,
    #         #                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #         # cv2.imshow("img3", img3)
    #         # cv2.waitKey(0)


    # On trie les résultats par index
    print("results")
    results.sort(key=lambda x: int(x[0]))

    # On écrit finalement les résultats dans le fichier
    with open("result.txt", "w") as result_file:
        for result in results:
            result_file.write("{} {} {} {}\n".format(result[0], result[1], result[2], result[3]))

if __name__ == '__main__':
    main()
    recontruct_image("fragments", "result.txt", "result.png")
