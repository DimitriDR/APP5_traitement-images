import math

import cv2
import numpy as np


def filtre_accumulateur(acc, degrade, direction):
    acc_filtre = np.zeros_like(acc)
    for r, c, rad in np.ndindex(acc.shape):
        grad_dir = direction[r, c]
        if np.abs(grad_dir) < 15 or np.abs(grad_dir - 180) < 15:
            if degrade[r, c] > t:
                acc_filtre[r, c, rad] = acc[r, c, rad]
            acc_filtre[r, c, rad] = acc[r, c, rad]
    return acc_filtre


if __name__ == '__main__':
    filename = 'four.png'
    # filename = 'fourn.png'
    # filename = 'coins.png'
    # filename = 'coins2.jpg'

    image_c = cv2.imread(filename)
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    N = 4  # nombre de cercles à détecter
    max_radius = int(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))  # la diagonale de l'image
    blur_kernel = (3, 3)

    # Pour le mesurer le temps d'exécution
    # start = cv2.getTickCount()

    if image is None:
        print("L'image n'a pas pu être chargée.")
    else:
        # 1.
        blur_sigma_X = 0  # 0 = auto
        image = cv2.GaussianBlur(image, blur_kernel, blur_sigma_X)

        # 2. et 3
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        cv2.imwrite('gradient.png', gradient_magnitude)

        t = 0.3 * np.max(gradient_magnitude)
        contour_pixels = [(x, y) for x, y in zip(*np.where(gradient_magnitude > t))]
        print(contour_pixels)

        # 4
        acc = np.zeros((image.shape[0], image.shape[1], max_radius))

        # 5
        i = 0
        for contour in contour_pixels:
            print("{}/{}".format(i, len(contour_pixels)))
            i += 1
            r, c = contour
            for r2 in range(image.shape[0]):
                for c2 in range(image.shape[1]):
                    rad = int(math.sqrt((r2 - r) ** 2 + (c2 - c) ** 2))
                    if 5 < rad < max_radius:
                        acc[r2, c2, rad] += 1 / rad

        gradient_direction = np.arctan2(sobel_y, sobel_x)

        filtre_acc = filtre_accumulateur(acc, gradient_magnitude, gradient_direction)

        print("Fin de la boucle")

        # 6
        sorted_indices = np.unravel_index(np.argsort(filtre_acc, axis=None)[-N:], filtre_acc.shape)
        top_values = []
        clean_range = range(-2, 3)
        print("calculating top values")
        for i in range(N):
            print("{}/{}".format(i, N))
            sorted_indices = np.unravel_index(np.argsort(acc, axis=None)[-N:], acc.shape)
            r, c, rad = sorted_indices[0][i], sorted_indices[1][i], sorted_indices[2][i]
            value = acc[r, c, rad]
            top_values.append((value, (r, c, rad)))
            for rr in clean_range:
                for cc in clean_range:
                    for rrad in clean_range:
                        if 0 <= r + rr < acc.shape[0] and 0 <= c + cc < acc.shape[1] and 5 <= rad + rrad < max_radius:
                            acc[r + rr, c + cc, rad + rrad] = 0

        # 7
        for circle in top_values:
            center = (circle[1][1], circle[1][0])
            radius = circle[1][2]
            value = circle[0]
            cv2.circle(image_c, center, 1, (0, 0, 255), 1)
            cv2.circle(image_c, center, radius, (255, 255, 25), 1)
            print("Cercle détecté : centre = {}, rayon = {}, value {}".format(center, radius, value))

        # Pour le mesurer le temps d'exécution
        # end = cv2.getTickCount()
        # print("Temps d'exécution : {}s".format((end - start) / cv2.getTickFrequency()))

        # Affichage de l'image avec les cercles détectés
        cv2.imshow('Cercles détectés', image_c)
        cv2.imwrite('cercles.png', image_c)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
