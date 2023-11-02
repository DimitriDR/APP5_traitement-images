import math

import cv2
import numpy as np

# filename = 'four.png'
# filename = 'fourn.png'
# filename = 'coins.png'
filename = 'coins2.jpg'
image_c = cv2.imread(filename)
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
N = 8  # nombre de cercles à détecter
# max_radius = 100  # en fonction de la taille de l'image et des pièces (en pixel)
max_radius = 50
# blur_kernel = (5, 5)
blur_kernel = (3, 3)

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
    # cv2.imshow('Gradient de Sobel', gradient_magnitude)
    # cv2.waitKey(0)
    cv2.imwrite('gradient.png', gradient_magnitude)

    t = 0.6 * np.max(gradient_magnitude)
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
        for rad in range(5, max_radius):
            for theta in range(360):
                r2 = int(r - rad * math.cos(theta * math.pi / 180))
                c2 = int(c - rad * math.sin(theta * math.pi / 180))
                if 0 <= r2 < image.shape[0] and 0 <= c2 < image.shape[1]:
                    acc[r2, c2, rad] += 1
                    # acc[r2, c2, rad] += (1/rad)  # Tentative de normalisation => consseillé en classe mais pas ouf ici, trouver que des petits cercles / cercles non existant, à débug car est théoriquement mieux pour pas favorisé les grands cercles
    # print(acc)
    print("end of loop")

    # 6
    sorted_indices = np.unravel_index(np.argsort(acc, axis=None)[-N:], acc.shape)
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
        cv2.circle(image_c, center, radius, (255, 255, 0), 1)
        print("Cercle détecté : centre = {}, rayon = {}, value {}".format(center, radius, value))

    # Affichez l'image avec les cercles détectés
    cv2.imshow('Cercles détectés', image_c)
    cv2.imwrite('cercles.png', image_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
