import cv2
import numpy as np

# img = cv2.imread("./assets/Michelangelo_ThecreationofAdam_1707x775.jpg", cv2.IMREAD_COLOR)

img_width = 1707
img_height = 775

# Creating a blank image to place fragments
reconstructed_image = np.zeros((img_height, img_width), np.uint8)

# TODO: Parcourir l'ensemble des images et selon la position des fragments indiquée dans le fichier fragments.txt
# TODO: les placer dans l'image finale



# Displaying the reconstructed image
cv2.imshow("image", reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()