import cv2
import numpy as np

img_width = 1707
img_height = 775

reconstructed_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

with open("./assets/fragments.txt", "r") as frag_direction_file:
    for line in frag_direction_file:
        line: str = line.strip()
        part: list[str] = line.split(" ")

        next_parts: list[str] = part[1:]

        x: int = int(next_parts[0])
        y: int = int(next_parts[1])
        angle: float = float(next_parts[2])

        frag_location: str = "./assets/frag_eroded/frag_eroded_" + part[0] + ".png"

        fragment = cv2.imread(frag_location)

        rows, cols, _ = fragment.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        fragment = cv2.warpAffine(fragment, M, (cols, rows))

        x_pos = int(x)
        y_pos = int(y)

        if x_pos >= 0 and y_pos >= 0 and x_pos + cols <= img_width and y_pos + rows <= img_height:
            reconstructed_image[y_pos:y_pos + rows, x_pos:x_pos + cols] = fragment

cv2.imshow("image", reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
