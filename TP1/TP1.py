import cv2
import numpy as np

img_width = 1707
img_height = 775

reconstructed_image = np.zeros((img_height, img_width, 4), dtype=np.uint8)
reconstructed_image.fill(255)
print("reconstructed_image ", reconstructed_image.shape)
with open("./assets/fragments.txt", "r") as frag_direction_file:
    for line in frag_direction_file:
        # print("-------------------")
        line: str = line.strip()
        parts: list[str] = line.split(" ")

        x: int = int(parts[1])
        y: int = int(parts[2])
        angle: float = float(parts[3])
        # print("x " + str(x) + " y " + str(y) + " angle " + str(angle) + " index " + parts[0])

        frag_location: str = "./assets/frag_eroded/frag_eroded_" + parts[0] + ".png"
        fragment = cv2.imread(frag_location, cv2.IMREAD_UNCHANGED)
        h = fragment.shape[0]
        w = fragment.shape[1]
        # height(nb rows), width(nb cols), channels <= .shape
        x_pos = int(x)
        y_pos = int(y)

        # print("fragment ", fragment.shape)
        M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=1)
        fragment = cv2.warpAffine(fragment, M, (w, h))

        # print("x_pos " + str(x_pos) + ", y_pos " + str(y_pos))

        # pour éviter les pbs d'out of range, on prend la différence et pas juste // 2 de chaque côté
        w_1 = w // 2
        h_1 = h // 2
        w_2 = w - w_1
        h_2 = h - h_1
        if x_pos - w_1 >= 0 \
                and y_pos - h_1 >= 0 \
                and x_pos + w_2 <= img_width \
                and y_pos + h_2 <= img_height:
            channels = cv2.split(fragment)
            alpha_channel = channels[3]
            alpha_mask = alpha_channel > 0  # on garde uniquement les pixels non à 0 sur le canal alpha

            reconstructed_image[y_pos - h_1:y_pos + h_2, x_pos - w_1:x_pos + w_2][alpha_mask] = \
                fragment[alpha_mask]
        else:
            print("out of range")

cv2.imshow("image", reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()