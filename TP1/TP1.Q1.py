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
        # x_pos = int(x)
        # y_pos = int(y)

        # print("fragment ", fragment.shape)
        M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=1)
        fragment = cv2.warpAffine(fragment, M, (w, h))

        x_pos = x - w // 2
        y_pos = y - h // 2

        for y_frag in range(h):
            for x_frag in range(w):
                alpha = fragment[y_frag, x_frag][3]
                if alpha > 0:
                    reconstructed_image[y_pos + y_frag, x_pos + x_frag, :3] = fragment[y_frag, x_frag][:3]



cv2.imshow("image", reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()