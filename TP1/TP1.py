import cv2

img = cv2.imread("./assets/Michelangelo_ThecreationofAdam_1707x775.jpg", cv2.IMREAD_COLOR)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()