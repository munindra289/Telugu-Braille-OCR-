import cv2

image = cv2.imread("archive (16)/Braille Dataset/Braille Dataset/g1.JPG11dim.jpg")
image = cv2.resize(image, (28, 60))
cv2.imwrite("output.jpg", image)
