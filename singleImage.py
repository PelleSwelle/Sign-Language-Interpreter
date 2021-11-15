import cv2 as cv
import argparse
import PreProcessing
from PIL import Image
from matplotlib import pyplot as plt
import math
import numpy as np
from copy import copy
import canny_edge_detector

referenceDir = "./reference/"

# read reference images using pillow
SIGN_A = Image.open(referenceDir + "A1008.jpg")
SIGN_F = Image.open(referenceDir + "F1001.jpg")
SIGN_P = Image.open(referenceDir + "P1014.jpg")

parser = argparse.ArgumentParser()  # argument parser

# parser.add_argument("square", help="display a square of a given number", type=int)
parser.add_argument("handpose", help="set input image as A, F or P")
args = parser.parse_args()

image = SIGN_A

if args.handpose == "a":
    print("processing sign for A")
    image = SIGN_A
elif args.handpose == "f":
    print("processing sign for F")
    image = SIGN_F
elif args.handpose == "p":
    print("processing sign for P")
    image = SIGN_P

img_grayscaled = PreProcessing.grayScale(image)

img_blurred = PreProcessing.blur_gaussian(np.array(img_grayscaled), 5)

# th = threshold value, img_thresholded is the image as an array
th, img_thresholded = cv.threshold(img_grayscaled, 120, 255, cv.THRESH_BINARY)  # TODO make own thresholder
img_canny = cv.Canny(img_thresholded, 100, 200)  # TODO make our own edge detection algorithm
# img_isolated = PreProcessing.removeOtherStuff(img_canny)

cannied = canny_edge_detector.CannyEdgeDetector

img_good_features_to_track = cv.goodFeaturesToTrack(image, 5, 0.1, 2)
img_good_features_to_track = np.int0(img_good_features_to_track)

for i in img_good_features_to_track:
    x, y = i.ravel()
    cv.circle(img_thresholded, (x, y), 3, 255, -1)

# cv.imshow("cannied", np.array(cannied))

# cv.imshow("isolated", img_isolated)

# find out which shape is hand
# if shape not hand, remove

# ************************************** DISPLAYING WINDOWS **************************************
step_one = image
step_two = img_blurred
step_three = img_thresholded
step_four = img_canny
step_five = img_good_features_to_track
# *******************         STEP ONE         *******************
stepOneTitle = str(step_one)
cv.namedWindow(stepOneTitle)
cv.imshow(stepOneTitle, np.array(step_one))

# *******************         STEP TWO        *******************
stepTwoTitle = str(step_two)
cv.namedWindow(stepTwoTitle)
cv.imshow(stepTwoTitle, np.array(step_two))

# *******************      STEP THREE     *******************
stepThreeTitle = str(step_three)
cv.namedWindow(stepThreeTitle)
cv.imshow(stepThreeTitle, np.array(step_three))

# *******************      STEP FOUR     *******************
stepFourTitle = str(step_four)
cv.namedWindow(stepFourTitle)
cv.imshow(stepFourTitle, np.array(step_four))

# *******************      STEP FIVE     *******************
stepFiveTitle = str(step_five)
cv.namedWindow(stepFiveTitle)
plt.imshow(stepFiveTitle), plt.show()
#cv.imshow(stepFiveTitle, np.array(step_five))

# cv.imshow("default gaus", cv.GaussianBlur(np.array(input_grayscaled), (5, 5), 0))


cv.waitKey(0)
cv.destroyAllWindows()
