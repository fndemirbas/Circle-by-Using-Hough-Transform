# Hough Circle Detection
Implementation of Simple Hough Circle Detection Algorithm by using OpenCV in Python.\
The Hough transform is a specific feature extraction technique used in digital image processing. The purpose of this technique is to find instances of objects of a particular shape class through a voting procedure. This voting procedure is performed using accumulation arrays. Hough transform can be applied to detect many random shapes in an image.
## Run
''' python circle_detection.py img_name '''

##Implementation
• Step 1: Threshold assignments and read the image
imread() is in OpenCV is used to read an image from the filesystem.I had to normalize the
accumulator array after I had it to get a standard threshold value between 0 and 1. This
threshold determines the occlusion ration and, as a result, the circle detection sensitivity.
Threshold is 0.55 .

• Step 2: Gaussian Blur the image to reduce noise
GaussianBlur() is in OpenCV is used to blur the image to a specified kernel size. I used
Gaussian Blur mainly to blur the image and remove sharp noise in images. This will make
the Canny edge detector work correctly.

• Step 3:Edge detection using Canny Edge Detector
Canny() is used to detect edges in the image. Canny edge detector is a very good algorithm
for detecting edges in an image. It has two thresholds assigned to it, in optimization I
decided 25 and 150 are ideal thresholds for both.

• Step 4: Find circle candicates
Different r and theta combinations are kept in the circle_candidates array in the given
radius range and theta range.

• Step 5:Find and vote the circle from candidate circles passing through a found edge pixel.


    For every pixel in image
      For thetas and r in circle_candicates
        x = x0 – R*cos (theta)
        y = y0 – R*sin(theta)
        Accumulator[R][x][y]+=1


• Step 6: Construct circles for every r,x,y according to votes
Sort the accumulator based on the votes for the candidate circles .First of all, the vote
percentages of the candicate circles are calculated and the values that are equal to or greater
than the bin_threshold value are determined. Then values close to pixel_threshold are
deleted. The remaining values tell us the circle to be drawn.

• Step 7: Draw Circle
circle() is in OpenCV is used to draw circles onto an image at a particular pixel x,y with a
radius r.

##Functions
hough_circles_detection()

## Input
The script requires one positional argument and few optional parameters:
* image_name - name of the image file for circle detection
## Output
The output of the script would be two files:
* circle_img.png - Image with the Circles drawn in Red color.
* Detected Circle window
* Original Image window
* Edge Detected window
