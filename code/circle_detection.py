import math
import sys
from turtle import distance
import cv2
import numpy as np
from collections import defaultdict

def hough_circles_detection(image, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold,gt_values):
  # Get image size from the image
  image_height, image_width = edge_image.shape[:2]
  
  # R and Theta ranges
  dtheta = int(360 / num_thetas)
  
  # Thetas is bins created from 0 to 360 degree with increment of the dtheta
  thetas = np.arange(0, 360, step=dtheta)
  
  ## Radius ranges from r_min to r_max 
  rs = np.arange(r_min, r_max, step=delta_r)
  
  cos_thetas_value = np.cos(np.deg2rad(thetas))
  sin_thetas_value = np.sin(np.deg2rad(thetas))

  circle_candidates = []
  for r in rs:
    for t in range(num_thetas):
      circle_candidates.append((r, int(r * cos_thetas_value[t]), int(r * sin_thetas_value[t])))

  # Create an accumulator
  accumulator_array = defaultdict(int)

  for y in range(image_height):
    for x in range(image_width):
      # Check white pixel
      if edge_image[y][x] != 0:
        # Find and vote the circle from candidate circles passing through a found edge pixel.
        for r, rcos_t, rsin_t in circle_candidates:
          x_center = x - rcos_t
          y_center = y - rsin_t
          # +1 for vote for current candidate
          accumulator_array[(x_center, y_center, r)] += 1
  
  # Assignment of the incoming picture to draw the circles on the incoming picture(output_image)
  output_image = image.copy()

  # Output list of detected circles
  # A single circle has be a tuple of (x,y,r,threshold)
  finded_circles = []
  
  # Sort the accumulator based on the votes for the candidate circles 
  for candidate_circle, votes in sorted(accumulator_array.items(), key=lambda i: -i[1]):
    x, y, r = candidate_circle
    # Calculate vote percentage
    current_vote_percentage = votes / num_thetas
    if current_vote_percentage >= bin_threshold: 
      # Shortlist the circle for final result
      finded_circles.append((x, y, r, current_vote_percentage))
      print(x, y, r, current_vote_percentage)
      


  pixel_threshold = 30
  process_circles = []
  circles=[]
  for x, y, r, v in finded_circles:
  # Remove nearby duplicates by pixel_threshold
    if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in process_circles):
       process_circles.append((x, y, r, v))
       circles.append((x, y,r))

  #The determined circles are ready
  out_circles = process_circles
  #circles=gt_values
  '''
  circ = []
  for i in gt_values:
      nearest = min(circles, key=lambda x: distance(x, i))
      print(nearest)
  '''
  iou_scores_for_average=[]
  # Calculating IOU Score
  for i in range(0, len(gt_values)):
      iou_scores = []
      for j in range(0, len(circles)):

          x0 = gt_values[i][0]
          x1 = circles[j][0]

          y0 = gt_values[i][1]
          y1 = circles[j][1]

          r0 = gt_values[i][2]
          r1 = circles[j][2]

          sqrt_r0 = r0 * r0
          sqrt_r1 = r1 * r1

          distance = math.sqrt(((x1 - x0) * (x1 - x0) )+ ((y1 - y0) * (y1 - y0)))
          area_of_union = (math.pi * r0 * r0) + (math.pi * r1 * r1)

          #There is no intersection so iou score is zero.
          if (distance > r1 + r0):
              iou_scores.append(0)

          elif ((distance <= np.abs(r0 - r1)) & (r0 >= r1)):
              area_of_intersects = math.pi * sqrt_r1
              iou = area_of_intersects / (area_of_union - area_of_intersects)
              iou_scores.append(iou)

          elif ((distance <= np.abs(r0 - r1)) & (r0 < r1)):
              area_of_intersects = math.pi * sqrt_r0
              iou = area_of_intersects / (area_of_union - area_of_intersects)
              iou_scores.append(iou)

          else:
              phi = (math.acos((sqrt_r0 + (distance * distance) - sqrt_r1) / (2 * r0 * distance))) * 2
              theta = (math.acos((sqrt_r1 + (distance * distance) - sqrt_r0) / (2 * r1 * distance))) * 2
              area1 = 0.5 * theta * sqrt_r1 - 0.5 * sqrt_r1 * math.sin(theta)
              area2 = 0.5 * phi * sqrt_r0 - 0.5 * sqrt_r0 * math.sin(phi)
              area_of_intersects = area1 + area2
              iou = area_of_intersects / (area_of_union - area_of_intersects)
              iou_scores.append(iou)

      if (len(iou_scores) != 0):
          iou_scores_for_average.append(float('{0:.3g}'.format(max(iou_scores))))
          print("Max IOU Score of " + str(i+1) + ". Circle: "+'{0:.3g}'.format(max(iou_scores)))

      else:
          print("There is no circle !")


  # Draw circles according to out_circles on the output image
  for x, y, r, v in out_circles:
    output_image = cv2.circle(output_image, (x,y), r, (0,0,255), 2)

  print("Average IOU score for this image: " + str('{0:.3g}'.format(np.mean(iou_scores_for_average))))
  return output_image, out_circles

def main():

    # Threshold Assignments
    # 27 25 30 40 47 49 46 52 65 55 68 69 88 94 100 106 108  113 101
    image_name =sys.argv[1]
    image_path = "dataset MY/Images/" + image_name + ".jpg"
    GT_file = open("dataset MY/GT/" + image_name + ".txt")
    gt_values = GT_file.readlines()
    values=[];
    #print(gt_values)

    #Adding (x,y,r) to list from GT files
    for j in range(1,len(gt_values)):
        x, y, r = (int(float(i))for i in gt_values[j].split())
        values.append((x,y,r))


    #Finding max and min radius for r range
    r=[]
    r_min=10
    r_max=200
    for a, b, c in values:
        r.append(c)

    if r:
        r_min = min(r)-5
        r_max = max(r)+5


    print("R range: "+str(r_min)+" "+str(r_max))
    delta_r = 1
    num_thetas = 100
    bin_threshold = 0.55  #for voting
    min_edge_threshold = 75 # for Canny
    max_edge_threshold = 150  # for Canny

    if r_max - r_min <= 10:
        bin_threshold=0.45
    if r_max-r_min>=50:
        bin_threshold=0.5

    # Read the image file by using OpenCV library
    input_img = cv2.imread(image_path)

    #Blurring of images by using GaussianBlur
    gauss=cv2.GaussianBlur(input_img,(5,5),0)
    edge_image = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
    # Edge detection on the input image by using Canny Edge Detector
    edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)

    # Show edge detected image
    
    if edge_image is not None:
        
        print ("Detecting Hough Circles Started!")
        circle_img, circles = hough_circles_detection(input_img, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold,values)

        cv2.imshow('Original Image', input_img)
        cv2.imshow('Edge Image', edge_image)
        cv2.imshow('Detected Circles', circle_img)
        cv2.waitKey(0)

        if circle_img is not None:
            cv2.imwrite("circles_img.png", circle_img)
    else:
        print ("Error in input image!")
            
    print ("Detecting Hough Circles Complete!")

if __name__ == "__main__":
    main()
