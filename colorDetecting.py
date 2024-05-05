from ultralytics import YOLO
import torch
import cv2
import numpy as np

model = YOLO('yolov8n-seg.pt')

def detect_color(image_path):

	#EXTRACT THE IMAGE START
	predict = model.predict(image_path, save=False, classes=[0], save_txt=False)

	# Extract human mask
	human_mask = (predict[0].masks.data[0].cpu().numpy() * 255).astype("uint8")

	# Load original image
	image = cv2.imread(image_path)

	# Resize human mask to match the dimensions of the original image
	human_mask_resized = cv2.resize(human_mask, (image.shape[1], image.shape[0]))

	# Create mask for background (inverse of human mask)
	background_mask = cv2.bitwise_not(human_mask_resized)

	# Set background to black
	image[background_mask == 255] = [0, 0, 0]

	# Convert human mask to binary
	human_mask_binary = cv2.threshold(human_mask_resized, 127, 255, cv2.THRESH_BINARY)[1]

	# Extract human's color
	human_color = cv2.bitwise_and(image, image, mask=human_mask_binary)
	# EXTRACT IMAGE END


	# COLOR DETECTION START
	input_image = human_color

	color_ranges = [
		((94, 80, 2), (120, 255, 255)), # Blue
		((25, 52, 72), (102, 255, 255)), # Green
		((0, 50, 50), (10, 255, 255)) # Red
	]

	# Create lists to store masks and areas
	# masks -> color masks (lower and upper values interval)
	# areas -> total count of pixels for each color
	masks = []
	areas = []

	# converting the input image to HSV values from BGR values
	hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

	# for:
	# 1st loop:
	# lower blue -> (94, 80, 2)
	# upper blue -> (120, 255, 255)
	# 2nd loop:
	# lower green -> (25, 52, 72)
	# upper green -> (102, 255, 255)
	# 3rd loop:
	# lower red -> (0, 50, 50)
	# upper red -> (10, 255, 255)
	for lower, upper in color_ranges:
		
		# creating masks for specific color intervals
		mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
		
		# append new mask to masks[] list
		masks.append(mask)
		
		# Getting contour values from mask values
		contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		# Calculate the area of contours and add them to the areas[]list
		total_area = 0
		for contour in contours:
			area = cv2.contourArea(contour)
			total_area += area
		
		# append the total area to areas[]list
		areas.append(total_area)


	max_area_index = areas.index(max(areas))
	colors = ['Blue', 'Green', 'Red']
	max_color = colors[max_area_index]

	return max_color

