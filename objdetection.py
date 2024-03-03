import cv2
import numpy as np

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Could not open camera")
    exit()

# Predefine kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Predefine Gaussian blur parameters
gaussian_kernel_size = (9, 9)
gaussian_sigma = 1

# Adjusted HSV range for red color to cover more shades
lower_range1 = np.array([0, 50, 50])    # lower range of red color in HSV
upper_range1 = np.array([10, 255, 255]) # upper range of red color in HSV
lower_range2 = np.array([170, 50, 50])  # lower range of red color in HSV (for hues around red)
upper_range2 = np.array([180, 255, 255])# upper range of red color in HSV (for hues around red)

# Minimum contour area threshold
min_contour_area = 1000

while True:
    # Read the next frame
    ret, frame = cap.read()

    # Check if the frame is not empty
    if not ret:
        break

    # Convert frame to HSV color space
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply Gaussian blur
    hsv_img = cv2.GaussianBlur(hsv_img, gaussian_kernel_size, gaussian_sigma)

    # Threshold to get mask for red color
    mask1 = cv2.inRange(hsv_img, lower_range1, upper_range1)
    mask2 = cv2.inRange(hsv_img, lower_range2, upper_range2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the larger contours
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Result', frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

