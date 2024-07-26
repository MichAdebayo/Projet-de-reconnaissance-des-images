# Load libraries
import cv2
import numpy as np


# Load the images
img_gray = cv2.imread('Original_coins_image.png', cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread('Original_coins_image', 1)

# Apply Gaussian Blur to gray scale image to reduce noise
blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Detect edges using Canny Edge Detector
edges = cv2.Canny(blurred_img, 12, 30)

# Apply morphological operations to remove small details
kernel = np.ones((5, 15), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Detect circles/approximate circle using Hough Circle Transform
h, w = closed_edges.shape[:]
circles = cv2.HoughCircles(closed_edges, cv2.HOUGH_GRADIENT, 1.022,
                           int(w/9), param1=2, param2=30, minRadius=int(w/100),
                           maxRadius=int(w/9))

# Convert circle's x,y,r values to integer
circles = np.uint16(np.around(circles))

# Draw detected circles in original image
count = 1
for i in circles[0, :]:
    # Draw the outer circle
    cv2.circle(original_image, (i[0], i[1]), i[2], (0, 255, 0),
               2)
    # Draw a smaller circle at the center of the circle
    cv2.circle(original_image, (i[0], i[1]), 2,
               (0, 0, 255), 3)
    # Add object count number
    # cv2.putText(original_image, str(count), (i[0], i[1]),
    # cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
    # Count how many rounded objects detected
    count += 1


# Define function to get the radius 'r' of the circular objects detected
def get_radius(circles):
    radius = []
    for coord in circles[0, :]:
        radius.append(coord[2])
    return radius


# Define function to calculate average pixel (i.e. brightness)
def av_pix(img, circles, size):
    av_value = []
    for coord in circles[0, :]:
        col = np.mean(img[coord[1] - size:coord[1] + size,
                      coord[0] - size:coord[0] + size])
        av_value.append(col)
    return av_value


# Extract the radius & convert radius to integer (for classification purposes)
radii = get_radius(circles)
radii_int = [int(i) for i in radii]

# Extract brightness values of each circular object (for classification)
bright_values = av_pix(blurred_img, circles, 20)
bright_values_int = [int(i) for i in bright_values]

# Initialize an empty coin value list
values = []

# Determine coin value based on radius and brightness
for a, b in zip(radii_int, bright_values_int):
    if a > 137 and b == 88:
        values.append(50)
    elif a > 120 and b > 150:
        values.append(10)
    elif a < 100 and b > 150:
        values.append(5)
    elif 119 < a < 137 and 84 < b < 97:
        values.append(2)
    elif 99 < a < 110:
        values.append(1)

# Annotate coin value on each coin
count = 0
for i in circles[0, :]:
    cv2.putText(original_image, str(values[count]) + 'p', (i[0], i[1]),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255),
                2)
    count += 1

# Annotate the total value on the image
cv2.putText(original_image, 'Estimated Total Value: ' + str(sum(values))
            + 'p', (200, 100), cv2.FONT_HERSHEY_SIMPLEX,
            1.3, 255)

# Save the final annotated image
cv2.imwrite('Classified_coins_image.jpg', original_image)

# Display the final image with annotations
cv2.imshow("Detected Coins", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
