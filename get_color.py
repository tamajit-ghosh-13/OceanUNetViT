import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("test_shallow.png")
# The image is 400x600 (since figsize=(4,6), dpi=100).
# Bottom of the plot is at y = 0.9 * 600 = 540.
# Let's just print the RGB value at the bottom center of the plot.
height, width, _ = img.shape
bottom_y = int(height * 0.85)
center_x = int(width * 0.4)
pixel = img[bottom_y, center_x]
print(f"Color at bottom (RGB): {pixel[:3]}")
