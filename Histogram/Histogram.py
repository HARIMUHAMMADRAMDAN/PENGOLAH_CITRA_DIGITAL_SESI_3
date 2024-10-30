import numpy as np
import imageio
import matplotlib.pyplot as plt

def calculate_histogram(image_path):
    
    image = imageio.imread(image_path)

    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    
    histogram, bin_edges = np.histogram(grayscale_image, bins=256, range=(0, 255))

    
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], histogram, width=1, color='black')
    plt.title('Histogram of Grayscale Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.xlim(0, 255)
    plt.grid()
    plt.show()

    
    total_pixels = np.sum(histogram)
    print(f'Total pixels: {total_pixels}')

    
    dominant_intensity = np.argmax(histogram)
    dominant_value = histogram[dominant_intensity]

    print(f'Dominant Intensity: {dominant_intensity}, Count: {dominant_value}')


image_path = 'RGB.jpg'
calculate_histogram(image_path)
