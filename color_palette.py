import numpy as np
import cv2 # type:ignore
from PIL import Image  # type:ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.cluster import KMeans  # type:ignore
import random

# Load the pre-trained U2NET model
U2NET_MODEL = load_model('/home/sherif-hodhod/Sherif/D/CS/GP/ImageRecoloring.h5', compile=False)

def preprocess_image(image, size):
    """Preprocess the image for the model."""
    img = image.convert('RGB')
    img = img.resize((size, size))
    img_array = np.array(img)
    return np.expand_dims(img_array, axis=0)

def apply_mask(image, mask):
    """Apply a binary mask to the image."""
    mask = (mask > 0.5).astype(np.uint8) * 255
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def mcts_generate_colors(base_colors, n_new_colors, n_iterations=1000, variation_range=60):
    """Generate new colors using Monte Carlo Tree Search."""
    best_new_colors = []
    for _ in range(n_new_colors):
        best_color = None
        best_score = float('inf')
        for _ in range(n_iterations):
            base_color = random.choice(base_colors)
            new_color = base_color + np.random.randint(-variation_range, variation_range, size=3)
            new_color = np.clip(new_color, 0, 255)
            score = np.min(np.linalg.norm(base_colors - new_color, axis=1))
            if score < best_score:
                best_score = score
                best_color = new_color
        best_new_colors.append(best_color)
    return np.array(best_new_colors)

def rgb_to_hex(rgb):
    """Convert RGB to HEX color."""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def plot_extended_palette(image, n_colors=5, n_new_colors=3):
    """Generate and return an extended color palette from the image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.array(image)
    pixels = image_rgb.reshape(-1, 3)
    mask = (pixels.sum(axis=1) != 0)
    pixels = pixels[mask]

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    new_colors = mcts_generate_colors(colors, n_new_colors=n_new_colors)
    all_colors = np.vstack((colors, new_colors))

    return [rgb_to_hex(color) for color in all_colors]
