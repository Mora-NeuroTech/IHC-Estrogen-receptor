import cv2
import numpy as np
from skimage import io, color, morphology, measure
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load image and ensure it's 3-channel RGB"""
    image = io.imread(image_path)
    
    # Remove alpha channel if present
    if image.shape[-1] == 4:
        image = image[..., :3]  # Keep only RGB channels
        
    # Convert to float and normalize if needed
    if image.dtype == 'uint8':
        image = image.astype('float32') / 255.0
        
    return image

def create_mask(image):
    lab = color.rgb2lab(image)
    L = lab[:, :, 0]
    
    # Apply thresholding
    thresh = threshold_otsu(L)*0.9  # Adjust threshold to reduce noise
    binary = L < thresh
    
    # Distance transform
    distance = ndi.distance_transform_edt(binary)
    
    # Find peaks for markers
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary, 
                            min_distance=4, exclude_border=True)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    
    # Watershed segmentation
    label = watershed(-distance, markers, mask=binary)
    
    # Remove small objects
    mask = morphology.remove_small_objects(label, min_size=50)
    
    return mask > 0

def clean_mask(mask):
    """Clean up the mask with morphological operations"""
    # Remove small objects and fill small holes
    mask = morphology.remove_small_holes(mask, area_threshold=50)
    mask = morphology.remove_small_objects(mask, min_size=50)
    return mask

def calculate_stats(image, mask):
    """Calculate statistics for mask"""
    # Calculate total image area
    total_pixels = image.shape[0] * image.shape[1]
    
    # Calculate area statistics
    pixels = np.sum(mask)
    area_percent = (pixels / total_pixels) * 100
    
    # Count individual cells using connected components
    labels = measure.label(mask)
    cell_count = labels.max()
    
    return {
        'area_percent': area_percent,
        'cell_count': cell_count,
        'labels': labels  # For visualization
    }

def visualize_contours(image, stats):
    """Visualize the results with cell contours overlaid on original image"""
    plt.figure(figsize=(12, 6))
    
    # Convert image to uint8 for OpenCV processing
    display_image = (image * 255).astype('uint8')
    
    # Find contours from the mask
    contours = measure.find_contours(stats['labels'], 0.5)
    
    # Display original image
    plt.imshow(image)
    
    # Plot all contours found
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
    
    plt.title(f'Cells: {stats["cell_count"]} cells\nArea: {stats["area_percent"]:.2f}%')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage with error handling
try:
    image = load_image('slide (2).jpg')
    mask = create_mask(image)
    
    # Calculate statistics
    stats = calculate_stats(image, mask)
    
    # Print results
    print(f"Cells: {stats['cell_count']} cells, {stats['area_percent']:.2f}% of area")
    
    # Visualize with contours
    visualize_contours(image, stats)
except Exception as e:
    print(f"Error processing image: {e}")
    print(f"Image shape: {image.shape if 'image' in locals() else 'Not loaded'}")