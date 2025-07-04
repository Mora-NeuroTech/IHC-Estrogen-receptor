
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
    if image.shape[-1] == 4:
        image = image[..., :3]  # Remove alpha channel
    if image.dtype == 'uint8':
        image = image.astype('float32') / 255.0
    return image

def create_color_masks_rgb(image):
    lab = color.rgb2lab(image)
    L = lab[:, :, 0]
    
    # Apply thresholding
    thresh = threshold_otsu(L)*1.0
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
    label_brown = watershed(-distance, markers, mask=binary)
    
    # Remove small objects
    brown_mask = morphology.remove_small_objects(label_brown, min_size=50)
    
    # Create blue mask using RGB ranges
    rgb_image = (image * 255).astype('uint8')
    lower_blue = np.array([101, 112, 143])  # R Min, G Min, B Min
    upper_blue = np.array([158, 255, 255])  # R Max, G Max, B Max
    label_blue = cv2.inRange(rgb_image, lower_blue, upper_blue) > 0
    blue_mask = clean_mask(label_blue)
    blue_mask = remove_elongated_objects(blue_mask)  # Remove elongated objects
    return blue_mask > 0, brown_mask > 0
def calculate_mask_stats(image, blue_mask, brown_mask):
    """Calculate statistics for both masks"""
    # Calculate total image area
    total_pixels = image.shape[0] * image.shape[1]
    
    # Calculate area statistics
    blue_pixels = np.sum(blue_mask)
    brown_pixels = np.sum(brown_mask)
    
    blue_area_percent = (blue_pixels / total_pixels) * 100
    brown_area_percent = (brown_pixels / total_pixels) * 100
    
    # Count individual cells using connected components
    blue_labels = measure.label(blue_mask)
    brown_labels = measure.label(brown_mask)
    
    blue_cell_count = blue_labels.max()  # Same as len(np.unique(blue_labels)) - 1
    brown_cell_count = brown_labels.max()

    Total_area = blue_area_percent + brown_area_percent
    brown_cell_percent = (brown_area_percent / Total_area) * 100

    if (brown_cell_percent < 1):
        status = 1
    elif (brown_cell_percent >= 1 and brown_cell_percent < 10):
        status = 2
    elif (brown_cell_percent >= 10 and brown_cell_percent < 33):
        status = 3
    elif (brown_cell_percent >= 33 and brown_cell_percent < 66):
        status = 4
    else:
        status = 5 

    return {
        'blue_area_percent': blue_area_percent,
        'brown_area_percent': brown_area_percent,
        'blue_cell_count': blue_cell_count,
        'brown_cell_count': brown_cell_count,
        'blue_labels': blue_labels,  # For visualization
        'brown_labels': brown_labels,
        'status': status
    }

def clean_mask(mask):
    """Clean up the mask with morphological operations"""
    mask = morphology.remove_small_holes(mask, area_threshold=50)
    mask = morphology.remove_small_objects(mask, min_size=70)
    return mask

def remove_elongated_objects(mask, max_eccentricity=0.9):
    """Remove objects with high eccentricity (elongated shapes)"""
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    
    clean_mask = np.zeros_like(mask)
    for prop in props:
        if prop.eccentricity < max_eccentricity:  # Only keep roundish objects
            clean_mask[labeled == prop.label] = 1
            
    return clean_mask

def draw_contours(image, mask, color, thickness=2):
    """Draw contours directly on the image"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)
    return image

def calculate_grayscale_intensity(image, mask):
    """Calculate grayscale intensity values within a mask"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    
    # Extract intensity values where mask is True
    intensity_values = gray_image[mask]
    
    if len(intensity_values) == 0:
        return None
    median = np.median(intensity_values)
    if (median < 0.3):
        status = 3
    elif (median >= 0.3 and median < 0.6):
        status = 2
    else:
        status = 1
    return {
        'mean_intensity': np.mean(intensity_values),
        'median_intensity': median,
        'std_intensity': np.std(intensity_values),
        'min_intensity': np.min(intensity_values),
        'max_intensity': np.max(intensity_values),
        'status': status,
        'all_values': intensity_values
    }

def calculate_marks(stats, brown_gray):
    intensity_status = brown_gray['status']
    brown_status = stats['status']

    total_marks = brown_status + intensity_status
    if total_marks <= 2:
        outcome = "Negative"
    elif total_marks == 3:
        outcome = "Low Positive"
    elif total_marks >= 4:
        outcome = "Positive"
    
    return outcome

def plot_intensity_distribution(intensity_data, mask_name):
    """Plot histogram of grayscale intensity values"""
    if intensity_data is None:
        print(f"No cells detected in {mask_name} mask")
        return
    
    plt.figure(figsize=(8, 5))
    plt.hist(intensity_data['all_values'], bins=50, color='gray', alpha=0.7)
    plt.title(f'Grayscale Intensity Distribution - {mask_name} Mask')
    plt.xlabel('Intensity (0=black, 1=white)')
    plt.ylabel('Pixel Count')
    
    # Add statistics annotations
    stats_text = (
        f"Mean: {intensity_data['mean_intensity']:.3f}\n"
        f"Median: {intensity_data['median_intensity']:.3f}\n"
        f"Std Dev: {intensity_data['std_intensity']:.3f}"
    )
    plt.annotate(stats_text, xy=(0.7, 0.7), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def visualize_results(image, blue_mask, brown_mask):
    
    result = np.copy(image)
    
    # Draw contours (using RGB colors)
    result = draw_contours(result, blue_mask.astype(np.uint8), (255, 0, 0))    # Blue
    result = draw_contours(result, brown_mask.astype(np.uint8), (139, 69, 19)) # Brown
    
    plt.figure(figsize=(15, 10))
    plt.imshow(result)
    plt.title(f'Blue cells: {stats["blue_cell_count"]} | Brown cells: {stats["brown_cell_count"]}')
    plt.axis('off')
    plt.show()

def visualize_stats(image, stats):
    """Visualize the results with statistics"""
    plt.figure(figsize=(18, 6))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Blue cells with count
    plt.subplot(1, 3, 2)
    plt.imshow(stats['blue_labels'], cmap='nipy_spectral')
    plt.title(f'Blue Cells: {stats["blue_cell_count"]}\nArea: {stats["blue_area_percent"]:.2f}%')
    plt.axis('off')
    
    # Brown cells with count
    plt.subplot(1, 3, 3)
    plt.imshow(stats['brown_labels'], cmap='nipy_spectral')
    plt.title(f'Brown Cells: {stats["brown_cell_count"]}\nArea: {stats["brown_area_percent"]:.2f}%')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
# Main execution
try:
    image = load_image('slide (1).png')
    blue_mask, brown_mask = create_color_masks_rgb(image)
    
    # Calculate statistics
    stats = calculate_mask_stats(image, blue_mask, brown_mask)

    # Calculate grayscale intensity values
    blue_gray = calculate_grayscale_intensity(image, blue_mask)
    brown_gray = calculate_grayscale_intensity(image, brown_mask)


    


    # Print intensity statistics
    print("\nGrayscale Intensity Analysis:")
    print("Blue Cells:")
    if blue_gray:
        print(f"  Mean: {blue_gray['mean_intensity']:.4f}")
        print(f"  Median: {blue_gray['median_intensity']:.4f}")
        print(f"  Std Dev: {blue_gray['std_intensity']:.4f}")
        print(f"  Range: {blue_gray['min_intensity']:.4f}-{blue_gray['max_intensity']:.4f}")
    else:
        print("  No blue cells detected")

    print("\nBrown Cells:")
    if brown_gray:
        print(f"  Mean: {brown_gray['mean_intensity']:.4f}")
        print(f"  Median: {brown_gray['median_intensity']:.4f}")
        print(f"  Std Dev: {brown_gray['std_intensity']:.4f}")
        print(f"  Range: {brown_gray['min_intensity']:.4f}-{brown_gray['max_intensity']:.4f}")
    else:
        print("  No brown cells detected")

    # Plot intensity distributions
    plot_intensity_distribution(blue_gray, "Blue")
    plot_intensity_distribution(brown_gray, "Brown")
    
    # Print results
    print(f"Blue Cells: {stats['blue_cell_count']} cells, {stats['blue_area_percent']:.2f}% of area")
    print(f"Brown Cells: {stats['brown_cell_count']} cells, {stats['brown_area_percent']:.2f}% of area")
    print(f"Status: {stats['status']}")
    # Visualize with circled cells
    visualize_stats(image, stats)
    visualize_results(image, blue_mask, brown_mask)

    marks = calculate_marks(stats,brown_gray)
    print(marks)
    
except Exception as e:
    print(f"Error processing image: {e}")
    print(f"Image shape: {image.shape if 'image' in locals() else 'Not loaded'}")
