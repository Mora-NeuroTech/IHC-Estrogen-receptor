import cv2
import numpy as np
from skimage import io, color, morphology, measure
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

class CellAnalyzer:
    """Advanced cell detection and analysis system"""
    
    # Configuration constants
    BLUE_RANGE = {
        'lower': np.array([101, 112, 143]),
        'upper': np.array([158, 255, 255])
    }
    
    INTENSITY_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.6
    }
    
    AREA_THRESHOLDS = {
        'status_1': 1,
        'status_2': 10,
        'status_3': 33,
        'status_4': 66
    }
    
    def __init__(self, threshold_multiplier: float = 0.95, min_distance: int = 3, 
                 min_object_size: int = 50, min_hole_size: int = 50):
        """Initialize analyzer with configurable parameters"""
        self.threshold_multiplier = threshold_multiplier
        self.min_distance = min_distance
        self.min_object_size = min_object_size
        self.min_hole_size = min_hole_size
        self.stats = None
        self.intensity_data = None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and normalize image to RGB float32"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = io.imread(str(image_path))
        
        # Handle RGBA
        if len(image.shape) == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        
        # Normalize to [0, 1] float32
        if image.dtype == 'uint8':
            image = image.astype('float32') / 255.0
        
        return image
    
    def create_color_masks_rgb(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create brown (via watershed) and blue (via color range) masks"""
        # Brown mask using LAB and watershed
        lab = color.rgb2lab(image)
        L = lab[:, :, 0]
        
        thresh = threshold_otsu(L) * self.threshold_multiplier
        binary = L < thresh
        
        distance = ndi.distance_transform_edt(binary)
        
        coords = peak_local_max(
            distance, 
            footprint=np.ones((3, 3)), 
            labels=binary,
            min_distance=self.min_distance, 
            exclude_border=True
        )
        
        marker_mask = np.zeros(distance.shape, dtype=bool)
        marker_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(marker_mask)
        
        label_brown = watershed(-distance, markers, mask=binary)
        brown_mask = morphology.remove_small_objects(label_brown, min_size=self.min_object_size)
        
        # Blue mask using RGB color range
        rgb_image = (image * 255).astype('uint8')
        label_blue = cv2.inRange(rgb_image, self.BLUE_RANGE['lower'], self.BLUE_RANGE['upper']) > 0
        blue_mask = self._clean_mask(label_blue)
        blue_mask = self._remove_elongated_objects(blue_mask)
        
        return blue_mask > 0, brown_mask > 0
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Remove noise from binary mask"""
        mask = morphology.remove_small_holes(mask, area_threshold=self.min_hole_size)
        mask = morphology.remove_small_objects(mask, min_size=self.min_object_size)
        return mask
    
    def _remove_elongated_objects(self, mask: np.ndarray, max_eccentricity: float = 0.9) -> np.ndarray:
        """Filter out elongated (non-round) objects"""
        labeled = measure.label(mask)
        props = measure.regionprops(labeled)
        
        clean_mask = np.zeros_like(mask, dtype=bool)
        for prop in props:
            if prop.eccentricity < max_eccentricity:
                clean_mask[labeled == prop.label] = True
        
        return clean_mask
    
    def calculate_mask_stats(self, image: np.ndarray, blue_mask: np.ndarray, 
                            brown_mask: np.ndarray) -> Dict:
        """Compute comprehensive statistics for both masks"""
        total_pixels = image.shape[0] * image.shape[1]
        
        blue_pixels = np.sum(blue_mask)
        brown_pixels = np.sum(brown_mask)
        
        blue_area_percent = (blue_pixels / total_pixels) * 100
        brown_area_percent = (brown_pixels / total_pixels) * 100
        
        blue_labels = measure.label(blue_mask)
        brown_labels = measure.label(brown_mask)
        
        blue_cell_count = blue_labels.max()
        brown_cell_count = brown_labels.max()
        
        total_area = blue_area_percent + brown_area_percent
        brown_cell_percent = (brown_area_percent / total_area * 100) if total_area > 0 else 0
        
        # Determine brown status based on percentage
        if brown_cell_percent < self.AREA_THRESHOLDS['status_1']:
            status = 1
        elif brown_cell_percent < self.AREA_THRESHOLDS['status_2']:
            status = 2
        elif brown_cell_percent < self.AREA_THRESHOLDS['status_3']:
            status = 3
        elif brown_cell_percent < self.AREA_THRESHOLDS['status_4']:
            status = 4
        else:
            status = 5
        
        self.stats = {
            'blue_area_percent': blue_area_percent,
            'brown_area_percent': brown_area_percent,
            'blue_cell_count': blue_cell_count,
            'brown_cell_count': brown_cell_count,
            'blue_labels': blue_labels,
            'brown_labels': brown_labels,
            'status': status,
            'brown_cell_percent': brown_cell_percent,
            'total_area_percent': total_area
        }
        
        return self.stats
    
    def calculate_grayscale_intensity(self, image: np.ndarray, mask: np.ndarray) -> Optional[Dict]:
        """Analyze grayscale intensity within mask"""
        if len(image.shape) == 3:
            gray_image = color.rgb2gray(image)
        else:
            gray_image = image
        
        intensity_values = gray_image[mask]
        
        if len(intensity_values) == 0:
            return None
        
        median = np.median(intensity_values)
        
        # Determine intensity status
        if median < self.INTENSITY_THRESHOLDS['low']:
            intensity_status = 3
        elif median < self.INTENSITY_THRESHOLDS['medium']:
            intensity_status = 2
        else:
            intensity_status = 1
        
        return {
            'mean_intensity': np.mean(intensity_values),
            'median_intensity': median,
            'std_intensity': np.std(intensity_values),
            'min_intensity': np.min(intensity_values),
            'max_intensity': np.max(intensity_values),
            'status': intensity_status,
            'all_values': intensity_values
        }
    
    def calculate_score(self, brown_intensity: Optional[Dict]) -> int:
        """Calculate Allred score from brown status and intensity"""
        if self.stats is None or brown_intensity is None:
            raise ValueError("Stats and intensity data must be calculated first")
        
        return self.stats['status'] + brown_intensity['status']
    
    def get_outcome(self, score: int) -> str:
        """Convert score to clinical outcome"""
        if score <= 2:
            return "Negative"
        elif score == 3:
            return "Low Positive"
        else:
            return "Positive"
    
    def draw_contours(self, image: np.ndarray, mask: np.ndarray, color: Tuple, 
                     thickness: int = 2) -> np.ndarray:
        """Draw contours on image"""
        result = np.copy(image)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, thickness)
        return result
    
    def visualize_results(self, image: np.ndarray, blue_mask: np.ndarray, 
                         brown_mask: np.ndarray, save_path: Optional[str] = None) -> None:
        """Visualize segmentation results"""
        result = image.copy()
        result = self.draw_contours(result, blue_mask.astype(np.uint8), (0, 0, 1), 2)  # Red
        result = self.draw_contours(result, brown_mask.astype(np.uint8), (0.55, 0.27, 0.07), 2)  # Brown
        
        plt.figure(figsize=(15, 10))
        plt.imshow(result)
        plt.title(f'Blue cells: {self.stats["blue_cell_count"]} | Brown cells: {self.stats["brown_cell_count"]}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_intensity_distribution(self, intensity_data: Dict, mask_name: str,
                                        save_path: Optional[str] = None) -> None:
        """Plot intensity histogram"""
        if intensity_data is None:
            print(f"No cells detected in {mask_name} mask")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(intensity_data['all_values'], bins=50, color='gray', alpha=0.7, edgecolor='black')
        plt.title(f'Grayscale Intensity Distribution - {mask_name} Mask')
        plt.xlabel('Intensity (0=black, 1=white)')
        plt.ylabel('Pixel Count')
        
        stats_text = (
            f"Mean: {intensity_data['mean_intensity']:.4f}\n"
            f"Median: {intensity_data['median_intensity']:.4f}\n"
            f"Std Dev: {intensity_data['std_intensity']:.4f}\n"
            f"Range: {intensity_data['min_intensity']:.4f}-{intensity_data['max_intensity']:.4f}"
        )
        plt.annotate(stats_text, xy=(0.65, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    verticalalignment='top', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_summary(self, brown_intensity: Dict) -> None:
        """Print comprehensive analysis summary"""
        print("\n" + "="*60)
        print("CELL ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nBlue Cells: {self.stats['blue_cell_count']} cells, {self.stats['blue_area_percent']:.2f}% of area")
        print(f"Brown Cells: {self.stats['brown_cell_count']} cells, {self.stats['brown_area_percent']:.2f}% of area")
        print(f"Brown Percentage: {self.stats['brown_cell_percent']:.2f}% (Status: {self.stats['status']})")
        
        print(f"\nBrown Cell Intensity:")
        print(f"  Mean: {brown_intensity['mean_intensity']:.4f}")
        print(f"  Median: {brown_intensity['median_intensity']:.4f}")
        print(f"  Std Dev: {brown_intensity['std_intensity']:.4f}")
        print(f"  Range: {brown_intensity['min_intensity']:.4f}-{brown_intensity['max_intensity']:.4f}")
        print(f"  Status: {brown_intensity['status']}")
        
        score = self.calculate_score(brown_intensity)
        outcome = self.get_outcome(score)
        print(f"\nAllred Score: {score}")
        print(f"Outcome: {outcome}")
        print("="*60 + "\n")


def main(image_path: str = 'images/slide (1).png', output_dir: Optional[str] = None) -> None:
    """Main execution function"""
    try:
        analyzer = CellAnalyzer()
        
        # Load and process image
        image = analyzer.load_image(image_path)
        blue_mask, brown_mask = analyzer.create_color_masks_rgb(image)
        
        # Calculate statistics
        stats = analyzer.calculate_mask_stats(image, blue_mask, brown_mask)
        brown_intensity = analyzer.calculate_grayscale_intensity(image, brown_mask)
        
        # Get results
        score = analyzer.calculate_score(brown_intensity)
        outcome = analyzer.get_outcome(score)
        
        # Print summary
        analyzer.print_summary(brown_intensity)
        
        # Visualizations
        analyzer.visualize_results(image, blue_mask, brown_mask, 
                                  f"{output_dir}/segmentation.png" if output_dir else None)
        analyzer.visualize_intensity_distribution(brown_intensity, "Brown",
                                                 f"{output_dir}/intensity.png" if output_dir else None)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()