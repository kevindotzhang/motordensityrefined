import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN

class RefinedMotorDensityEstimator:
    def __init__(self, debug=False):
        self.debug = debug
        self.roi_mask = None
        
    def detect_container_roi(self, image):
        """Detect container/bin boundaries to create ROI mask"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Detect circular/rectangular containers
        # Use edge detection to find container boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Find large contours that could be container edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        roi_mask = np.ones(gray.shape, dtype=np.uint8) * 255
        
        # Look for large rectangular or circular containers
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0.1 * gray.shape[0] * gray.shape[1]:  # Large contour
                # Check if it's roughly rectangular or circular
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if area / hull_area > 0.7:  # Reasonably convex (container-like)
                    # Create mask inside this contour
                    temp_mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(temp_mask, [hull], 255)
                    roi_mask = cv2.bitwise_and(roi_mask, temp_mask)
        
        return roi_mask
    
    def create_edge_suppression_mask(self, image, border_percent=8):
        """Suppress outer edges where background is likely"""
        h, w = image.shape[:2]
        border_h = int(h * border_percent / 100)
        border_w = int(w * border_percent / 100)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[border_h:h-border_h, border_w:w-border_w] = 255
        
        return mask
    
    def background_color_masking(self, image):
        """Remove known background colors (cardboard, concrete, etc.)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define background color ranges
        background_ranges = [
            # Cardboard brown
            ([10, 50, 50], [25, 255, 200]),
            # Concrete gray  
            ([0, 0, 100], [180, 30, 180]),
            # Very light colors (overexposed areas)
            ([0, 0, 200], [180, 50, 255])
        ]
        
        background_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for (lower, upper) in background_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            background_mask = cv2.bitwise_or(background_mask, mask)
        
        # Invert to get foreground mask
        foreground_mask = cv2.bitwise_not(background_mask)
        
        return foreground_mask
    
    def create_combined_roi_mask(self, image):
        """Combine all masking techniques for final ROI"""
        # Container detection
        container_mask = self.detect_container_roi(image)
        
        # Edge suppression
        edge_mask = self.create_edge_suppression_mask(image, border_percent=5)
        
        # Background color filtering
        color_mask = self.background_color_masking(image)
        
        # Combine all masks (intersection)
        combined_mask = cv2.bitwise_and(container_mask, edge_mask)
        combined_mask = cv2.bitwise_and(combined_mask, color_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        self.roi_mask = combined_mask
        return combined_mask
    
    def apply_roi_mask(self, mask, roi_mask):
        """Apply ROI mask to segmentation result"""
        return cv2.bitwise_and(mask, roi_mask)
    
    def post_segmentation_cleanup(self, mask, min_area=200, max_area=None):
        """Remove small noise and very large false positives"""
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        cleaned_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by area
            if area < min_area:
                continue
            if max_area and area > max_area:
                continue
                
            # Filter by aspect ratio (motors shouldn't be extremely elongated)
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 5:  # Very elongated, likely noise
                continue
            
            # Add to cleaned mask
            component_mask = (labels == i).astype(np.uint8) * 255
            cleaned_mask = cv2.bitwise_or(cleaned_mask, component_mask)
        
        return cleaned_mask
    
    def hsv_saturation_method(self, image, roi_mask=None):
        """HSV Saturation method with optional ROI masking"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        _, mask = cv2.threshold(hsv[:,:,1], 30, 255, cv2.THRESH_BINARY)
        
        # Apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            mask = self.apply_roi_mask(mask, roi_mask)
        
        # Post-processing cleanup
        mask = self.post_segmentation_cleanup(mask)
        
        return mask
    
    def lab_a_channel_method(self, image, roi_mask=None):
        """LAB A-channel method with optional ROI masking"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        _, mask = cv2.threshold(lab[:,:,1], 127, 255, cv2.THRESH_BINARY)
        
        # Apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            mask = self.apply_roi_mask(mask, roi_mask)
        
        # Post-processing cleanup
        mask = self.post_segmentation_cleanup(mask)
        
        return mask
    
    def adaptive_threshold_method(self, image, roi_mask=None):
        """Adaptive threshold method with optional ROI masking"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 21, 8)
        
        # Apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            mask = self.apply_roi_mask(mask, roi_mask)
        
        # Post-processing cleanup
        mask = self.post_segmentation_cleanup(mask)
        
        return mask
    
    def ensemble_prediction(self, image, roi_mask=None, method="weighted_average"):
        """Combine multiple methods for final prediction"""
        # Get predictions from all methods
        hsv_mask = self.hsv_saturation_method(image, roi_mask)
        lab_mask = self.lab_a_channel_method(image, roi_mask)
        adaptive_mask = self.adaptive_threshold_method(image, roi_mask)
        
        if method == "intersection":
            # Conservative: only where all methods agree
            final_mask = cv2.bitwise_and(hsv_mask, lab_mask)
            final_mask = cv2.bitwise_and(final_mask, adaptive_mask)
            
        elif method == "union":
            # Liberal: where any method detects material
            final_mask = cv2.bitwise_or(hsv_mask, lab_mask)
            final_mask = cv2.bitwise_or(final_mask, adaptive_mask)
            
        elif method == "weighted_average":
            # Weighted combination based on past performance
            weights = [0.4, 0.4, 0.2]  # HSV, LAB, Adaptive
            
            # Convert to float for averaging
            combined = (hsv_mask.astype(np.float32) * weights[0] + 
                       lab_mask.astype(np.float32) * weights[1] + 
                       adaptive_mask.astype(np.float32) * weights[2])
            
            # Threshold at 50% agreement
            final_mask = (combined > 127).astype(np.uint8) * 255
            
        elif method == "majority_vote":
            # Where at least 2/3 methods agree
            combined = (hsv_mask.astype(np.float32) + 
                       lab_mask.astype(np.float32) + 
                       adaptive_mask.astype(np.float32)) / 255
            
            final_mask = (combined >= 2).astype(np.uint8) * 255
        
        return final_mask, {
            'hsv': hsv_mask,
            'lab': lab_mask, 
            'adaptive': adaptive_mask
        }
    
    def calculate_density(self, mask):
        """Calculate density metrics"""
        if self.roi_mask is not None:
            # Calculate density only within ROI
            roi_area = np.sum(self.roi_mask > 0)
            solid_area = np.sum(cv2.bitwise_and(mask, self.roi_mask) > 0)
            density = solid_area / roi_area if roi_area > 0 else 0
        else:
            total_area = mask.shape[0] * mask.shape[1]
            solid_area = np.sum(mask > 0)
            density = solid_area / total_area
        
        return density
    
    def process_image(self, image):
        """Complete processing pipeline"""
        # Step 1: Create ROI mask
        roi_mask = self.create_combined_roi_mask(image)
        
        # Step 2: Get ensemble prediction
        final_mask, individual_masks = self.ensemble_prediction(image, roi_mask, 
                                                               method="majority_vote")
        
        # Step 3: Calculate densities
        results = {
            'final_density': self.calculate_density(final_mask),
            'final_mask': final_mask,
            'roi_mask': roi_mask,
            'individual_masks': individual_masks,
            'individual_densities': {
                method: self.calculate_density(mask) 
                for method, mask in individual_masks.items()
            }
        }
        
        if self.debug:
            self.visualize_results(image, results)
        
        return results
    
    def visualize_results(self, image, results):
        """Visualize processing steps"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0,0].imshow(image)
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # ROI mask
        axes[0,1].imshow(results['roi_mask'], cmap='gray')
        axes[0,1].set_title('ROI Mask')
        axes[0,1].axis('off')
        
        # Final result
        axes[0,2].imshow(results['final_mask'], cmap='gray')
        axes[0,2].set_title(f'Final Result\nDensity: {results["final_density"]:.3f}')
        axes[0,2].axis('off')
        
        # Individual methods
        methods = ['hsv', 'lab', 'adaptive']
        for i, method in enumerate(methods):
            mask = results['individual_masks'][method]
            density = results['individual_densities'][method]
            axes[1,i].imshow(mask, cmap='gray')
            axes[1,i].set_title(f'{method.upper()}\nDensity: {density:.3f}')
            axes[1,i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Usage example
def main():
    estimator = RefinedMotorDensityEstimator(debug=True)
    
    # Process test images
    image_paths = ["image1.png", "image2.png", "image3.png", "image4.png"]
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            print(f"\nProcessing {image_path}...")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = estimator.process_image(image)
            
            print(f"Final Density: {results['final_density']:.3f}")
            print("Individual method densities:")
            for method, density in results['individual_densities'].items():
                print(f"  {method}: {density:.3f}")

if __name__ == "__main__":
    import os
    main()