import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        # Initialize parameters for object detection
        self.min_area = 5000  # Minimum area for an object
        self.min_edge_density = 0.1  # Minimum edge density for image content
        self.max_text_density = 0.3  # Maximum allowed text density
        self.cluster_distance = 60  # Maximum distance between objects to be clustered
        
    def detect_objects(self, image, text_mask):
        """
        Detect objects in the image while excluding text regions
        """
        if image is None:
            raise ValueError("Input image is None")
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        height, width = gray.shape
        objects = []
        
        # Apply preprocessing to enhance object detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Multi-scale object detection
        scales = [1.0, 0.75, 0.5]
        for scale in scales:
            # Resize image for current scale
            if scale != 1.0:
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                scaled_img = cv2.resize(blurred, (scaled_width, scaled_height))
                scaled_text_mask = cv2.resize(text_mask, (scaled_width, scaled_height))
            else:
                scaled_img = blurred
                scaled_text_mask = text_mask
                
            # Detect objects at current scale
            objects_at_scale = self._detect_at_scale(scaled_img, scaled_text_mask, scale)
            objects.extend(objects_at_scale)
            
        # Remove overlapping detections
        objects = self._remove_overlaps(objects)
        
        # Cluster nearby objects
        objects = self._cluster_objects(objects)
        return objects
    
    def _detect_at_scale(self, gray_img, text_mask, scale):
        """
        Detect objects at a specific scale
        """
        height, width = gray_img.shape
        min_area_at_scale = self.min_area * (scale ** 2)
        
        # Edge detection
        edges = cv2.Canny(gray_img, 50, 150)
        
        # Find contours in edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too small
            if w * h < min_area_at_scale:
                continue
                
            # Skip if too narrow or too wide
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
                
            # Check region characteristics
            roi = gray_img[y:y+h, x:x+w]
            roi_text = text_mask[y:y+h, x:x+w]
            
            # Calculate densities
            edge_density = self._calculate_edge_density(roi)
            text_density = cv2.countNonZero(roi_text) / (w * h)
            
            # Check if region is likely an object
            if edge_density > self.min_edge_density and text_density < self.max_text_density:
                # Scale coordinates back to original size
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                orig_w = int(w / scale)
                orig_h = int(h / scale)
                
                objects.append({
                    'type': 'image',
                    'bounds': {
                        'left': orig_x,
                        'top': orig_y,
                        'right': orig_x + orig_w,
                        'bottom': orig_y + orig_h
                    },
                    'confidence': float(edge_density)
                })
                
        return objects
    
    def _calculate_edge_density(self, roi):
        """
        Calculate edge density for a region
        """
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and threshold
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        edge_pixels = np.sum(gradient_magnitude > 50)
        
        return edge_pixels / roi.size
    
    def _remove_overlaps(self, objects, overlap_threshold=0.3):
        """
        Remove overlapping detections, keeping the one with higher confidence
        """
        if not objects:
            return []
            
        # Sort by confidence
        objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)
        
        kept_objects = []
        for obj in objects:
            should_keep = True
            
            for kept_obj in kept_objects:
                overlap_ratio = self._calculate_overlap(obj, kept_obj)
                if overlap_ratio > overlap_threshold:
                    should_keep = False
                    break
                    
            if should_keep:
                kept_objects.append(obj)
                
        return kept_objects
    
    def _calculate_overlap(self, obj1, obj2):
        """
        Calculate overlap ratio between two objects
        """
        # Get coordinates
        x1 = max(obj1['bounds']['left'], obj2['bounds']['left'])
        y1 = max(obj1['bounds']['top'], obj2['bounds']['top'])
        x2 = min(obj1['bounds']['right'], obj2['bounds']['right'])
        y2 = min(obj1['bounds']['bottom'], obj2['bounds']['bottom'])
        
        # Calculate areas
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        intersection = w * h
        
        area1 = (obj1['bounds']['right'] - obj1['bounds']['left']) * \
                (obj1['bounds']['bottom'] - obj1['bounds']['top'])
        area2 = (obj2['bounds']['right'] - obj2['bounds']['left']) * \
                (obj2['bounds']['bottom'] - obj2['bounds']['top'])
                
        # Calculate overlap ratio
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0 
    
    def _cluster_objects(self, objects, distance_threshold=None):
        """
        Cluster nearby objects and merge them into larger bounding boxes
        """
        if not objects:
            return []
            
        # First, merge boxes that are contained within other boxes
        objects = self._merge_contained_boxes(objects)
        
        if distance_threshold is None:
            distance_threshold = self.cluster_distance
            
        clusters = []
        processed = set()
        
        for i, obj in enumerate(objects):
            if i in processed:
                continue
                
            cluster = [obj]
            processed.add(i)
            
            # Find all objects close to the current cluster
            while True:
                cluster_expanded = False
                
                for j, other_obj in enumerate(objects):
                    if j in processed:
                        continue
                        
                    # Check if other_obj is close to any object in the cluster
                    for cluster_obj in cluster:
                        if self._calculate_distance(cluster_obj, other_obj) <= distance_threshold:
                            cluster.append(other_obj)
                            processed.add(j)
                            cluster_expanded = True
                            break
                            
                if not cluster_expanded:
                    break
            
            # Merge objects in the cluster
            merged_obj = self._merge_cluster(cluster)
            clusters.append(merged_obj)
            
        return clusters
    
    def _calculate_distance(self, obj1, obj2):
        """
        Calculate the minimum distance between two bounding boxes
        """
        x1_min = obj1['bounds']['left']
        y1_min = obj1['bounds']['top']
        x1_max = obj1['bounds']['right']
        y1_max = obj1['bounds']['bottom']
        
        x2_min = obj2['bounds']['left']
        y2_min = obj2['bounds']['top']
        x2_max = obj2['bounds']['right']
        y2_max = obj2['bounds']['bottom']
        
        # Calculate horizontal distance
        if x1_max < x2_min:
            dx = x2_min - x1_max
        elif x2_max < x1_min:
            dx = x1_min - x2_max
        else:
            dx = 0
            
        # Calculate vertical distance
        if y1_max < y2_min:
            dy = y2_min - y1_max
        elif y2_max < y1_min:
            dy = y1_min - y2_max
        else:
            dy = 0
            
        return np.sqrt(dx**2 + dy**2)
    
    def _merge_cluster(self, cluster):
        """
        Merge a cluster of objects into a single bounding box
        """
        if not cluster:
            return None
            
        # Find the minimum and maximum coordinates
        left = min(obj['bounds']['left'] for obj in cluster)
        top = min(obj['bounds']['top'] for obj in cluster)
        right = max(obj['bounds']['right'] for obj in cluster)
        bottom = max(obj['bounds']['bottom'] for obj in cluster)
        
        # Calculate average confidence
        avg_confidence = sum(obj['confidence'] for obj in cluster) / len(cluster)
        
        return {
            'type': 'image',
            'bounds': {
                'left': left,
                'top': top,
                'right': right,
                'bottom': bottom
            },
            'confidence': avg_confidence
        }

    def _merge_contained_boxes(self, objects):
        """
        Merge boxes that are contained within other boxes
        """
        if not objects:
            return []

        # Sort objects by area (largest first)
        objects = sorted(objects, key=lambda x: (
            (x['bounds']['right'] - x['bounds']['left']) * 
            (x['bounds']['bottom'] - x['bounds']['top'])
        ), reverse=True)

        merged_objects = []
        processed = set()

        for i, larger_obj in enumerate(objects):
            if i in processed:
                continue

            contained_boxes = [larger_obj]
            
            # Find all smaller boxes contained within this box
            for j, smaller_obj in enumerate(objects):
                if j == i or j in processed:
                    continue
                    
                if self._is_contained_within(smaller_obj, larger_obj):
                    contained_boxes.append(smaller_obj)
                    processed.add(j)
            
            processed.add(i)
            merged_obj = self._merge_cluster(contained_boxes)
            merged_objects.append(merged_obj)

        return merged_objects

    def _is_contained_within(self, box1, box2):
        """
        Check if box1 is contained within box2
        """
        return (
            box1['bounds']['left'] >= box2['bounds']['left'] and
            box1['bounds']['right'] <= box2['bounds']['right'] and
            box1['bounds']['top'] >= box2['bounds']['top'] and
            box1['bounds']['bottom'] <= box2['bounds']['bottom']
        ) 