import cv2
import numpy as np

class BlockDetector:
    @staticmethod
    def detect_images_morphological(gray_image, text_mask):
        """
        Enhanced image region detection with adjusted thresholds
        """
        # Create a copy of grayscale image
        gray = gray_image.copy()
        h, w = gray.shape
        
        # Adjust thresholds
        min_area = h * w * 0.001  # Reduced minimum area threshold
        max_text_density = 0.8    # Increased maximum text density
        min_edge_density = 0.05   # Reduced minimum edge density
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Morphological operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # Find contours
        contours, _ = cv2.findContours(
            opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        image_blocks = []
        valid_regions = []
        
        print(f"Found {len(contours)} potential regions")  # Debug print
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Calculate edge density
            roi = gray[y:y+ch, x:x+cw]
            edges = cv2.Canny(roi, 50, 150)  # Adjusted Canny thresholds
            edge_density = cv2.countNonZero(edges) / (cw * ch)
            
            # Calculate text density
            roi_text_mask = text_mask[y:y+ch, x:x+cw]
            text_density = cv2.countNonZero(roi_text_mask) / (cw * ch)
            
            print(f"Region at ({x},{y}): edge_density={edge_density:.3f}, text_density={text_density:.3f}")  # Debug print
            
            # Modified image detection criteria
            is_image = (edge_density > min_edge_density and 
                       text_density < max_text_density and
                       cw > 50 and ch > 50)  # Minimum size requirements
            
            if is_image:
                # Check for containment
                is_contained = False
                for ex, ey, ew, eh in valid_regions:
                    if (x >= ex and y >= ey and 
                        x + cw <= ex + ew and y + ch <= ey + eh):
                        is_contained = True
                        break
                
                if not is_contained:
                    valid_regions.append((x, y, cw, ch))
                    image_blocks.append({
                        'type': 'image',
                        'bounds': {
                            'left': x,
                            'top': y,
                            'right': x + cw,
                            'bottom': y + ch
                        },
                        'confidence': float(edge_density)
                    })
                    print(f"Added image block at ({x},{y})")  # Debug print
        
        print(f"Detected {len(image_blocks)} image blocks")  # Debug print
        return image_blocks

    @staticmethod
    def blocks_are_close_enough(b1, b2, max_x_gap=20, max_y_gap=20,
                              min_x_overlap_ratio=0.3, min_y_overlap_ratio=0.0):
        """Enhanced proximity check for blocks"""
        l1, r1, t1, btm1 = b1['bounds']['left'], b1['bounds']['right'], b1['bounds']['top'], b1['bounds']['bottom']
        l2, r2, t2, btm2 = b2['bounds']['left'], b2['bounds']['right'], b2['bounds']['top'], b2['bounds']['bottom']
        w1, w2 = (r1-l1), (r2-l2)
        h1, h2 = (btm1 - t1), (btm2 - t2)

        # Calculate vertical centers
        center_y1 = (t1 + btm1) / 2
        center_y2 = (t2 + btm2) / 2
        vertical_center_diff = abs(center_y1 - center_y2)
        
        # More lenient horizontal gap for text blocks on same line
        if b1.get('type', '') == 'text' and b2.get('type', '') == 'text':
            if vertical_center_diff < min(h1, h2) * 0.5:
                max_x_gap = max(max_x_gap * 2, min(w1, w2) * 0.5)

        # Calculate overlaps
        overlap_left = max(l1, l2)
        overlap_right = min(r1, r2)
        overlap_width = max(0, overlap_right - overlap_left)
        
        overlap_top = max(t1, t2)
        overlap_bottom = min(btm1, btm2)
        overlap_height = max(0, overlap_bottom - overlap_top)

        min_w = min(w1, w2) if min(w1,w2)>0 else 1
        x_overlap_ratio = overlap_width / float(min_w)

        min_h = min(h1, h2) if min(h1,h2)>0 else 1
        y_overlap_ratio = overlap_height / float(min_h)

        # Calculate gaps
        horiz_gap = 0
        if r1 < l2:
            horiz_gap = l2 - r1
        elif r2 < l1:
            horiz_gap = l1 - r2

        vert_gap = 0
        if btm1 < t2:
            vert_gap = t2 - btm1
        elif btm2 < t1:
            vert_gap = t1 - btm2

        # Check proximity conditions
        close_by_gap = (horiz_gap <= max_x_gap) or (vert_gap <= max_y_gap)
        close_by_overlap = (x_overlap_ratio >= min_x_overlap_ratio) or \
                          (y_overlap_ratio >= min_y_overlap_ratio)

        # Special case for text blocks on same line
        if b1.get('type', '') == 'text' and b2.get('type', '') == 'text':
            if vertical_center_diff < min(h1, h2) * 0.5:
                return horiz_gap <= max_x_gap

        return (close_by_gap or close_by_overlap)
