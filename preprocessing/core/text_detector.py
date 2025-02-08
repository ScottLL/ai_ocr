import cv2
import numpy as np
import easyocr
import concurrent.futures
from preprocessing.handlers.rotation_handler import RotationHandler
from difflib import SequenceMatcher

class TextDetector:
    def __init__(self, reader=None):
        """
        :param reader: an EasyOCR Reader object; if None, one is created
        """
        self.reader = reader if reader else easyocr.Reader(['en'], gpu=False)

    def _preprocess_text_image(self, gray_image):
        """
        Balanced text preprocessing that preserves region detection while enhancing text
        """
        # Get grayscale copy
        gray = gray_image.copy()
        
        # 1. Gentle Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Mild Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,  
            5
        )
        
        # 4. Minimal Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        processed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return processed

    def _detect_text_regions(self, gray_image):
        """
        Detect potential text regions using morphological operations
        Returns a binary mask of text regions
        """
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray_image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )

        # Create kernels for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Dilate to connect text components
        dilated = cv2.dilate(thresh, kernel, iterations=3)
        
        # Close gaps
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        return closed

    def _rotate_box(self, box, angle, center):
        """
        Rotate a box around a center point by given angle
        :param box: List of points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        :param angle: Angle in degrees
        :param center: Center point [x,y]
        :return: Rotated box points
        """
        angle_rad = np.radians(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        cx, cy = center
        
        rotated_box = []
        for point in box:
            x, y = point
            # Translate point to origin
            tx = x - cx
            ty = y - cy
            # Rotate point
            rx = tx * cos_val - ty * sin_val
            ry = tx * sin_val + ty * cos_val
            # Translate back
            rotated_box.append([int(rx + cx), int(ry + cy)])
            
        return rotated_box

    def _remove_duplicate_text(self, text_blocks, similarity_threshold=0.8):
        """
        Remove duplicate text blocks based on text content similarity
        and box overlap
        """
        def text_similarity(text1, text2):
            return SequenceMatcher(None, text1, text2).ratio()
        
        def boxes_overlap(box1, box2, overlap_threshold=0.5):
            """
            box1, box2: Each is a list of corner points [[x,y], [x,y], ...].
            We'll treat them as polygons for overlap check. 
            A simple approach can be boundingRect-based, or a more advanced polygon intersection.
            For simplicity, use boundingRect intersection ratio here.
            """
            # Convert polygon to bounding rectangles
            min_x1 = min(p[0] for p in box1); max_x1 = max(p[0] for p in box1)
            min_y1 = min(p[1] for p in box1); max_y1 = max(p[1] for p in box1)
            min_x2 = min(p[0] for p in box2); max_x2 = max(p[0] for p in box2)
            min_y2 = min(p[1] for p in box2); max_y2 = max(p[1] for p in box2)
            
            # Compute intersection rect
            inter_left = max(min_x1, min_x2)
            inter_top = max(min_y1, min_y2)
            inter_right = min(max_x1, max_x2)
            inter_bottom = min(max_y1, max_y2)
            
            if inter_right < inter_left or inter_bottom < inter_top:
                intersection_area = 0
            else:
                intersection_area = (inter_right - inter_left) * (inter_bottom - inter_top)
            
            box1_area = (max_x1 - min_x1) * (max_y1 - min_y1)
            box2_area = (max_x2 - min_x2) * (max_y2 - min_y2)
            
            # Compute overlap ratio with respect to smaller box area to be conservative
            smaller_area = min(box1_area, box2_area) if box1_area and box2_area else 1
            overlap_ratio = intersection_area / float(smaller_area) if smaller_area != 0 else 0
            
            return overlap_ratio > overlap_threshold

        unique_blocks = []
        used_indices = set()

        for i, block1 in enumerate(text_blocks):
            if i in used_indices:
                continue

            current_block = block1
            used_indices.add(i)

            # Compare with other blocks
            for j, block2 in enumerate(text_blocks):
                if j in used_indices or i == j:
                    continue

                # Check text similarity and box overlap
                text_sim = text_similarity(block1['text'], block2['text'])
                boxes_overlap_check = boxes_overlap(
                    block1['polygon'], 
                    block2['polygon']
                )

                if text_sim > similarity_threshold or boxes_overlap_check:
                    # Keep the block with higher confidence
                    if block2['confidence'] > current_block['confidence']:
                        current_block = block2
                    used_indices.add(j)

            unique_blocks.append(current_block)

        return unique_blocks

    def detect_text_blocks(self, image, gray_image):
        """
        Enhanced text block detection with improved rotation handling
        and duplicate removal
        """
        # Detect text regions first
        text_mask = self._detect_text_regions(gray_image)
        contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = text_mask.shape
        min_area = h * w * 0.0001  # Minimum area threshold
        text_regions = []
        
        # Filter contours to get valid text regions
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
                
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            # Add padding to ensure complete text capture
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_cnt = min(image.shape[1] - x, w_cnt + 2 * padding)
            h_cnt = min(image.shape[0] - y, h_cnt + 2 * padding)
            
            text_regions.append((x, y, w_cnt, h_cnt))
        
        # Process each region once
        text_blocks = []
        for x, y, w_cnt, h_cnt in text_regions:
            roi = image[y : y + h_cnt, x : x + w_cnt]
            
            # Perform OCR on the region
            results = self.reader.readtext(roi)
            
            for box, text, confidence in results:
                # Skip very short text
                if len(text.strip()) < 2:
                    continue
                
                # Skip text with very low confidence
                if confidence < 0.2:
                    continue
                
                # Convert local ROI coordinates to absolute image coordinates
                absolute_box = []
                min_x = float('inf')
                min_y = float('inf')
                max_x = float('-inf')
                max_y = float('-inf')
                for (bx, by) in box:
                    abs_x = x + bx
                    abs_y = y + by
                    absolute_box.append([abs_x, abs_y])
                    min_x = min(min_x, abs_x)
                    min_y = min(min_y, abs_y)
                    max_x = max(max_x, abs_x)
                    max_y = max(max_y, abs_y)
                
                # Approximate angle for reference (not necessarily used here)
                angle = 0
                
                text_blocks.append({
                    'type': 'text',
                    'text': text.strip(),
                    'confidence': confidence,
                    'angle': angle,
                    'polygon': absolute_box,
                    'bounds': {
                        'left': min_x,
                        'top': min_y,
                        'right': max_x,
                        'bottom': max_y
                    }
                })
        
        # After creating text_blocks, remove duplicates
        text_blocks = self._remove_duplicate_text(text_blocks)
        
        # Sort blocks by position (top to bottom, left to right)
        text_blocks.sort(key=lambda x: (x['bounds']['top'], x['bounds']['left']))
        
        return text_blocks 