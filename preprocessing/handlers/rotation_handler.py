import cv2
import numpy as np
from math import degrees, atan2

class RotationHandler:
    @staticmethod
    def detect_text_orientations(gray_image, tolerance=5):
        """
        Enhanced detection of multiple text orientations in the image
        Returns a list of probable rotation angles
        """
        # Apply adaptive thresholding to get cleaner edges
        thresh = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Apply edge detection with optimized parameters
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        # Use probabilistic Hough transform with adjusted parameters for multiple angles
        lines = cv2.HoughLinesP(
            edges, 
            rho=1,
            theta=np.pi/180,
            threshold=30,  # Lower threshold to detect more lines
            minLineLength=30,  # Shorter minimum line length to catch smaller text
            maxLineGap=10
        )
        
        if lines is None:
            return [0]  # Return only horizontal if no lines detected
        
        # Calculate angles and weights
        angle_weights = {}  # Dictionary to store angle frequencies
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if x2 - x1 == 0:  # Vertical line
                angle = 90
            else:
                angle = degrees(atan2(y2 - y1, x2 - x1))
            
            # Normalize angle to -90 to 90 range
            angle = angle % 180
            if angle > 90:
                angle -= 180
            
            # Round angle to nearest degree for clustering
            rounded_angle = round(angle)
            
            # Weight longer lines more heavily
            weight = length
            
            # Add to angle_weights dictionary
            if rounded_angle in angle_weights:
                angle_weights[rounded_angle] += weight
            else:
                angle_weights[rounded_angle] = weight
        
        # Find major angles (peaks in the angle distribution)
        major_angles = []
        total_weight = sum(angle_weights.values())
        
        # Sort angles by weight
        sorted_angles = sorted(angle_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Keep angles that represent significant portions of the text
        weight_threshold = total_weight * 0.1  # 10% of total weight
        current_weight = 0
        
        for angle, weight in sorted_angles:
            # Skip angles too close to already detected ones
            if not any(abs(angle - existing) < tolerance for existing in major_angles):
                major_angles.append(angle)
                current_weight += weight
            
                # Stop if we've covered most of the significant angles
                if current_weight > total_weight * 0.8:  # 80% coverage
                    break
        
        # Always include 0 degrees if no angles are close to it
        if not any(abs(angle) < tolerance for angle in major_angles):
            major_angles.append(0)
        
        # Sort angles by absolute value (try smallest rotations first)
        major_angles.sort(key=abs)
        
        # Limit the number of angles to prevent excessive processing
        max_angles = 4  # Maximum number of angles to try
        if len(major_angles) > max_angles:
            # Keep 0 degree if present, and the most significant other angles
            zero_angle = 0 if 0 in major_angles else None
            filtered_angles = sorted(major_angles, key=lambda x: abs(x))[:max_angles]
            if zero_angle is not None and zero_angle not in filtered_angles:
                filtered_angles[0] = zero_angle
            major_angles = filtered_angles
        
        return major_angles

    @staticmethod
    def _cluster_angles(angles, tolerance=5):
        """
        Cluster similar angles together
        Returns list of representative angles for each cluster
        """
        if not angles:
            return [0]
            
        # Normalize angles to 0-180 range
        normalized_angles = [a % 180 for a in angles]
        clusters = []
        
        for angle in normalized_angles:
            added = False
            for cluster in clusters:
                if min(abs(angle - a) for a in cluster) < tolerance:
                    cluster.append(angle)
                    added = True
                    break
            if not added:
                clusters.append([angle])
        
        # Get mean angle for each significant cluster
        significant_angles = []
        min_cluster_size = len(angles) * 0.1  # 10% of total lines
        
        for cluster in clusters:
            if len(cluster) >= min_cluster_size:
                mean_angle = sum(cluster) / len(cluster)
                # Convert to closest multiple of 90 if within tolerance
                for base_angle in [0, 90, 180, 270]:
                    if abs(mean_angle - base_angle) < tolerance:
                        mean_angle = base_angle
                        break
                significant_angles.append(mean_angle)
        
        return significant_angles if significant_angles else [0]

    @staticmethod
    def rotate_image(image, angle):
        """
        Rotate image by given angle
        Returns rotated image and transformation matrix
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)
        
        # Adjust translation part of the matrix
        rotation_matrix[0, 2] += new_width / 2 - center[0]
        rotation_matrix[1, 2] += new_height / 2 - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, rotation_matrix

    @staticmethod
    def transform_coordinates(points, rotation_matrix, angle,
                            orig_width, orig_height,
                            rotated_width, rotated_height):
        """
        Transform coordinates from rotated image back to original image space
        """
        # Create inverse transformation matrix
        center = (rotated_width // 2, rotated_height // 2)
        inverse_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        # Adjust for translation
        inverse_matrix[0, 2] += orig_width / 2 - center[0]
        inverse_matrix[1, 2] += orig_height / 2 - center[1]
        
        # Transform each point
        transformed_points = []
        for point in points:
            x, y = point
            original_point = np.dot(inverse_matrix, [x, y, 1])
            transformed_points.append([int(original_point[0]), int(original_point[1])])
        
        return transformed_points

    @staticmethod
    def process_with_rotation(image, reader, angle):
        """
        Enhanced processing of text at specific orientation
        """
        # Rotate image
        rotated_image, rotation_matrix = RotationHandler.rotate_image(image, angle)
        
        # Convert to grayscale
        gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Add white border
        border_size = 20
        bordered = cv2.copyMakeBorder(
            binary,
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT,
            value=[255]
        )
        
        # Perform OCR with optimized parameters for rotated text
        results = reader.readtext(
            bordered,
            paragraph=False,  # Disable paragraph grouping for rotated text
            batch_size=4,
            contrast_ths=0.1,
            text_threshold=0.6,
            width_ths=0.7,
            height_ths=0.7,
            decoder='beamsearch',
            beamWidth=10,
            rotation_info=[0]
        )
        
        blocks = []
        for result in results:
            # Handle cases where confidence score is not returned
            if len(result) == 3:
                box, text, conf = result
            else:
                box, text = result
                conf = 0.8
            
            # Skip empty or very short text
            if len(text.strip()) < 2:
                continue
            
            # Adjust coordinates for border
            adjusted_box = [[p[0] - border_size, p[1] - border_size] for p in box]
            
            transformed_box = RotationHandler.transform_coordinates(
                adjusted_box, rotation_matrix, angle,
                image.shape[1], image.shape[0],
                rotated_image.shape[1], rotated_image.shape[0]
            )
            
            left = min(p[0] for p in transformed_box)
            right = max(p[0] for p in transformed_box)
            top = min(p[1] for p in transformed_box)
            bottom = max(p[1] for p in transformed_box)
            
            width = right - left
            height = bottom - top
            aspect_ratio = width / height if height > 0 else 0
            
            # Enhanced filtering conditions
            if (conf > 0.3 and 
                0.1 < aspect_ratio < 10 and
                width > 5 and height > 5 and
                width < image.shape[1] * 0.9 and  # Not too wide
                height < image.shape[0] * 0.9):   # Not too tall
                
                blocks.append({
                    'type': 'text',
                    'text': text.strip(),
                    'confidence': conf,
                    'bounds': {
                        'left': left,
                        'right': right,
                        'top': top,
                        'bottom': bottom
                    },
                    'rotation': angle
                })
        
        return blocks