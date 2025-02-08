import cv2
import numpy as np
from typing import List, Dict

class RegionGrouper:
    def __init__(self, vertical_gap_threshold: int = 50):
        """
        Initialize RegionGrouper with configurable parameters
        
        :param vertical_gap_threshold: Maximum vertical gap between elements to be considered in same region
        """
        self.vertical_gap_threshold = vertical_gap_threshold
        self.REGION_COLOR = (0, 165, 255)  # Orange in BGR format
        
    def group_elements_into_regions(self, elements: List[Dict], columns: List[Dict] = None) -> List[Dict]:
        """
        Group text and image elements into regions based on vertical proximity
        and associate columns with each region based on position
        
        :param elements: List of text and image elements with bounds information
        :param columns: List of text columns with position information
        :return: List of regions containing grouped elements and their associated columns
        """
        if not elements:
            return []
            
        # Sort elements by vertical position (top coordinate)
        sorted_elements = sorted(elements, key=lambda x: x['bounds']['top'])
        
        regions = []
        current_region = {
            'elements': [sorted_elements[0]],
            'bounds': dict(sorted_elements[0]['bounds']),
            'columns': []  # Will store columns that belong to this region
        }
        
        for element in sorted_elements[1:]:
            prev_bottom = current_region['bounds']['bottom']
            current_top = element['bounds']['top']
            
            # Check if element should be part of current region
            if current_top - prev_bottom <= self.vertical_gap_threshold:
                # Update region bounds
                current_region['bounds']['right'] = max(
                    current_region['bounds']['right'],
                    element['bounds']['right']
                )
                current_region['bounds']['left'] = min(
                    current_region['bounds']['left'],
                    element['bounds']['left']
                )
                current_region['bounds']['bottom'] = max(
                    current_region['bounds']['bottom'],
                    element['bounds']['bottom']
                )
                current_region['elements'].append(element)
            else:
                # Before starting new region, assign columns to current region
                if columns:
                    current_region['columns'] = self._assign_columns_to_region(
                        current_region['bounds'], 
                        columns
                    )
                # Start new region
                regions.append(current_region)
                current_region = {
                    'elements': [element],
                    'bounds': dict(element['bounds']),
                    'columns': []
                }
        
        # Add last region and its columns
        if columns:
            current_region['columns'] = self._assign_columns_to_region(
                current_region['bounds'], 
                columns
            )
        regions.append(current_region)
        return regions
    
    def _assign_columns_to_region(self, region_bounds: Dict, columns: List[Dict]) -> List[Dict]:
        """
        Assign columns to a region based on vertical position overlap
        
        :param region_bounds: Dictionary containing region boundaries
        :param columns: List of text columns with position information
        :return: List of columns that belong to this region
        """
        region_columns = []
        
        for column in columns:
            # Skip if column is empty
            if not column:
                continue
            
            try:
                # Handle case where column is a list of text elements
                if isinstance(column, list):
                    # Get valid elements with bounds
                    valid_elements = [elem for elem in column if isinstance(elem, dict) and 'bounds' in elem]
                    
                    # Skip if no valid elements
                    if not valid_elements:
                        continue
                        
                    # Extract bounds from text elements in the column
                    column_top = min(elem['bounds']['top'] for elem in valid_elements)
                    column_bottom = max(elem['bounds']['bottom'] for elem in valid_elements)
                    column_left = min(elem['bounds']['left'] for elem in valid_elements)
                    column_right = max(elem['bounds']['right'] for elem in valid_elements)
                    
                    column_bounds = {
                        'top': column_top,
                        'bottom': column_bottom,
                        'left': column_left,
                        'right': column_right
                    }
                else:
                    # Handle case where column is a dictionary
                    column_bounds = column.get('bounds', {})
                    if not column_bounds:
                        continue
                
                # Calculate vertical overlap
                vertical_overlap = (
                    min(region_bounds['bottom'], column_bounds['bottom']) -
                    max(region_bounds['top'], column_bounds['top'])
                )
                
                # Calculate column height
                column_height = column_bounds['bottom'] - column_bounds['top']
                
                # If there's significant overlap (more than 50% of column height)
                if vertical_overlap > (column_height * 0.5):
                    # Check horizontal containment
                    if (column_bounds['left'] >= region_bounds['left'] - 20 and  # Allow small margin
                        column_bounds['right'] <= region_bounds['right'] + 20):
                        region_columns.append(column)
                    
            except Exception as e:
                print(f"Warning: Skipping invalid column due to: {str(e)}")
                continue
        
        return region_columns
    
    def visualize_grouped_regions(self, image, regions: List[Dict], output_path: str = None):
        """
        Visualize the grouped regions with orange boxes
        
        :param image: Original image (numpy array)
        :param regions: List of grouped regions
        :param output_path: Optional path to save the visualization
        """
        vis_img = image.copy()
        
        def draw_dashed_rectangle(img, bounds, color, thickness=2, dash_length=20):
            """Draw a dashed rectangle around the region"""
            points = [
                (bounds['left'], bounds['top']),
                (bounds['right'], bounds['top']),
                (bounds['right'], bounds['bottom']),
                (bounds['left'], bounds['bottom']),
                (bounds['left'], bounds['top'])
            ]
            
            for i in range(4):
                pt1, pt2 = points[i], points[i + 1]
                dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                dist = np.sqrt(dx * dx + dy * dy)
                
                if dist == 0:
                    continue
                    
                dashes = int(dist / (2 * dash_length))
                dashes = max(dashes, 1)
                
                for j in range(dashes):
                    start_ratio = j * 2 * dash_length / dist
                    end_ratio = min((j * 2 + 1) * dash_length / dist, 1)
                    
                    start = (int(pt1[0] + dx * start_ratio), 
                            int(pt1[1] + dy * start_ratio))
                    end = (int(pt1[0] + dx * end_ratio), 
                          int(pt1[1] + dy * end_ratio))
                    
                    cv2.line(img, start, end, color, thickness)
        
        # Draw regions
        for region in regions:
            # Draw orange dashed rectangle around region
            draw_dashed_rectangle(vis_img, region['bounds'], self.REGION_COLOR, thickness=2)
            
            # Optionally draw element boxes inside region
            for element in region['elements']:
                color = (0, 255, 0) if element['type'] == 'text' else (0, 0, 255)
                draw_dashed_rectangle(vis_img, element['bounds'], color, thickness=1)
        
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"Visualization saved to {output_path}")
        
        return vis_img 