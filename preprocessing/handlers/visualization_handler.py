import cv2
import numpy as np
import matplotlib.pyplot as plt

class VisualizationHandler:
    @staticmethod
    def visualize_regions(image, regions, output_path=None, show_confidence=True, show_boxes=True):
        """
        Visualization matching the original output style
        """
        if image is None:
            raise RuntimeError("No image provided for visualization")

        vis_img = image.copy()

        # Color schemes (BGR format)
        TEXT_COLOR = (0, 255, 0)     # Green for text
        IMAGE_COLOR = (0, 0, 255)    # Red for images
        REGION_COLOR = (255, 0, 0)   # Blue for region boundaries

        # Add debug print to check regions
        print(f"Number of regions to visualize: {len(regions)}")

        def draw_block(img, block, color, alpha=0.2, thickness=2):
            bounds = block['bounds']
            left = int(bounds['left'])
            top = int(bounds['top'])
            right = int(bounds['right'])
            bottom = int(bounds['bottom'])
            
            # For text blocks, use semi-transparent fill
            if block['type'] == 'text':
                overlay = img.copy()
                cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                
                # Draw border
                if show_boxes:
                    cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
            
            # For image blocks, draw thicker border
            else:
                cv2.rectangle(img, (left, top), (right, bottom), color, thickness)

        def draw_dashed_line(img, start_point, end_point, color, thickness=2, dash_length=15):
            x1, y1 = start_point
            x2, y2 = end_point
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx*dx + dy*dy)
            dashes = int(dist / (2 * dash_length))
            
            # Ensure at least one dash is drawn
            dashes = max(dashes, 1)
            
            for i in range(dashes):
                start_ratio = i * 2 * dash_length / dist
                end_ratio = min((i * 2 + 1) * dash_length / dist, 1)
                
                start = (int(x1 + dx*start_ratio), int(y1 + dy*start_ratio))
                end = (int(x1 + dx*end_ratio), int(y1 + dy*end_ratio))
                
                cv2.line(img, start, end, color, thickness)

        # Draw regions first (so they appear behind blocks)
        for idx, region in enumerate(regions):
            if 'elements' in region and region['elements']:
                print(f"Region {idx} has {len(region['elements'])} elements")
                # Get region bounds
                left = int(min(e['bounds']['left'] for e in region['elements']))
                right = int(max(e['bounds']['right'] for e in region['elements']))
                top = int(region['start_y'])
                bottom = int(region['end_y'])
                
                # Draw dashed rectangle with thicker lines
                points = [(left, top), (right, top), 
                         (right, bottom), (left, bottom)]
                for i in range(4):
                    draw_dashed_line(vis_img, points[i], points[(i+1)%4], REGION_COLOR, 
                                   thickness=2, dash_length=15)

        # Then draw blocks with thicker borders
        for region in regions:
            if 'elements' in region:
                for element in region['elements']:
                    print(f"Drawing element of type: {element['type']}")
                    color = TEXT_COLOR if element['type'] == 'text' else IMAGE_COLOR
                    draw_block(vis_img, element, color, thickness=2)

        if output_path:
            success = cv2.imwrite(output_path, vis_img)
            print(f"Image saved to {output_path}: {'Success' if success else 'Failed'}")
        else:
            plt.figure(figsize=(12,12))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
