import cv2
import numpy as np
import easyocr
import concurrent.futures
from preprocessing.handlers.block_detector import BlockDetector
from preprocessing.handlers.cluster_handler import ClusterHandler
from preprocessing.handlers.region_handler import RegionHandler
from preprocessing.handlers.rotation_handler import RotationHandler
from preprocessing.handlers.visualization_handler import VisualizationHandler
from preprocessing.utils.text_merger import TextMerger
from preprocessing.core.text_detector import TextDetector
from preprocessing.handlers.object_detector import ObjectDetector

class ImprovedImagePreprocessor:
    def __init__(self, image_path, reader=None):
        """
        :param image_path: path to the input image
        :param reader: an EasyOCR Reader object; if None, one is created
        """
        self.image_path = image_path
        self.reader = reader if reader else easyocr.Reader(['en'], gpu=False)
        self.text_detector = TextDetector(reader=self.reader)
        self.object_detector = ObjectDetector()
        self.image = None
        self.gray_image = None

        # Lists of detected items
        self.text_blocks = []    # each: {type='text', text=..., bounds={left,top,right,bottom}, confidence=...}
        self.image_blocks = []   # each: {type='image', bounds={left,top,right,bottom}, area=...}

        # Final output regions for improved_preprocessor
        self.regions = []        # each region is {start_y, end_y, elements: [...]}

    def process(self):
        """
        Main processing pipeline:
        1) Load image
        2) Enhanced OCR text detection and recognition
        3) Improved image region detection
        4) Phase-1 BFS (strict) => small clusters
        5) Convert small clusters => "mini-regions"
        6) Phase-2 BFS (looser) merges mini-regions => final "big" regions
        7) Sort + return
        """
        self._load_image()
        self._detect_text_blocks()
        self._detect_images_morphological()

        # Combine all blocks
        all_blocks = self.text_blocks + self.image_blocks

        # --------- Phase 1: Strict BFS on individual blocks ----------
        strict_clusters = ClusterHandler.cluster_blocks_2d(
            blocks=all_blocks, 
            max_x_gap=20,        
            max_y_gap=25,        
            min_x_overlap_ratio=0.2,  
            min_y_overlap_ratio=0.1   
        )

        # Convert to "mini-regions"
        mini_regions = ClusterHandler.clusters_to_regions(strict_clusters)

        # --------- Phase 2: Looser BFS on these mini-region bounding boxes ----------
        region_blocks = []
        for r in mini_regions:
            region_bounds = {
                'left':   min(e['bounds']['left']   for e in r['elements']),
                'right':  max(e['bounds']['right']  for e in r['elements']),
                'top':    min(e['bounds']['top']    for e in r['elements']),
                'bottom': max(e['bounds']['bottom'] for e in r['elements'])
            }
            region_blocks.append({
                'type': 'regionblock',
                'elements': r['elements'],
                'bounds': region_bounds
            })

        # Phase 2 BFS with adjusted parameters
        big_clusters = ClusterHandler.cluster_blocks_2d(
            blocks=region_blocks,
            max_x_gap=25,       
            max_y_gap=25,       
            min_x_overlap_ratio=0.1,  
            min_y_overlap_ratio=0.1   
        )

        # Merge clusters into final regions
        final_regions = RegionHandler.merge_region_blocks(big_clusters)

        # Merge adjacent regions
        final_regions = RegionHandler.merge_adjacent_regions(
            final_regions,
            vertical_gap_threshold=20,  
            horizontal_gap_threshold=100,
            overlap_threshold=0.3
        )

        # Sort top->bottom and assign to self.regions
        final_regions.sort(key=lambda r: r['start_y'])
        self.regions = final_regions
        return self.regions

    def _load_image(self):
        """Load and prepare the image"""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def _detect_text_blocks(self):
        """
        Enhanced text block detection using TextDetector
        """
        if self.image is None or self.gray_image is None:
            raise RuntimeError("Image not loaded. Call _load_image first.")

        # Use TextDetector to detect text blocks
        self.text_blocks = self.text_detector.detect_text_blocks(self.image, self.gray_image)

    def _detect_text_regions(self):
        """
        Get text regions mask using TextDetector's internal method
        """
        return self.text_detector._detect_text_regions(self.gray_image)

    def _detect_images_morphological(self):
        """
        Detect image regions using morphological operations
        """
        if self.image is None or self.gray_image is None:
            raise RuntimeError("Image not loaded. Call _load_image first.")

        # Create text mask from text blocks
        height, width = self.gray_image.shape
        text_mask = np.zeros((height, width), dtype=np.uint8)
        for block in self.text_blocks:
            x1 = block['bounds']['left']
            y1 = block['bounds']['top']
            x2 = block['bounds']['right']
            y2 = block['bounds']['bottom']
            text_mask[y1:y2, x1:x2] = 255

        # Use ObjectDetector to find image regions
        self.image_blocks = self.object_detector.detect_objects(self.image, text_mask)

    def _blocks_are_close_enough(self, b1, b2, max_x_gap=20, max_y_gap=20,
                               min_x_overlap_ratio=0.3, min_y_overlap_ratio=0.0):
        """Enhanced proximity check for blocks using BlockDetector"""
        return BlockDetector.blocks_are_close_enough(
            b1, b2, 
            max_x_gap=max_x_gap,
            max_y_gap=max_y_gap,
            min_x_overlap_ratio=min_x_overlap_ratio,
            min_y_overlap_ratio=min_y_overlap_ratio
        )

    def visualize_regions(self, output_path=None, show_confidence=True, show_boxes=True):
        """
        Enhanced visualization with improved text box merging
        """
        if self.image is None:
            raise RuntimeError("No image loaded. Did you call .process() yet?")

        return VisualizationHandler.visualize_regions(
            image=self.image,
            regions=self.regions,
            output_path=output_path,
            show_confidence=show_confidence,
            show_boxes=show_boxes
        )

    def merge_overlapping_results(self, results, overlap_threshold=0.5, position_threshold=5):
        """
        Enhanced merge logic to handle text box merging
        """
        return TextMerger.merge_overlapping_results(
            results,
            overlap_threshold=overlap_threshold,
            position_threshold=position_threshold
        )