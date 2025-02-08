import os
import sys
import traceback
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import easyocr
import json
from preprocessing.core.image_preprocessor import ImprovedImagePreprocessor
from preprocessing.utils.numpy_converter import NumpyConverter
from preprocessing.utils.text_column_handler import TextColumnHandler
from preprocessing.utils.text_line_merger import TextLineMerger
from preprocessing.handlers.region_grouper import RegionGrouper
from preprocessing.handlers.vision_extractor import VisionExtractor


def test_improved_preprocessor(image_path):
    """Test the improved image preprocessor, ensuring each region group is processed individually."""

    # Get the directory where image_test.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    reader = easyocr.Reader(['en', 'ch_sim'])
    preprocessor = ImprovedImagePreprocessor(image_path, reader)

    try:
        # 1) Preprocess the image and get initial regions. 
        #    (These "regions" are often just bounding boxes from text detection.)
        regions = preprocessor.process()

        # 2) Merge overlapping text results within each region
        for region in regions:
            if 'elements' in region:
                text_elements = [(
                    [
                        [e['bounds']['left'], e['bounds']['top']],
                        [e['bounds']['right'], e['bounds']['top']],
                        [e['bounds']['right'], e['bounds']['bottom']],
                        [e['bounds']['left'], e['bounds']['bottom']]
                    ],
                    e['text'],
                    e.get('confidence', 1.0)
                ) for e in region['elements'] if e['type'] == 'text']
                
                merged_elements = preprocessor.merge_overlapping_results(
                    text_elements,
                    overlap_threshold=0.5,
                    position_threshold=5
                )
                
                new_elements = []
                for box, text, conf in merged_elements:
                    left = min(p[0] for p in box)
                    right = max(p[0] for p in box)
                    top = min(p[1] for p in box)
                    bottom = max(p[1] for p in box)
                    
                    new_elements.append({
                        'type': 'text',
                        'text': text,
                        'confidence': conf,
                        'bounds': {
                            'left': left,
                            'right': right,
                            'top': top,
                            'bottom': bottom
                        }
                    })
                
                non_text_elements = [e for e in region['elements'] if e['type'] != 'text']
                region['elements'] = non_text_elements + new_elements

        # Convert numpy types to native Python types
        converted_regions = NumpyConverter.convert_numpy_types(regions)

        # Collect *all* text/image elements into one list
        all_elements = []
        for region in converted_regions:
            if 'elements' in region:
                all_elements.extend(region['elements'])

        # 3) Group elements into multiple region "groups" based on vertical proximity
        region_grouper = RegionGrouper(vertical_gap_threshold=50)
        grouped_regions = region_grouper.group_elements_into_regions(all_elements)
        
        # Optional: visualize the grouped regions
        region_grouper.visualize_grouped_regions(
            preprocessor.image,
            grouped_regions,
            "grouped_regions_visualization.jpg"
        )

        #######################################################
        #  NOTE: we treat each grouped
        #        region individually, extracting its columns,
        #        merging lines, and so on.
        #######################################################
        final_regions = []
        for g_idx, region_group in enumerate(grouped_regions, start=1):
            # region_group['elements'] holds the elements for THIS region group

            # Put this single region into a structure so we can run the column handler:
            single_region_list = [region_group]

            # 4) Process text columns within this *single region*
            with_columns = TextColumnHandler.group_text_by_column_and_lines(
                single_region_list,
                vertical_threshold=15,
                horizontal_threshold=100
            )
            
            # Merge columns that are very close
            for r in with_columns:
                r['columns'] = TextColumnHandler.merge_close_columns(
                    r['columns'], 
                    merge_threshold=50
                )
            
            # 5) Merge lines into paragraphs
            merged_line_output = TextLineMerger.improved_merge_lines_across_columns(
                with_columns,
                line_vertical_threshold=25
            )

            # merged_line_output is a list with a single item—our region dictionary
            region_final = merged_line_output[0]

            # Assign an ID for clarity
            region_final["region_id"] = g_idx

            # Add this newly processed region group to final list
            final_regions.append(region_final)

        ####################################################
        # Before passing final_regions to the VisionExtractor,
        # Let's save the merged paragraphs to a JSON file.
        ####################################################
        ocr_results_path = os.path.join(current_dir, 'ocr_results.json')
        ocr_data = []
        for region_dict in final_regions:
            region_id = region_dict.get("region_id", None)
            paragraphs = region_dict.get("merged_paragraphs", [])
            ocr_data.append(
                " ".join(paragraphs)
            )

        with open(ocr_results_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_data, f, indent=2, ensure_ascii=False)

        print(f"\nOCR results saved to '{ocr_results_path}'")

        ###############################################    
        # 6) Pass final_regions (now truly multiple
        #    region groups) to the VisionExtractor
        ###############################################
        vision_extractor = VisionExtractor()
        vision_output_path = os.path.join(current_dir, 'vision_analysis_results.json')

        vision_extractor.process_regions(
            image=preprocessor.image,
            regions=final_regions,  # multiple region groups to be processed
            output_path=vision_output_path
        )

        # 7) Print merged paragraphs to confirm logic
        print("\nMerged Paragraphs by Region Group:")
        print("----------------------------------")
        for i, region_dict in enumerate(final_regions, start=1):
            if 'merged_paragraphs' in region_dict:
                print(f"\nRegion Group {i} (ID={region_dict['region_id']}):")
                for j, paragraph in enumerate(region_dict['merged_paragraphs']):
                    print(f"  Paragraph {j+1}: {paragraph}")

        return final_regions

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return None

def save_regions_to_json(regions, output_file='region_results.json'):
    """Save regions to a JSON file."""
    if regions:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(regions, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to '{output_file}'")
        except Exception as e:
            print(f"Error saving JSON: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # image_path = "../image.jpg"
    # image_path = "../output.jpg"
    # image_path = "../image2.png"
    # image_path = "../333.jpeg"
    image_path = "../222.webp"
    regions = test_improved_preprocessor(image_path)
    if regions:
        save_regions_to_json(regions, output_file='region_results.json')