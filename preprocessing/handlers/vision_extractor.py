import base64
import requests
import json
import cv2
import numpy as np
from typing import List, Dict
import os
from multiprocessing import Pool

class VisionExtractor:
    def __init__(self, model: str = "llama3.2-vision:11b", api_url: str = "http://localhost:11434/v1/chat/completions"):
        """
        Initialize VisionExtractor
        
        :param model: Vision model to use
        :param api_url: API endpoint URL
        """
        self.model = model
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json"
        }
        self.max_processes = 6  # Set maximum number of parallel processes

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64
        
        :param image: numpy array of the image
        :return: base64 encoded string
        """
        # Convert numpy array to jpg image
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Could not encode image")
            
        # Convert to base64
        return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

    def _process_single_region(self, args) -> Dict:
        """
        Process a single region (used for parallel processing)
        
        :param args: Tuple of (image, region)
        :return: Dictionary with region information and vision model response
        """
        image, region = args
        return self.extract_region_info(image, region)

    def extract_region_info(self, image: np.ndarray, region: Dict) -> Dict:
        """
        Extract information about a specific region using vision model
        
        :param image: Full image as numpy array
        :param region: Region dictionary containing bounds information
        :return: Dictionary with region information and vision model response
        """
        try:
            # Extract region from image using bounds and convert to integers
            bounds = {
                'top': int(region['bounds']['top']),
                'bottom': int(region['bounds']['bottom']),
                'left': int(region['bounds']['left']),
                'right': int(region['bounds']['right'])
            }
            region_image = image[bounds['top']:bounds['bottom'], 
                               bounds['left']:bounds['right']]
            
            # Encode the region image
            encoded_image = self._encode_image(region_image)
            
            # Prepare the request
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the structure of this image region, output the image structure and OCR all the text from the image. Focus on the spatial layout and relationships between elements."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    ]
                }]
            }

            # Make the request
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Add vision model response to region info
            region_info = {
                'start_y': bounds['top'],
                'end_y': bounds['bottom'],
                'bounds': bounds,
                'elements': region['elements'],
                'vision_description': response.json()
            }
            
            return region_info
            
        except requests.exceptions.RequestException as e:
            print(f"Error processing region: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error processing region: {str(e)}")
            return None

    def process_regions(self, image: np.ndarray, regions: List[Dict], output_path: str) -> None:
        """Process all regions and save results to JSON"""
        try:
            vision_results = []
            
            for idx, region in enumerate(regions):
                # Get region bounds and convert to integers
                bounds = {
                    'top': int(region['bounds']['top']),
                    'bottom': int(region['bounds']['bottom']),
                    'left': int(region['bounds']['left']),
                    'right': int(region['bounds']['right'])
                }
                
                # Crop region from image
                region_image = image[bounds['top']:bounds['bottom'], 
                                   bounds['left']:bounds['right']]
                
                # Skip empty regions
                if region_image.size == 0:
                    continue
                    
                # Encode image
                base64_image = self._encode_image(region_image)
                
                # Create prompt for vision model
                prompt = {
                    "model": self.model,
                    "messages": [{
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": "Describe the visual layout and content of this region. Focus on the spatial arrangement and visual elements:"
                            },
                            {
                                "type": "image_url", 
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }]
                }
                
                # Get response from API
                response = requests.post(self.api_url, headers=self.headers, json=prompt)
                response.raise_for_status()
                
                # Extract text content from columns in this region
                text_columns = []
                current_column = []
                
                # Get all text elements from the region's elements
                text_elements = []
                for element in region.get('elements', []):
                    if element.get('type') == 'text' and 'text' in element:
                        # Check if text element is within region bounds
                        if (element.get('v_center', 0) >= bounds['top'] and 
                            element.get('v_center', 0) <= bounds['bottom'] and
                            element.get('left', 0) >= bounds['left'] and
                            element.get('avg_right', 0) <= bounds['right']):
                            text_elements.append({
                                'text': element['text'],
                                'v_center': element.get('v_center', 0),
                                'left': element.get('left', 0),
                                'avg_right': element.get('avg_right', 0)
                            })
                
                # Sort text elements by vertical position and left position
                text_elements.sort(key=lambda x: (x['v_center'], x['left']))
                
                # Group text elements into columns based on horizontal position
                if text_elements:
                    current_column = [text_elements[0]]
                    for elem in text_elements[1:]:
                        # If element is horizontally aligned with previous element
                        if abs(elem['left'] - current_column[0]['left']) < 50:  # Adjust threshold as needed
                            current_column.append(elem)
                        else:
                            # Start new column
                            if current_column:
                                text_columns.append(current_column)
                            current_column = [elem]
                    
                    # Add last column
                    if current_column:
                        text_columns.append(current_column)
                
                # Create section with vision description and text content
                section = {
                    "region_id": idx + 1,
                    "bounds": bounds,
                    "vision_description": response.json()['choices'][0]['message']['content'],
                    "text_content": {
                        "columns": text_columns,
                        "merged_paragraphs": region.get('merged_paragraphs', [])
                    }
                }
                
                vision_results.append(section)
            
            # Create final output structure
            final_output = {
                "total_regions": len(vision_results),
                "regions": vision_results
            }
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
                
            print(f"Vision analysis results saved to {output_path}")
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise e

    def _generate_final_analysis(self, results: List[Dict]) -> str:
        """
        Generate a final analysis of all sections
        
        :param results: List of processed region results
        :return: String containing the final analysis
        """
        try:
            # Prepare the prompt for final analysis
            sections_text = []
            for i, result in enumerate(results):
                vision_desc = result.get('vision_description', {})
                if 'choices' in vision_desc:
                    content = vision_desc['choices'][0]['message']['content']
                    sections_text.append(f"Section {i+1}: {content}")
            
            # Create analysis prompt
            prompt = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Based on these sections, provide a comprehensive analysis:\n\n" + 
                                   "\n\n".join(sections_text)
                        }
                    ]
                }]
            }
            
            # Get analysis from API
            response = requests.post(self.api_url, headers=self.headers, json=prompt)
            response.raise_for_status()
            
            return response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error generating final analysis: {str(e)}")
            return "Error generating final analysis"