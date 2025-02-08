class RegionHandler:
    @staticmethod
    def merge_region_blocks(cluster_of_region_blocks):
        """
        Enhanced region block merging with better text handling
        """
        final_regions = []
        for cluster in cluster_of_region_blocks:
            # Gather all elements and calculate bounds
            all_elements = []
            bounds = {
                'top': float('inf'),
                'bottom': float('-inf'),
                'left': float('inf'),
                'right': float('-inf')
            }
            
            for rb in cluster:
                all_elements.extend(rb['elements'])
                b = rb['bounds']
                bounds['top'] = min(bounds['top'], b['top'])
                bounds['bottom'] = max(bounds['bottom'], b['bottom'])
                bounds['left'] = min(bounds['left'], b['left'])
                bounds['right'] = max(bounds['right'], b['right'])

            # Sort elements top-to-bottom and left-to-right
            all_elements.sort(key=lambda e: (e['bounds']['top'], e['bounds']['left']))

            final_regions.append({
                'start_y': bounds['top'],
                'end_y': bounds['bottom'],
                'bounds': bounds,
                'elements': all_elements
            })

        return final_regions

    @staticmethod
    def merge_adjacent_regions(regions, vertical_gap_threshold=20, 
                             horizontal_gap_threshold=100, overlap_threshold=0.3):
        """
        Enhanced region merging with better gap detection
        """
        if not regions:
            return []

        def calculate_gap_density(r1, r2):
            """Calculate the text density in the gap between regions"""
            # Get the gap area
            if r1['bounds']['right'] < r2['bounds']['left']:
                gap_left = r1['bounds']['right']
                gap_right = r2['bounds']['left']
            else:
                gap_left = r2['bounds']['right']
                gap_right = r1['bounds']['left']
                
            gap_top = min(r1['bounds']['bottom'], r2['bounds']['top'])
            gap_bottom = max(r1['bounds']['top'], r2['bounds']['bottom'])
            
            gap_width = gap_right - gap_left
            gap_height = gap_bottom - gap_top
            
            if gap_width <= 0 or gap_height <= 0:
                return 1.0  # No gap
                
            # Count text elements in gap
            text_in_gap = sum(1 for e in r1['elements'] + r2['elements']
                            if e['type'] == 'text' and
                            e['bounds']['left'] >= gap_left and
                            e['bounds']['right'] <= gap_right and
                            e['bounds']['top'] >= gap_top and
                            e['bounds']['bottom'] <= gap_bottom)
                            
            return text_in_gap / (gap_width * gap_height) if gap_width * gap_height > 0 else 0

        def should_merge(r1, r2):
            """Enhanced merge decision with gap analysis"""
            # Calculate gaps
            vertical_gap = abs(r2['start_y'] - r1['end_y'])
            horizontal_gap = min(abs(r2['bounds']['left'] - r1['bounds']['right']),
                            abs(r1['bounds']['left'] - r2['bounds']['right']))
            
            # Calculate gap density
            gap_density = calculate_gap_density(r1, r2)
            
            # Don't merge if gap density is too low (indicates distinct clusters)
            if gap_density < 0.1:
                return False
                
            # Don't merge if gaps are too large
            if vertical_gap > vertical_gap_threshold * 2 or horizontal_gap > horizontal_gap_threshold * 2:
                return False
                
            # Calculate overlap
            overlap_top = max(r1['bounds']['top'], r2['bounds']['top'])
            overlap_bottom = min(r1['bounds']['bottom'], r2['bounds']['bottom'])
            overlap_height = max(0, overlap_bottom - overlap_top)
            
            min_height = min(r1['bounds']['bottom'] - r1['bounds']['top'],
                            r2['bounds']['bottom'] - r2['bounds']['top'])
            
            overlap_ratio = overlap_height / min_height if min_height > 0 else 0
            
            return overlap_ratio >= overlap_threshold or gap_density >= 0.2

        # Sort regions by vertical position
        regions = sorted(regions, key=lambda r: r['start_y'])
        
        # Iterative merging
        while True:
            merged_any = False
            merged_regions = []
            skip_next = False
            
            for i in range(len(regions)):
                if skip_next:
                    skip_next = False
                    continue
                    
                if i == len(regions) - 1:
                    merged_regions.append(regions[i])
                    continue
                
                current = regions[i]
                next_region = regions[i + 1]
                
                if should_merge(current, next_region):
                    # Merge regions
                    merged = {
                        'start_y': min(current['start_y'], next_region['start_y']),
                        'end_y': max(current['end_y'], next_region['end_y']),
                        'bounds': {
                            'left': min(current['bounds']['left'], next_region['bounds']['left']),
                            'right': max(current['bounds']['right'], next_region['bounds']['right']),
                            'top': min(current['bounds']['top'], next_region['bounds']['top']),
                            'bottom': max(current['bounds']['bottom'], next_region['bounds']['bottom'])
                        },
                        'elements': current['elements'] + next_region['elements']
                    }
                    merged_regions.append(merged)
                    skip_next = True
                    merged_any = True
                else:
                    merged_regions.append(current)
            
            if not merged_any:
                break
                
            regions = merged_regions
        
        return regions
