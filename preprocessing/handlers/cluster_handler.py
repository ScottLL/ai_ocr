from collections import deque

class ClusterHandler:
    @staticmethod
    def cluster_blocks_2d(blocks, max_x_gap=30, max_y_gap=30,
                         min_x_overlap_ratio=0.3, min_y_overlap_ratio=0.0):
        """
        Enhanced BFS clustering that's more sensitive to large gaps between text clusters
        """
        def calculate_density(b1, b2):
            """Calculate the text density between two blocks"""
            # Get the bounding box that contains both blocks
            left = min(b1['bounds']['left'], b2['bounds']['left'])
            right = max(b1['bounds']['right'], b2['bounds']['right'])
            top = min(b1['bounds']['top'], b2['bounds']['top'])
            bottom = max(b1['bounds']['bottom'], b2['bounds']['bottom'])
            
            # Calculate area between blocks
            width = right - left
            height = bottom - top
            area = width * height
            
            # Calculate combined area of blocks
            b1_area = (b1['bounds']['right'] - b1['bounds']['left']) * (b1['bounds']['bottom'] - b1['bounds']['top'])
            b2_area = (b2['bounds']['right'] - b2['bounds']['left']) * (b2['bounds']['bottom'] - b2['bounds']['top'])
            
            # Calculate density
            return (b1_area + b2_area) / area if area > 0 else 0

        def should_be_connected(b1, b2):
            """Enhanced connection criteria considering gaps and density"""
            # Calculate basic metrics
            horiz_gap = max(0, min(b2['bounds']['left'] - b1['bounds']['right'],
                                b1['bounds']['left'] - b2['bounds']['right']))
            vert_gap = max(0, min(b2['bounds']['top'] - b1['bounds']['bottom'],
                                b1['bounds']['top'] - b2['bounds']['bottom']))
            
            # Calculate text density between blocks
            density = calculate_density(b1, b2)
            
            # Adjust thresholds based on density
            density_threshold = 0.15  # Minimum density to consider blocks connected
            large_gap_threshold = 100  # Pixels gap to consider as definite separation
            
            # Don't connect if there's a very large gap
            if horiz_gap > large_gap_threshold or vert_gap > large_gap_threshold:
                return False
                
            # Don't connect if density is too low (indicates sparse/separated clusters)
            if density < density_threshold:
                return False
                
            # Original proximity checks
            l1, r1 = b1['bounds']['left'], b1['bounds']['right']
            l2, r2 = b2['bounds']['left'], b2['bounds']['right']
            t1, b1 = b1['bounds']['top'], b1['bounds']['bottom']
            t2, b2 = b2['bounds']['top'], b2['bounds']['bottom']
            
            # Calculate overlaps
            x_overlap = max(0, min(r1, r2) - max(l1, l2))
            y_overlap = max(0, min(b1, b2) - max(t1, t2))
            
            min_width = min(r1 - l1, r2 - l2)
            min_height = min(b1 - t1, b2 - t2)
            
            x_overlap_ratio = x_overlap / min_width if min_width > 0 else 0
            y_overlap_ratio = y_overlap / min_height if min_height > 0 else 0
            
            # Combine all criteria
            return ((horiz_gap <= max_x_gap and y_overlap_ratio >= min_y_overlap_ratio) or
                    (vert_gap <= max_y_gap and x_overlap_ratio >= min_x_overlap_ratio))

        # Build adjacency list with enhanced connection criteria
        n = len(blocks)
        adj_list = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                if should_be_connected(blocks[i], blocks[j]):
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        # Run BFS to find clusters
        visited = [False] * n
        clusters = []
        
        for start_idx in range(n):
            if not visited[start_idx]:
                cluster = []
                queue = deque([start_idx])
                visited[start_idx] = True
                
                while queue:
                    cur = queue.popleft()
                    cluster.append(blocks[cur])
                    for neigh in adj_list[cur]:
                        if not visited[neigh]:
                            visited[neigh] = True
                            queue.append(neigh)
                
                clusters.append(cluster)
        
        return clusters

    @staticmethod
    def clusters_to_regions(clusters):
        """
        Convert each cluster (list of blocks) into a "mini-region" dict 
        with bounding box = union of blocks. Sort blocks linewise if you want.
        """
        def sort_cluster_linewise(blocks, y_threshold=10):
            """
            Sort blocks in a cluster in top->bottom + left->right reading order.
            Enhanced to better handle multi-column layouts.
            """
            # Group blocks by vertical position using y_threshold
            vertical_groups = []
            for block in sorted(blocks, key=lambda b: b['bounds']['top']):
                block_center_y = (block['bounds']['top'] + block['bounds']['bottom']) / 2
                
                # Try to add to existing group
                added_to_group = False
                for group in vertical_groups:
                    group_center_y = sum((b['bounds']['top'] + b['bounds']['bottom'])/2 for b in group) / len(group)
                    if abs(block_center_y - group_center_y) <= y_threshold:
                        group.append(block)
                        added_to_group = True
                        break
                
                # Create new group if needed
                if not added_to_group:
                    vertical_groups.append([block])

            # Sort blocks within each vertical group by x-position
            for group in vertical_groups:
                group.sort(key=lambda b: b['bounds']['left'])

            # Flatten groups into final sorted list
            return [block for group in vertical_groups for block in group]

        mini_regions = []
        for cl in clusters:
            # Optional linewise re-order
            sorted_blocks = sort_cluster_linewise(cl)

            top    = min(b['bounds']['top'] for b in sorted_blocks)
            bottom = max(b['bounds']['bottom'] for b in sorted_blocks)
            region = {
                'start_y': top,
                'end_y': bottom,
                'elements': sorted_blocks,
                'bounds': {
                    'top': top,
                    'bottom': bottom,
                    'left': min(b['bounds']['left'] for b in sorted_blocks),
                    'right': max(b['bounds']['right'] for b in sorted_blocks)
                }
            }
            mini_regions.append(region)

        # Sort all mini-regions top->bottom
        mini_regions.sort(key=lambda r: r['start_y'])
        return mini_regions
