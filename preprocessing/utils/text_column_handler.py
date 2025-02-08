class TextColumnHandler:
    @staticmethod
    def group_text_by_column_and_lines(regions, vertical_threshold=15, horizontal_threshold=100):
        """Group text elements into columns and lines."""
        def cluster_by_horizontal(text_elements, horizontal_threshold):
            sorted_elements = sorted(text_elements, key=lambda e: e['bounds']['left'])
            clusters = []
            current_cluster = []
            current_cluster_lefts = []
            for elem in sorted_elements:
                left = elem['bounds']['left']
                if not current_cluster:
                    current_cluster.append(elem)
                    current_cluster_lefts.append(left)
                else:
                    avg_left = sum(current_cluster_lefts) / len(current_cluster_lefts)
                    if abs(left - avg_left) > horizontal_threshold:
                        clusters.append(current_cluster)
                        current_cluster = [elem]
                        current_cluster_lefts = [left]
                    else:
                        current_cluster.append(elem)
                        current_cluster_lefts.append(left)
            if current_cluster:
                clusters.append(current_cluster)
            return clusters

        new_regions = []
        for region in regions:
            text_elements = [elem for elem in region['elements'] if elem['type'] == 'text']
            columns_clusters = cluster_by_horizontal(text_elements, horizontal_threshold)
            columns_clusters.sort(key=lambda col: sum(e['bounds']['left'] for e in col) / len(col))
            
            columns = []
            for col in columns_clusters:
                col.sort(key=lambda e: e['bounds']['top'])
                lines = []
                current_line_text = []
                current_line_tops = []
                current_line_bottoms = []
                current_line_lefts = []
                current_line_rights = []
                prev_bottom = None
                for elem in col:
                    top = elem['bounds']['top']
                    bottom = elem['bounds']['bottom']
                    left = elem['bounds']['left']
                    right = elem['bounds']['right']
                    if prev_bottom is not None and (top - prev_bottom) > vertical_threshold:
                        line_text = " ".join(current_line_text)
                        v_center = (min(current_line_tops) + max(current_line_bottoms)) / 2
                        avg_left = sum(current_line_lefts) / len(current_line_lefts)
                        avg_right = sum(current_line_rights) / len(current_line_rights)
                        lines.append({
                            'text': line_text, 
                            'v_center': v_center, 
                            'left': avg_left, 
                            'avg_right': avg_right
                        })
                        current_line_text = []
                        current_line_tops = []
                        current_line_bottoms = []
                        current_line_lefts = []
                        current_line_rights = []
                    current_line_text.append(elem['text'])
                    current_line_tops.append(top)
                    current_line_bottoms.append(bottom)
                    current_line_lefts.append(left)
                    current_line_rights.append(right)
                    prev_bottom = bottom
                if current_line_text:
                    line_text = " ".join(current_line_text)
                    v_center = (min(current_line_tops) + max(current_line_bottoms)) / 2
                    avg_left = sum(current_line_lefts) / len(current_line_lefts)
                    avg_right = sum(current_line_rights) / len(current_line_rights)
                    lines.append({
                        'text': line_text, 
                        'v_center': v_center, 
                        'left': avg_left, 
                        'avg_right': avg_right
                    })
                columns.append(lines)
            region['columns'] = columns
            new_regions.append(region)
        return new_regions

    @staticmethod
    def merge_close_columns(columns, merge_threshold=50):
        """Merge adjacent columns if their horizontal gap is small."""
        if not columns:
            return columns

        def col_boundaries(col):
            min_left = min(line.get('min_left', line.get('left')) for line in col)
            max_right = max(line.get('max_right', line.get('avg_right', line.get('left'))) for line in col)
            return min_left, max_right

        merged = []
        current = columns[0]
        for next_col in columns[1:]:
            current_min_left, current_max_right = col_boundaries(current)
            next_min_left, next_max_right = col_boundaries(next_col)
            gap = next_min_left - current_max_right
            if gap < merge_threshold:
                current = current + next_col
                current.sort(key=lambda l: l['v_center'])
            else:
                merged.append(current)
                current = next_col
        merged.append(current)
        return merged 