class TextMerger:
    @staticmethod
    def merge_overlapping_results(results, overlap_threshold=0.5, position_threshold=5):
        """
        Enhanced merge logic to handle:
        1. Text boxes split mid-sentence
        2. Partial overlaps
        3. Boxes that continue each other's text
        """
        if not results:
            return []

        # Sort by x-coordinate and y-coordinate
        sorted_results = sorted(results, key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))
        merged = []
        current = sorted_results[0]

        def should_merge_text(text1, text2):
            """Check if texts should be merged based on content"""
            # Check if one text ends with part of the other
            text1_words = text1.split()
            text2_words = text2.split()
            
            # Check last few words of text1 appearing at start of text2
            for i in range(min(3, len(text1_words))):
                check_phrase = " ".join(text1_words[-(i+1):])
                if text2.startswith(check_phrase):
                    return True
                
            # Check first few words of text2 appearing at end of text1
            for i in range(min(3, len(text2_words))):
                check_phrase = " ".join(text2_words[:i+1])
                if text1.endswith(check_phrase):
                    return True
            
            return False

        def boxes_are_adjacent(box1, box2, threshold=50):
            """Check if boxes are adjacent or very close"""
            box1_left = min(p[0] for p in box1)
            box1_right = max(p[0] for p in box1)
            box1_top = min(p[1] for p in box1)
            box1_bottom = max(p[1] for p in box1)
            
            box2_left = min(p[0] for p in box2)
            box2_right = max(p[0] for p in box2)
            box2_top = min(p[1] for p in box2)
            box2_bottom = max(p[1] for p in box2)
            
            # Check horizontal adjacency
            horizontal_overlap = (box1_right >= box2_left - threshold and 
                                box1_left <= box2_right + threshold)
            
            # Check vertical adjacency
            vertical_overlap = (box1_bottom >= box2_top - threshold and 
                              box1_top <= box2_bottom + threshold)
            
            return horizontal_overlap and vertical_overlap

        for next_result in sorted_results[1:]:
            current_box, current_text, current_conf = current
            next_box, next_text, next_conf = next_result

            # Calculate positions
            current_left = min(p[0] for p in current_box)
            current_right = max(p[0] for p in current_box)
            current_top = min(p[1] for p in current_box)
            current_bottom = max(p[1] for p in current_box)
            
            next_left = min(p[0] for p in next_box)
            next_right = max(p[0] for p in next_box)
            next_top = min(p[1] for p in next_box)
            next_bottom = max(p[1] for p in next_box)

            # Check if boxes are at nearly identical position
            same_position = (abs(current_left - next_left) <= position_threshold and 
                            abs(current_top - next_top) <= position_threshold)

            # Check if boxes are adjacent and text continues
            text_continues = should_merge_text(current_text, next_text)
            boxes_adjacent = boxes_are_adjacent(current_box, next_box)

            if same_position or text_continues or boxes_adjacent:
                # Merge boxes
                merged_box = [
                    [min(current_left, next_left), min(current_top, next_top)],
                    [max(current_right, next_right), min(current_top, next_top)],
                    [max(current_right, next_right), max(current_bottom, next_bottom)],
                    [min(current_left, next_left), max(current_bottom, next_bottom)]
                ]
                
                # If same position, keep longer text
                if same_position:
                    merged_text = current_text if len(current_text) > len(next_text) else next_text
                else:
                    # Combine texts, avoiding duplicates
                    if text_continues:
                        # Find the overlap point and merge properly
                        merged_text = current_text
                        if not next_text.startswith(current_text):
                            for i in range(min(len(current_text.split()), 3)):
                                overlap = " ".join(current_text.split()[-(i+1):])
                                if next_text.startswith(overlap):
                                    merged_text = current_text + next_text[len(overlap):]
                                    break
                    else:
                        merged_text = f"{current_text} {next_text}"
                
                merged_conf = max(current_conf, next_conf)
                current = (merged_box, merged_text, merged_conf)
            else:
                merged.append(current)
                current = next_result

        merged.append(current)
        return merged
