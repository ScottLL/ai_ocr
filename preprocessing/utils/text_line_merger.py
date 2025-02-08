class TextLineMerger:
    _nlp = None  # Class variable to store spacy model

    @staticmethod
    def improved_merge_lines_across_columns(regions, line_vertical_threshold=25):
        """Merge lines from all columns into rows."""
        new_regions = []
        for region in regions:
            all_lines = []
            if 'columns' not in region:
                new_regions.append(region)
                continue
            
            for col in region['columns']:
                for line in col:
                    all_lines.append(line)

            all_lines.sort(key=lambda l: l['v_center'])

            merged_rows = []
            current_row = []
            current_row_v = None

            for line in all_lines:
                if current_row and abs(line['v_center'] - current_row_v) > line_vertical_threshold:
                    current_row.sort(key=lambda l: l['left'])
                    merged_text = " ".join(l['text'] for l in current_row)
                    merged_rows.append(merged_text)
                    current_row = []

                current_row.append(line)
                current_row_v = sum(l['v_center'] for l in current_row) / len(current_row)

            if current_row:
                current_row.sort(key=lambda l: l['left'])
                merged_text = " ".join(l['text'] for l in current_row)
                merged_rows.append(merged_text)

            merged_paragraphs = TextLineMerger.split_large_blocks(merged_rows)
            region['merged_paragraphs'] = merged_paragraphs
            new_regions.append(region)
        return new_regions

    @staticmethod
    def split_large_blocks(merged_rows):
        """Split large merged paragraphs based on heuristic rules."""
        paragraphs = []
        current_paragraph = []

        for line in merged_rows:
            if len(line.split()) < 5 or line.endswith(":"):
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        return paragraphs

    @staticmethod
    def semantic_similarity(text1, text2):
        """Calculate semantic similarity between two text segments."""
        try:
            # Lazy load spacy only when this method is called
            if TextLineMerger._nlp is None:
                try:
                    import spacy
                    TextLineMerger._nlp = spacy.load('en_core_web_sm')
                except ImportError:
                    print("Warning: spacy not installed. Semantic similarity will return 0.0")
                    return 0.0
                except OSError:
                    print("Warning: en_core_web_sm not found. Run 'python -m spacy download en_core_web_sm'")
                    return 0.0
            
            doc1 = TextLineMerger._nlp(text1.lower().strip())
            doc2 = TextLineMerger._nlp(text2.lower().strip())
            
            tokens1 = [token for token in doc1 if not token.is_stop and not token.is_punct]
            tokens2 = [token for token in doc2 if not token.is_stop and not token.is_punct]
            
            if not tokens1 or not tokens2:
                return 0.0
                
            similarity = doc1.similarity(doc2)
            return float(similarity)
        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {str(e)}")
            return 0.0 