from typing import List, Tuple

class OffsetMapper:
    """
    Tracks character-level shifts between original and normalized text.
    Allows mapping normalized offsets back to original source offsets.
    """
    def __init__(self, original_text: str):
        self.original_text = original_text
        self.normalized_text = original_text
        # map[normalized_index] = original_index
        # We store it as a list of original indices for each position in normalized text
        self._mapping = list(range(len(original_text) + 1))

    def delete(self, start: int, end: int):
        """Record a deletion in the normalized text."""
        # normalized_text[start:end] is being removed
        # we need to remove the corresponding range from self._mapping
        del self._mapping[start:end]

    def replace(self, start: int, end: int, new_text: str):
        """Record a replacement in the normalized text."""
        # text[start:end] is replaced by new_text
        # The first char of new_text maps to the original start.
        # Subsequent chars of new_text also map to the original start (or spread depending on strategy)
        # Here we use the 'first-char' mapping for replacements.
        original_start = self._mapping[start]
        new_mapping = [original_start] * len(new_text)
        self._mapping[start:end] = new_mapping

    def insert(self, index: int, new_text: str):
        """Record an insertion in the normalized text."""
        original_index = self._mapping[index]
        new_mapping = [original_index] * len(new_text)
        self._mapping[index:index] = new_mapping

    def map_back(self, start: int, end: int) -> Tuple[int, int]:
        """Map normalized (start, end) back to original (start, end)."""
        if start >= len(self._mapping):
            orig_start = self._mapping[-1]
        else:
            orig_start = self._mapping[start]

        if end >= len(self._mapping):
            orig_end = self._mapping[-1]
        else:
            orig_end = self._mapping[end]
            
        return orig_start, orig_end

    def set_normalized(self, text: str):
        """Sets the final normalized text (internal use)."""
        self.normalized_text = text
