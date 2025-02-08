import numpy as np

class NumpyConverter:
    @staticmethod
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: NumpyConverter.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [NumpyConverter.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return [NumpyConverter.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj 