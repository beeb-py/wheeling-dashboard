import pandas as pd
from typing import List, Optional, Tuple

class FileLoader:
    """
    Loads and validates CSV or Excel files.
    """

    def __init__(self):
        pass

    def load_file(self, file, expected_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Loads csv or excel file based on extension.
        """

        filename = file.name.lower()

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file type. Upload CSV or Excel only.")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

        # Validate columns
        if expected_columns is not None:
            missing = [col for col in expected_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

        return df
