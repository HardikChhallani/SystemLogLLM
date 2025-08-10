from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """
    Base class for datasets.

    Datasets should inherit from this class and implement the required methods
    for preprocessing and create required data structures for training.
    """

    def __init__(self,
                 dataset_name: str,
                 dataset_path: str,
                 **kwargs):
        """
        Initialize the dataset.

        Args:
            dataset_name: Name of the dataset
            split: Split of the dataset
            **kwargs: Additional dataset-specific arguments
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    def load_csv(self):
        """
        Load dataset from a CSV file.

        Returns:
            Loaded dataset
        """
        import pandas as pd
        return pd.read_csv(self.dataset_path)

    @abstractmethod
    def format_csv(self):
        pass
