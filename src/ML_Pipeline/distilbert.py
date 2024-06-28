import pandas as pd
import ktrain
from ktrain import text

class DistilBERT:

    def __init__(self):
        # Initialize the DistilBERT model settings
        self.model_name = "distilbert-base-uncased"  # Pre-trained DistilBERT model name
        self.maxlen = 512  # Maximum sequence length
        self.classes = ['Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder', 'MeanOfTransportation',
                        'Building', 'NaturalPlace', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'WrittenWork']
        # List of class labels for your classification task
        self.batch_size = 16  # Batch size for data processing

    def create_transformer(self):
        # Create and return a DistilBERT transformer using ktrain
        return text.Transformer(self.model_name, self.maxlen, self.classes, self.batch_size)
