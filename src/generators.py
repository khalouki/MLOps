
import pandas as pd

def csv_line_generator(file_path):
    """Générateur qui lit un CSV ligne par ligne"""
    for chunk in pd.read_csv(file_path, chunksize=1):
        yield chunk.iloc[0].to_dict()
