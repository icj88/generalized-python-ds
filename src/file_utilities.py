# -*- coding: utf-8 -*-
# %% 
"""
Created on Tue Jul 29 14:30:15 2025

@author: ijohnson
"""

from pathlib import Path
import yaml
from typing import Dict, List
from datetime import datetime
import re

DATASET_DIR = Path("C:/Users/ijohnson/OneDrive - University of Vermont/Documents/datasets")

# %%

class DataImporter:
    """Load data from a variety of sources into poandas dataframes"""
    
    def extract_date_from_fname(self, file: Path) -> datetime:
        fstem = file.stem
        re.
        
    
    def get_file_list(self, directory: Path) -> List:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found at {directory}")
        
        flist = []
        for root, directory, files in directory.walk():
            flist = files
        for
        return flist
    

        
def main():
    di = DataImporter()
    test_path = (DATASET_DIR / "UV_ACTIVE_EMPLS")
    flist = di.get_file_list(test_path)
    
    
    print("debug?")
    
    
if __name__ == "__main__":
    main()
    