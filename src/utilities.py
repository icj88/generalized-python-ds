# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:30:15 2025

@author: ijohnson
"""

from pathlib import Path
import yaml
from typing import Dict

def load_config() -> Dict[str, Any]:
    config_path = Path("./config/config_settings.yaml")
    config = yaml.safe_load("./")