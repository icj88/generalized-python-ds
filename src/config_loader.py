import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
from pathlib import Path
from typing import Any, Dict


# %%
def load_config(config_file: Path = Path('../config/config_settings.yaml')) -> Dict[str, Any]:
    """Load and apply visualization settings from YAML."""
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Apply matplotlib settings
    mpl_settings = config.get('matplotlib', {})
    for key, value in mpl_settings.items():
        if key == 'axes.prop_cycle':
            mpl.rcParams[key] = cycler('color', value)
        else:
            mpl.rcParams[key] = value
    
    # Apply seaborn settings
    sns_settings = config.get('seaborn', {})
    if 'style' in sns_settings:
        sns.set_style(sns_settings['style'])
    if 'palette' in sns_settings:
        sns.set_palette(sns_settings['palette'])
    if 'context' in sns_settings:
        sns.set_context(sns_settings['context'])
    
    # Create output directory
    output_dir = config.get('output', {}).get('dir', './figures')
    Path(output_dir).mkdir(exist_ok=True)
    
    return config

# %%
if __name__ == "__main__":
    # Initialize configuration
    config = load_config()
    
    # Create a sample plot
    import numpy as np
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label='sin(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Sample Plot')
    ax.legend()
    
    plt.show()
    print('deeebug.')