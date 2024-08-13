import os 
import sys 
from pathlib import Path 
import logging
from logging import basicConfig, getLogger, FileHandler, StreamHandler
from rich.logging import RichHandler

logger_str_format = "[%(asctime)s:%(module)s:%(levelname)s:%(message)s]"

log_dir = Path('logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

log_file = Path(os.path.join(log_dir,'pipe.log'))
if not os.path.exists(log_file):
    with open(log_file, 'w') as file:
        pass 
    

basicConfig(
    level=logging.NOTSET,
    format=logger_str_format,
    handlers=[
        FileHandler(log_file),
        # StreamHandler(sys.stdout),
        RichHandler()
        ]
)

log = getLogger("rich")