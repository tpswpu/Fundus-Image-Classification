import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import logging
from core.trainer import TrainingEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config_path = "configs/swin_entropy_cfp.yaml"
    try:
        with open(config_path, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
        engine = TrainingEngine(config)
        engine.train()
    except Exception as e:
        logger.critical(f"Training failed: {e}", exc_info=True)