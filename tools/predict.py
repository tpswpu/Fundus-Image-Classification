import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import logging
from core.evaluator import EvaluatorEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config_path = "configs/swin_entropy_cfp.yaml"
    try:
        with open(config_path, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
        evaluator = EvaluatorEngine(config)
        evaluator.evaluate()
    except Exception as e:
        logger.critical(f"Prediction failed: {e}", exc_info=True)
