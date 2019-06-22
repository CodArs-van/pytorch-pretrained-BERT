import argparse
import logging
import os
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--masklm_model", type=str, required=True,
                    help="The path of pretrained bert model.")
args = parser.parse_args()

if __name__ == "__main__":
    path = args.masklm_model
    logger.info("path: {}".format(path))
    state_dict = torch.load(path)
    if hasattr(state_dict, "module"):
        state_dict = state_dict.module.state_dict()
    torch.save(state_dict, path)
    logger.info("save done!")