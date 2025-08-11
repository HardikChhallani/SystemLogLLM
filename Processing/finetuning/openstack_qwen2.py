from base_finetuning import BaseFineTune
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MODEL_PATH = "models/openstack_qwen2"

class ADFAQwen2FineTune(BaseFineTune):
    def __init__(self):
        self.model_name = MODEL_NAME
        self.local_model_path = LOCAL_MODEL_PATH
        super().__init__(self.model_name,logger=logger)

    def finetune(self):
        # Implement fine-tuning logic for ADFA Qwen2 model
        pass

    def save_checkpoint(self):
        # Implement saving checkpoint logic for ADFA Qwen2 model
        pass

    def upload_to_hf(self):
        # Implement uploading to Hugging Face logic for ADFA Qwen2 model
        pass