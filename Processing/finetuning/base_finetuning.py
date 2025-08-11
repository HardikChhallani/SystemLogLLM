import logging
import os
from abc import abstractmethod, ABC
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL_PATH = "QWEN2/qwen2.5-1.5B-instruct"

class BaseFineTune(ABC):
    def __int__(self,
                model_name: str,
                logger = None,
                *args, **kwargs):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.logger = logger if logger else logging.getLogger(__name__)

        if not os.path.exists(local_model_path):
            self.load()

    def load(self):
        """
        Load the ADFA Qwen2 model and tokenizer.
        """
        try:
            self.logger.info("Loading ADFA Qwen2 model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.logger.info("Model and tokenizer loaded successfully.")
            self.logger.info(f"Saving model and tokenizer to: {self.local_model_path}")
            self.model.save_pretrained(LOCAL_MODEL_PATH)
            self.tokenizer.save_pretrained(LOCAL_MODEL_PATH)
            self.logger.info("✅ Model saved successfully to your local machine!")
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_name}: {e}")
            raise e

    @abstractmethod
    def finetune(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def upload_to_hf(self):
        pass