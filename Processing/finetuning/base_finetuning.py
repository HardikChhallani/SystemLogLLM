from abc import abstractmethod, ABC

class BaseFineTune(ABC):
    def __int__(self,
                model_name: str):
        self.model_name = model_name

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def finetune(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def upload_to_hf(self):
        pass