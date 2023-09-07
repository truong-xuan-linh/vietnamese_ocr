import os
import yaml

class Setup():
    def __init__(self) -> None:
        self.config = yaml.load(open("./config/config.yml"), yaml.loader.SafeLoader)
        self.ocr_model = self.config["ocr_model"]

    def ocr_model_downloader(self) -> None:
        os.system("python -m pip install gdown --upgrade")
        import gdown
        if "ocr_model.pth" not in os.listdir(("./storage")):
            gdown.download(self.ocr_model, "./storage/ocr_model.pth", quiet=False)