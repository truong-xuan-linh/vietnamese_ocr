from paddleocr import PaddleOCR
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import cv2
import requests
import unidecode
import numpy as np
from PIL import Image, ImageFont, ImageDraw

class OCRDetector:
    def __init__(self) -> None:
        self.paddle_ocr = PaddleOCR(lang='en', use_angle_cls=False)
        # config['weights'] = './weights/transformerocr.pth'
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = "./storage/ocr_model.pth"
        self.config['cnn']['pretrained']=False
        self.config['device'] =  "cpu"
        self.config['predictor']['beamsearch']=False
        self.viet_ocr = Predictor(self.config)
                
    def find_box(self, image):
        '''Xác định box dựa vào mô hình paddle_ocr'''
        result = self.paddle_ocr.ocr(image, cls = False)
        result = result[0]
        # Extracting detected components
        boxes = [res[0] for res in result] 
        texts = [{"text": res[1][0], "score": res[1][1]} for res in result]
        
        # scores = [res[1][1] for res in result]
        return boxes, texts
        
    def vietnamese_text(self, boxes, image):
        '''Xác định text dựa vào mô hình viet_ocr'''
        texts = []
        for box in boxes:
            A = box[0]
            B = box[1]
            C = box[2]
            D = box[3]
            y1 = min(A[1], B[1])
            y1 = int(max(0, y1 - max(0, 10 - abs(A[1] - B[1]))))
            y2 = max(C[1], D[1])
            y2 = int(y2 + max(0, 10 - abs(A[1] - B[1])))
            x1 = int(max(0, min(A[0], D[0]) ))
            x2 = int(max(B[0], C[0]) )
            cut_image = image[y1:y2, x1:x2]
            cut_image = Image.fromarray(np.uint8(cut_image))
            text, score = self.viet_ocr.predict(cut_image, return_prob=True)
            texts.append({"text": text,
                          "score": score})
        return texts

    #Merge
    def text_detector(self, image_path, is_local=False):
        if is_local:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        image = np.array(image)
        boxes, paddle_texts = self.find_box(image)
        if not boxes:
            return image, None, None
        viet_texts = self.vietnamese_text(boxes, image)
        results_texts = []
        for i, viet_txt in enumerate(viet_texts):
            if viet_txt["text"] != unidecode.unidecode(viet_txt["text"]):
                results_texts.append(viet_txt)
            else:
                results_texts.append(paddle_texts[i])
        if results_texts != []:
            return image, results_texts, boxes
        else:
            return image, None, None
    
    
    def visualize_ocr(self, image, texts, boxes):
        if not texts:
            return image
        
        img = image.copy()
        for box, text in zip(boxes, texts):
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
            
            h = y3 - y1
            scl = max(h//1000,1)
            font = ImageFont.truetype("./storage/Roboto-Black.ttf", 22*scl)
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x3), int(y3)), (0, 255, 0), 1)
            
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((int(x1), int(y1-h//2)), text["text"], font = font, fill = (255, 51, 51))
            img = np.array(img_pil)
            # img = cv2.putText(img, text["text"], (int(x1), int(y1)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
        return img
