"""
Этот модуль содержит модель для обработки присланных фото.

Some code was taken from https://github.com/pytorch/examples/tree/master/fast_neural_style

"""

import numpy as np
from PIL import Image
import re
from scipy import misc
import torch
from torchvision import transforms
from transformer_net import TransformerNet


class StyleTransferModel:
    """
    Данный класс описывает используемую модель.

    """
    
    def __init__(self):
        """
        Инициализирует ряд неизменяемых переменных.

        К неизменяемым переменным относятся:
        размер output'a;
        имя устройства (gpu или cpu);
        перечень трансформаций, применяемых к исходному изображению;
        загрузка статов модели, обученной для переноса конкретного стиля;
        инициализация модели.

        """
        self.imsize = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.ToTensor()])
        self.state_dict = torch.load('mosaic.pth')
        self.style_model = TransformerNet()
        
        pass
    
    def transfer_style(self, content_img_stream):
        """
        Переносит дефолтный стиль (витражная мозаика)
        на пользовательскую картинку контента.

        """
        content_image = self.process_image(content_img_stream)
        with torch.no_grad():
            # Убираем лишние статы, записанные в процессе обучения модели.
            for k in list(self.state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del self.state_dict[k]
            self.style_model.load_state_dict(self.state_dict)
            self.style_model.to(self.device)
            output = self.style_model(content_image).cpu()
            output = np.array(output.squeeze(0))
            output = output.transpose(1, 2, 0).astype("uint8")
        
        return misc.toimage(output)

    def process_image(self, img_stream):
        """
        Подгружает картинку контента и применяет к ней ряд трансформаций.

        Перечень трансформаций:
        изменение размера картинки (256 пикселей по одной из сторон);
        превращение картинки в тензор.

        """
        print(self.device)
        image = Image.open(img_stream)
        image = self.loader(image).unsqueeze(0)
        
        return image.to(self.device, torch.float)
