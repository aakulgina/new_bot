"""
Some code was taken from https://github.com/pytorch/examples/tree/master/fast_neural_style
"""

import re
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from transformer_net import TransformerNet
from vgg import Vgg16

class StyleTransferModel:
    def __init__(self):
        self.imsize = 700
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = transforms.Compose([
            transforms.Resize(self.imsize),  # нормируем размер изображения
            transforms.ToTensor()])  # превращаем в удобный формат
        self.state_dict = torch.load('mosaic.pth')
        self.style_model = TransformerNet()
        
        pass
    
    def transfer_style(self, content_img_stream):
        content_image = self.process_image(content_img_stream)
        with torch.no_grad():
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(self.state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del self.state_dict[k]
            self.style_model.load_state_dict(self.state_dict)
            self.style_model.to(self.device)
            output = self.style_model(content_image).cpu()
            output = np.array(output.squeeze(0))
            output = output.transpose(1, 2, 0).astype("uint8")
        
        return output

    def process_image(self, img_stream):
        print(self.device)
        image = Image.open(img_stream)
        image = self.loader(image).unsqueeze(0)
        
        return image.to(self.device, torch.float)
