"""
Some code was taken from https://github.com/pytorch/examples/tree/master/fast_neural_style
"""

import re
from PIL import Image
import torch
from torch.optim import Adam
from torchvision import transforms

from transformer_net import TransformerNet
from vgg import Vgg16

class StyleTransferModel:
    def __init__(self):
        # Сюда необходимо перенести всю иницализацию, вроде загрузки свеерточной сети и т.д.
        pass

    def transfer_style(self, content_img_stream):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        content_image = self.process_image(content_img_stream)
        model_path = '../../mosaic.pth'
        
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(model_path)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            output = style_model(content_image).cpu()
            # НЕ ЗАБУДЬ ВОТ ТУТ ПОДУМАТЬ ПРО ТО, ЧТОБЫ СДЕЛАТЬ ЕМУ СКВИЗ
            # А ТО ЕСТЬ ОПАСНОСТЬ, ЧТО ТАК ОНО НА СТАДИИ БОТА НИХРЕНА НЕ ЗАРАБОТАЕТ
            # а еще есть смысл подумать о том, чтобы махнуть местами дайменшнс
            # и, МЭЙБИ, о том, чтобы сделать из этого нампи эррэй
        
        return output

    def process_image(self, img_stream):
        # TODO размер картинки, device и трансформации не меняются в течении всей работы модели,
        # поэтому их нужно перенести в конструктор!
        imsize = 512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        loader = transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # превращаем в удобный формат

        image = Image.open(img_stream)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)
