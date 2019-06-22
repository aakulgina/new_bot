from model import StyleTransferModel
from config import greeting, explaining, start_working_text, end_working_text
from telegram_token import token
import numpy as np
from PIL import Image
from io import BytesIO

model = StyleTransferModel()
content_image_file = {}

def greet_n_explain(bot, update):
    chat_id = update.message.chat_id
    bot.send_message(chat_id, text=greeting_text)
    print("Greeted {} and explained them the working principle".format(chat_id))

def send_prediction_on_photo(bot, update):
    # Нам нужно получить одну картинку, которая станет content image.
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    bot.send_message(chat_id, text=start_working_text)
    print("Notified {} about start of the working process".format(chat_id))

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    content_image_file[chat_id] = image_file

    content_image_stream = BytesIO()
    content_image_file[chat_id].download(out=content_image_stream)
    del content_image_file[chat_id]

    output = model.transfer_style(content_image_stream)

    # теперь отправим назад фото
    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    bot.send_message(chat_id, text=end_working_text)
    print("Notified {} before sending photo".format(chat_id))
    bot.send_photo(chat_id, photo=output_stream)
    print("Sent Photo to {}".format(chat_id))


if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    # используем прокси, так как без него у меня ничего не работало.
    # если есть проблемы с подключением, то попробуйте убрать прокси или сменить на другой
    # проекси ищется в гугле как "socks4 proxy"
    updater = Updater(token=token,  request_kwargs={'proxy_url': 'socks4://168.195.171.42:44880'})

    # В реализации большого бота скорее всего будет удобнее использовать Conversation Handler
    # вместо назначения handler'ов таким способом
    updater.dispatcher.add_handler(MessageHandler(greet_n_explain, Filters.photo, send_prediction_on_photo))
    updater.start_polling()
