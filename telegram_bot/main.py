"""
Данный модуль содержит основной код бота.

Бот устроен так, чтобы каждое его действие
по отношению к пользователю отображалось в логгинге.
Это сделано для того, чтобы было проще отследить,
на каком конкретно этапе возникла ошибка.

"""

from io import BytesIO
import numpy as np
from PIL import Image
from telegram.ext import CommandHandler
from config import end_working_text, see_you_text
from config import greeting_text, start_working_text
from model import StyleTransferModel
from telegram_token import token


model = StyleTransferModel()
content_image_file = {}


def greet_n_explain(bot, update):
    """
    Отправляет пользователю приветственное сообщение.

    Помимо команды /start, на эту функцию
    ссылается также команда /help.

    """
    chat_id = update.message.chat_id
    bot.send_message(chat_id, text=greeting_text)
    print("Greeted {} and explained the working principle".format(chat_id))


def send_prediction_on_photo(bot, update):
    """
    Достает из чата фото, обрабатывает его и присылает результат.

    """
    # Получаем одну картинку, которая станет картинкой контента.
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    bot.send_message(chat_id, text=start_working_text)
    print("Notified {} about start of the working process".format(chat_id))

    # Получаем информацию о картинке.
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    content_image_file[chat_id] = image_file

    content_image_stream = BytesIO()
    content_image_file[chat_id].download(out=content_image_stream)
    del content_image_file[chat_id]

    output = model.transfer_style(content_image_stream)

    # Теперь отправим назад фото.
    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    bot.send_message(chat_id, text=end_working_text)
    print("Notified {} before sending photo".format(chat_id))
    bot.send_photo(chat_id, photo=output_stream)
    print("Sent Photo to {}".format(chat_id))
    bot.send_message(chat_id, text=see_you_text)
    print("Sent {} further instructions".format(chat_id))


if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках.
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)

    updater = Updater(token=token,
        request_kwargs={'proxy_url': 'socks4://89.169.8.89:4145'})

    updater.dispatcher.add_handler(CommandHandler('start', greet_n_explain))
    updater.dispatcher.add_handler(CommandHandler('help', greet_n_explain))
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, send_prediction_on_photo))
    updater.start_polling()
