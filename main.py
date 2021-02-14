"""
This is a bot that uses neural network to transfer style
For any comments contact telegram: @jenya_pu
"""

import logging
import aiogram.utils.markdown as md
import PIL
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ParseMode
from aiogram import Bot, Dispatcher, executor, types
import style_changer

API_TOKEN = '1606767587:AAH7EIHDYeX1rpqaijD0f4PR3XROOfD6rKo'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("Привет, я бот!\nЯ обрабатываю картинки с помощью нейронной сети. Пришли мне две картинки - и "
                        "я наложу стиль второй картинки на первую.\nЕсли есть вопросы - пишите @jenya_pu")


# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# States
class Form(StatesGroup):
    first_image = State()  # Will be represented in storage as 'Form:first_image'
    second_image = State()  # Will be represented in storage as 'Form:second_image'
    what_to_do = State()  # Will be represented in storage as 'Form:what_to_do'


@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    await Form.first_image.set()
    await message.reply("Привет! Пришли мне первую картинку.")


# You can use state '*' if you need to handle all states
@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return
    logging.info('Cancelling state %r', current_state)
    # Cancel state and inform user about it
    await state.finish()
    # And remove keyboard (just in case)
    await message.reply('Отменено.', reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(state=Form.first_image, content_types=['photo'])
async def process_first_image(message: types.Message, state: FSMContext):
    async with state.proxy():
        await message.photo[-1].download('images/photo1.jpg')
    size = 512, 512
    im = PIL.Image.open('images/photo1.jpg')
    im.thumbnail(size, PIL.Image.ANTIALIAS)
    im.save('images/photo1.jpg', "JPEG")
    await Form.next()
    await Form.second_image.set()
    await state.update_data(photo_1=message.photo)
    await message.reply("Пришли вторую картинку")


# Check message. It must be an image
@dp.message_handler(lambda message: not message.photo, state=Form.second_image)
async def process_image_invalid(message: types.Message):
    return await message.reply("Здесь должно быть изображение.")


@dp.message_handler(state=Form.second_image, content_types=['photo'])
async def process_second_image(message: types.Message, state: FSMContext):
    async with state.proxy():
        await message.photo[-1].download('images/photo2.jpg')
    size = 512, 512
    im = PIL.Image.open('images/photo2.jpg')
    im.thumbnail(size, PIL.Image.ANTIALIAS)
    im.save('images/photo2.jpg', "JPEG")
    # Update state and data
    await Form.next()
    await state.update_data(photo_2=message.photo)

    # Configure ReplyKeyboardMarkup
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("первую на вторую")

    await message.reply("Какой стиль наложить на какую картинку?", reply_markup=markup)


@dp.message_handler(state=Form.what_to_do)
async def select_operation(message: types.Message, state: FSMContext):
    # Remove keyboard
    markup = types.ReplyKeyboardRemove()

    # And send message
    await bot.send_message(
        message.chat.id,
        md.text('Сейчас сделаю обработанное изображение. Нужно подождать минутку.'),
        reply_markup=markup,
        parse_mode=ParseMode.MARKDOWN,
    )
    style_changer.make_style_transfer()

    await bot.send_photo(
        message.chat.id,
        photo=open('images/result.jpg', 'rb')
    )
    # Finish conversation
    await state.finish()

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
