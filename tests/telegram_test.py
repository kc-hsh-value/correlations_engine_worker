# test_telegram.py
import asyncio
import os
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def test():
    bot = Bot(token=TOKEN)
    try:
        print(f"Testing Telegram bot... {TOKEN}",)
        me = await bot.get_me()
        print(f"Success! Bot username: @{me.username}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await bot.shutdown()

asyncio.run(test())