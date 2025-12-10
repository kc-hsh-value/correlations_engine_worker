# test_direct_interface.py
import asyncio
import socket
from dotenv import load_dotenv
import os
from telegram import Bot
from telegram.request import HTTPXRequest
import httpx

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def test():
    # Create httpx client that binds to en1 (your Wi-Fi interface)
    transport = httpx.AsyncHTTPTransport(
        local_address="192.168.1.8"  # Your Mac Mini's Wi-Fi IP
    )
    
    request = HTTPXRequest(
        http_version="1.1",
        connect_timeout=30.0,
        read_timeout=30.0,
    )
    
    # Monkey patch to use our transport
    http_client = httpx.AsyncClient(transport=transport, timeout=30.0)
    request._client = http_client
    
    bot = Bot(token=TOKEN, request=request)
    try:
        print("Testing with direct interface binding...")
        me = await bot.get_me()
        print(f"✅ Success! Bot username: @{me.username}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await bot.shutdown()
        await http_client.aclose()

asyncio.run(test())
