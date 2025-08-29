# helpers/telegram_bot.py

import os
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode

# ⬇️ switched to Supabase helper
from .supabase_db import add_telegram_subscription

# --- Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and instructions; supports message/edited/channel contexts."""
    msg = update.message or update.edited_message or update.channel_post
    if not msg:
        return

    welcome_text = (
        "👋 *Welcome to the PolyMarket Alpha Bot\\!* \n\n"
        "I find breaking news tweets and correlate them with relevant PolyMarket markets\\. \n\n"
        "*To get started, set me up in a channel:* \n"
        "1\\. Add this bot to your Telegram channel or group\\. \n"
        "2\\. Promote it to **Admin** so it can post\\. \n"
        "3\\. Type `/setup` in that channel\\. \n\n"
        "That’s it — you’ll start receiving alerts as they’re found\\."
    )
    await msg.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN_V2)


async def setup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Saves the chat id to receive alerts."""
    msg = update.message or update.edited_message or update.channel_post
    if not msg:
        return

    chat = msg.chat
    if chat.type in ("group", "supergroup", "channel"):
        chat_id = str(chat.id)
        try:
            # ⬇️ write to Supabase
            add_telegram_subscription(chat_id)
            await msg.reply_text("✅ *Success\\!* This chat will now receive alpha alerts\\.", parse_mode=ParseMode.MARKDOWN_V2)
            print(f"[telegram] subscribed chat_id={chat_id}")
        except Exception as e:
            print(f"[telegram] DB error while subscribing {chat_id}: {e}")
            await msg.reply_text("❌ *Error\\!* Could not save subscription\\. Please contact the bot admin\\.", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await msg.reply_text("This command must be used inside a group, supergroup, or channel.")


def setup_telegram_handlers(application: Application):
    """Register command handlers."""
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("setup", setup_command))
    print("[telegram] handlers registered")