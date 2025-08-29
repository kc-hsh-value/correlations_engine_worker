# bot.py
import discord
import os
import asyncio
from dotenv import load_dotenv

# from main import alpha_cycle_loop # We will refactor main.py into this
from helpers.supabase_db import (
    add_discord_subscription,
    seed_database_if_empty,
)
from helpers.embeddings import generate_embeddings

# --- Telegram Imports ---
from telegram.ext import Application
from helpers.telegram_bot import setup_telegram_handlers # <-- Our new setup function


load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
# DISCORD_BOT_TOKEN_DEV = os.getenv("DISCORD_BOT_TOKEN_DEV")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") 

# --- Bot Class Definition ---
# This structure manages the persistent connection and commands
class PolyMarketBot(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        print("bot is initializing")
        super().__init__(intents=intents)
        # CommandTree is the modern way to handle slash commands
        self.tree = discord.app_commands.CommandTree(self)

    async def setup_hook(self):
        # This copies the global commands to your guild.
        await self.tree.sync()

    async def on_ready(self):
        print("on ready just got called")
        if self.user:
            print(f'Logged in as {self.user} (ID: {self.user.id})')
        else:
            print('Logged in, but self.user is None')
        print('------')
        # self.loop.create_task(alpha_cycle_loop(self))

# --- Bot Instance and Intents ---
intents = discord.Intents.default()
bot = PolyMarketBot(intents=intents)

# --- Slash Command Definition ---
# In bot.py

# --- Slash Command Definition ---
@bot.tree.command(name="setup", description="Set this channel to receive PolyMarket Alpha alerts.")
async def setup(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    if not interaction.guild or not interaction.channel:
        await interaction.followup.send("This command can only be used inside a server channel.")
        return

    if not isinstance(interaction.channel, (discord.TextChannel, discord.Thread)):
        await interaction.followup.send("This command can only be used in a standard text channel or thread.")
        return

    server_id = str(interaction.guild.id)
    channel_id = str(interaction.channel.id)

    try:
        # Optionally pass owner_user_id=None (you can wire this later)
        add_discord_subscription(server_id, channel_id, owner_user_id=None)
        await interaction.followup.send(
            f"✅ **Success!** This channel ({interaction.channel.mention}) will now receive alpha alerts."
        )
    except Exception as e:
        print(f"Error during /setup: {e}")
        await interaction.followup.send("❌ Error! Could not save subscription. Please contact the bot admin.")


# --- THE NEW MAIN EXECUTION BLOCK ---
async def main():
    """Initializes and runs both Discord and Telegram bots concurrently."""
    # Perform initial setup once at the very start
    print("Performing initial setup...")
    seed_database_if_empty(generate_embeddings)
    print("Initial setup complete.")

    # Check for necessary tokens
    if not DISCORD_BOT_TOKEN:
        raise ValueError("DISCORD_BOT_TOKEN_DEV is not set in the environment.")
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in the environment.")

    # 1. Initialize the Telegram Application
    telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # 2. Register the command handlers from our helper file
    setup_telegram_handlers(telegram_app)
    
    # 3. --- Manually start the Telegram bot's components ---
    # This sets them up to run in the background on the existing event loop.
    print("Initializing Telegram bot...")
    await telegram_app.initialize()  # Prepares the application
    
    print("Starting Telegram polling in background...")
    await telegram_app.updater.start_polling()  # Starts fetching updates
    
    print("Starting Telegram application in background...")
    await telegram_app.start()  # Starts processing updates
    
    # 4. Start the Discord bot in the foreground.
    # The `bot.start()` method is a blocking call that keeps the script alive.
    # Since the Telegram bot is already running in the background, we just need
    # this one call to keep the whole process running.
    print("--- Starting Discord bot (will keep all services alive) ---")
    await bot.start(DISCORD_BOT_TOKEN)

    # 5. --- Graceful Shutdown (when bot.start() is cancelled) ---
    print("--- Shutting down services ---")
    await telegram_app.updater.stop()
    await telegram_app.stop()
    await telegram_app.shutdown()


if __name__ == '__main__':
    # Use asyncio.run() to start our main async function.
    # This correctly manages the entire lifecycle.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBots stopped by user.")
