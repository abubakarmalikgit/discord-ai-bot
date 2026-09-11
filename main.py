import os
import sys
import gc
import time
import asyncio
import logging
import threading
import unicodedata
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import discord
from discord import app_commands
from discord.ext import commands, tasks

# -------------------------------------------------------------
# 1. Configuration & Logging
# -------------------------------------------------------------
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()
AI_API_BASE_URL = os.getenv("AI_API_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "meta/llama-3.3-70b-instruct").strip()
PORT = int(os.getenv("PORT", 8080))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MalixAris-Core")
logging.getLogger("discord.client").setLevel(logging.ERROR)

# -------------------------------------------------------------
# 2. Render Keep-Alive Web Server
# -------------------------------------------------------------
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"MalixAris AI Gateway Healthy")

    def log_message(self, format, *args):
        return

def run_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthCheckHandler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# -------------------------------------------------------------
# 3. Discord Bot Setup
# -------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = False
intents.presences = False

bot = commands.Bot(command_prefix="!", intents=intents)

conversation_memory = {}
user_cooldowns = {}

SYSTEM_PROMPT = (
    "You are 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI. "
    "Do not identify yourself as Llama, Nemotron, or an assistant made by NVIDIA. "
    "Respond directly, intelligently, and candidly. "
    "Never output safety evaluation tags like 'User Safety: safe'."
)

def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8").lower()

# -------------------------------------------------------------
# 4. Synchronous Threaded NVIDIA Worker (Prevents Event-Loop Stalls)
# -------------------------------------------------------------
def fetch_nvidia_completion(messages: list) -> str:
    endpoint = f"{AI_API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "MalixArisBot/1.0"
    }

    models_to_attempt = [AI_MODEL_NAME, "meta/llama-3.3-70b-instruct", "mistralai/mistral-large-2407"]

    for model in models_to_attempt:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 800
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                return text.replace("User Safety: safe", "").replace("User Safety: unsafe", "").strip()
            logger.warning(f"Model {model} returned status {resp.status_code}: {resp.text}")
        except requests.exceptions.Timeout:
            logger.warning(f"Model {model} timed out after 12 seconds. Attempting next available model...")
        except Exception as e:
            logger.error(f"Request failed for {model}: {e}")

    return "⚠️ *The AI backend took too long to respond. Please try again.*"

async def generate_ai_reply(user_id: int, user_text: str) -> str:
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    conversation_memory[user_id].append({"role": "user", "content": user_text})
    if len(conversation_memory[user_id]) > 4:
        conversation_memory[user_id] = conversation_memory[user_id][-4:]

    full_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_memory[user_id]

    # Offload network request to a separate thread to prevent blocking Discord heartbeat
    reply = await asyncio.to_thread(fetch_nvidia_completion, full_payload)

    conversation_memory[user_id].append({"role": "assistant", "content": reply})
    return reply

# -------------------------------------------------------------
# 5. Slash Commands
# -------------------------------------------------------------
@bot.tree.command(name="chat", description="Chat with 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI")
@app_commands.describe(prompt="Your message")
async def chat_cmd(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    reply = await generate_ai_reply(interaction.user.id, prompt)
    
    if len(reply) <= 1950:
        await interaction.followup.send(reply)
    else:
        for i in range(0, len(reply), 1900):
            if i == 0:
                await interaction.followup.send(reply[i:i+1900])
            else:
                await interaction.channel.send(reply[i:i+1900])

@bot.tree.command(name="reset", description="Clear your chat memory")
async def reset_cmd(interaction: discord.Interaction):
    if interaction.user.id in conversation_memory:
        del conversation_memory[interaction.user.id]
        await interaction.response.send_message("🧠 Memory cleared.", ephemeral=True)
    else:
        await interaction.response.send_message("No active context found.", ephemeral=True)

@bot.tree.command(name="ping", description="Check gateway latency")
async def ping_cmd(interaction: discord.Interaction):
    ping_ms = round(bot.latency * 1000)
    await interaction.response.send_message(f"⚡ 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI Gateway: `{ping_ms}ms` | Operational", ephemeral=True)

# -------------------------------------------------------------
# 6. Event Handling
# -------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions
    clean_channel = normalize_name(getattr(message.channel, "name", ""))
    is_dedicated = "malixaris" in clean_channel

    if not (is_dedicated or is_mentioned or is_dm):
        return

    # Cooldown (2 seconds per user)
    now = time.time()
    if now - user_cooldowns.get(message.author.id, 0) < 2.0:
        return
    user_cooldowns[message.author.id] = now

    clean_text = message.content.replace(f"<@{bot.user.id}>", "").strip()
    if not clean_text:
        await message.channel.send("What would you like to ask?")
        return

    async with message.channel.typing():
        reply = await generate_ai_reply(message.author.id, clean_text)
        for i in range(0, len(reply), 1900):
            await message.channel.send(reply[i:i+1900])

@tasks.loop(minutes=15)
async def cleanup_task():
    if len(conversation_memory) > 20:
        conversation_memory.clear()
    user_cooldowns.clear()
    gc.collect()

@bot.event
async def on_ready():
    logger.info(f"Bot connected as: {bot.user.name} ({bot.user.id})")
    if not cleanup_task.is_running():
        cleanup_task.start()
    try:
        await bot.tree.sync()
        logger.info("Slash command tree synced.")
    except Exception as e:
        logger.error(f"Slash command sync error: {e}")
    await bot.change_presence(activity=discord.Game(name="Chat with 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI"))

# -------------------------------------------------------------
# 7. Start Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        logger.critical("FATAL: DISCORD_BOT_TOKEN is missing.")
        sys.exit(1)
    if not AI_API_KEY:
        logger.critical("FATAL: AI_API_KEY is missing.")
        sys.exit(1)

    bot.run(DISCORD_BOT_TOKEN)
