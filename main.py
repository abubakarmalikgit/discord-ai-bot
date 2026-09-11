import os
import sys
import gc
import time
import asyncio
import logging
import threading
import unicodedata
from datetime import date
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Set

import requests
import discord
from discord import app_commands
from discord.ext import commands, tasks

# -------------------------------------------------------------
# 1. Environment & Logging
# -------------------------------------------------------------
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()
AI_API_BASE_URL = os.getenv("AI_API_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "meta/llama-3.1-8b-instruct").strip()
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "6"))
FREE_TIER_DAILY_LIMIT = int(os.getenv("FREE_TIER_DAILY_LIMIT", "50"))
PORT = int(os.getenv("PORT", 8080))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
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
# 3. Discord Bot Setup & Memory Stores
# -------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = False
intents.presences = False

bot = commands.Bot(command_prefix="!", intents=intents)

conversation_memory: Dict[int, List[Dict[str, str]]] = {}
user_cooldowns: Dict[int, float] = {}
guild_personas: Dict[int, str] = {}
daily_usage: Dict[str, int] = {}
premium_users: Set[int] = set()

stats_tracker = {
    "total_requests": 0,
    "successful_completions": 0,
    "failed_requests": 0,
    "start_time": time.time()
}

SYSTEM_PROMPT = (
    "Your name and identity is strictly 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI. "
    "Never refer to yourself as Llama, Nemotron, or an assistant made by NVIDIA. "
    "Respond directly, intelligently, candidly, and concisely. "
    "Never output safety evaluation tags like 'User Safety: safe'."
)

def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8").lower()

def sanitize_input(text: str) -> str:
    disallowed = ["system:", "[system]", "### system", "<|im_start|>system"]
    cleaned = text
    for prefix in disallowed:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned

# -------------------------------------------------------------
# 4. NVIDIA Fast AI Engine with Strict Error Reporting
# -------------------------------------------------------------
def fetch_nvidia_completion(messages: list) -> str:
    endpoint = f"{AI_API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "MalixArisBot/1.0"
    }

    # Primary lightweight fast model, with a slightly heavier fallback
    candidate_models = [
        AI_MODEL_NAME,
        "meta/llama-3.1-70b-instruct"
    ]

    last_error = ""
    for model in candidate_models:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 1024
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                return text.replace("User Safety: safe", "").replace("User Safety: unsafe", "").strip()
            
            # Capture exact API rejection reason
            last_error = f"API rejected {model} with Status {resp.status_code}: {resp.text}"
            logger.warning(last_error)
        except requests.exceptions.Timeout:
            last_error = f"Model {model} timed out after 15 seconds."
            logger.warning(last_error)
        except Exception as e:
            last_error = f"Model {model} exception: {e}"
            logger.error(last_error)

    # If it fails, raise the exact error so it gets printed in Discord
    raise RuntimeError(last_error)

async def execute_chat_pipeline(user: discord.User, channel: discord.TextChannel, prompt: str) -> str:
    user_id = user.id
    guild_id = channel.guild.id if channel.guild else None
    today_key = f"{user_id}:{date.today().isoformat()}"

    # Anti-spam cooldown (1.5s)
    now = time.time()
    if now - user_cooldowns.get(user_id, 0) < 1.5:
        return "⏳ *Slow down a second.*"
    user_cooldowns[user_id] = now

    # Daily quota check
    is_premium = user_id in premium_users
    usage = daily_usage.get(today_key, 0)
    if not is_premium and usage >= FREE_TIER_DAILY_LIMIT:
        return f"⚡ **Daily Limit Reached:** Quota of **{FREE_TIER_DAILY_LIMIT} messages/day** exceeded."

    daily_usage[today_key] = usage + 1
    stats_tracker["total_requests"] += 1

    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    clean_prompt = sanitize_input(prompt)
    conversation_memory[user_id].append({"role": "user", "content": clean_prompt})

    if len(conversation_memory[user_id]) > MAX_CONTEXT_MESSAGES:
        conversation_memory[user_id] = conversation_memory[user_id][-MAX_CONTEXT_MESSAGES:]

    system_content = guild_personas.get(guild_id, SYSTEM_PROMPT) if guild_id else SYSTEM_PROMPT
    full_messages = [{"role": "system", "content": system_content}] + conversation_memory[user_id]

    try:
        reply = await asyncio.to_thread(fetch_nvidia_completion, full_messages)
        conversation_memory[user_id].append({"role": "assistant", "content": reply})
        stats_tracker["successful_completions"] += 1
        return reply
    except Exception as e:
        stats_tracker["failed_requests"] += 1
        # This will now print the exact NVIDIA error straight to your Discord channel
        return f"⚠️ **NVIDIA Backend Error:** `{e}`"

# -------------------------------------------------------------
# 5. Slash Commands
# -------------------------------------------------------------
@bot.tree.command(name="chat", description="Chat with 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI")
@app_commands.describe(prompt="Your message")
async def chat_cmd(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    reply = await execute_chat_pipeline(interaction.user, interaction.channel, prompt)
    
    if len(reply) <= 1950:
        await interaction.followup.send(reply)
    else:
        for i in range(0, len(reply), 1900):
            if i == 0:
                await interaction.followup.send(reply[i:i+1900])
            else:
                await interaction.channel.send(reply[i:i+1900])

@bot.tree.command(name="ask", description="Ask a fast single question")
@app_commands.describe(question="Your question")
async def ask_cmd(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    reply = await execute_chat_pipeline(interaction.user, interaction.channel, question)
    
    if len(reply) <= 1950:
        await interaction.followup.send(reply)
    else:
        for i in range(0, len(reply), 1900):
            if i == 0:
                await interaction.followup.send(reply[i:i+1900])
            else:
                await interaction.channel.send(reply[i:i+1900])

@bot.tree.command(name="reset", description="Clear your conversation memory")
async def reset_cmd(interaction: discord.Interaction):
    if interaction.user.id in conversation_memory:
        del conversation_memory[interaction.user.id]
        await interaction.response.send_message("🧠 Memory cleared.", ephemeral=True)
    else:
        await interaction.response.send_message("No active context found.", ephemeral=True)

@bot.tree.command(name="persona", description="Set a custom AI system prompt (Admin only)")
@app_commands.describe(prompt="The new persona instructions")
async def persona_cmd(interaction: discord.Interaction, prompt: str):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("❌ Admin privileges required.", ephemeral=True)
        return
    guild_personas[interaction.guild_id] = prompt
    await interaction.response.send_message(f"✅ **Server persona updated:**\n`{prompt[:200]}...`")

@bot.tree.command(name="stats", description="View bot runtime analytics")
async def stats_cmd(interaction: discord.Interaction):
    uptime_sec = int(time.time() - stats_tracker["start_time"])
    hours, rem = divmod(uptime_sec, 3600)
    mins, secs = divmod(rem, 60)

    embed = discord.Embed(title="📊 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI Stats", color=discord.Color.blue())
    embed.add_field(name="Gateway Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="Uptime", value=f"{hours}h {mins}m {secs}s", inline=True)
    embed.add_field(name="Total Requests", value=str(stats_tracker["total_requests"]), inline=True)
    embed.add_field(name="Completed", value=str(stats_tracker["successful_completions"]), inline=True)
    embed.add_field(name="Active Engine", value=f"`{AI_MODEL_NAME}`", inline=False)
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="ping", description="Check gateway latency")
async def ping_cmd(interaction: discord.Interaction):
    ping_ms = round(bot.latency * 1000)
    await interaction.response.send_message(f"⚡ 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI Gateway: `{ping_ms}ms` | Operational", ephemeral=True)

# -------------------------------------------------------------
# 6. Event Handling & Auto-Chat
# -------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return

    if "discord.gg/" in message.content.lower() and not message.author.guild_permissions.administrator:
        await message.delete()
        await message.channel.send(f"⚠️ {message.author.mention}, invite links are prohibited.", delete_after=4)
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions
    clean_channel = normalize_name(getattr(message.channel, "name", ""))
    is_dedicated = "malixaris" in clean_channel

    if not (is_dedicated or is_mentioned or is_dm):
        return

    clean_text = message.content.replace(f"<@{bot.user.id}>", "").strip()
    if not clean_text:
        await message.channel.send("Hello! What can I help you with today?")
        return

    async with message.channel.typing():
        reply = await execute_chat_pipeline(message.author, message.channel, clean_text)
        for i in range(0, len(reply), 1900):
            await message.channel.send(reply[i:i+1900])

@tasks.loop(minutes=20)
async def cleanup_task():
    if len(conversation_memory) > 30:
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
        logger.error(f"Command sync error: {e}")
    await bot.change_presence(activity=discord.Game(name="Chat with 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI"))

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN or not AI_API_KEY:
        logger.critical("FATAL: Missing tokens in environment variables.")
        sys.exit(1)
    bot.run(DISCORD_BOT_TOKEN)
