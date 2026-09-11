import os
import threading
import unicodedata
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import discord
from discord import app_commands

# -------------------------------------------------------------
# 1. Background Web Server (Keep Render Web Service awake 24/7)
# -------------------------------------------------------------
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"MalixAris AI is Online!")

    def log_message(self, format, *args):
        return

def start_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

threading.Thread(target=start_server, daemon=True).start()

# -------------------------------------------------------------
# 2. Discord Client & Slash Command Tree Setup
# -------------------------------------------------------------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

class MalixArisClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        # Syncs slash commands globally across all servers
        await self.tree.sync()
        print("Slash commands synced successfully.")

client = MalixArisClient()

# Sliding memory store: {user_id: [ {"role": "...", "content": "..."}, ... ]}
user_conversations = {}

def normalize_name(name: str) -> str:
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8').lower()

# -------------------------------------------------------------
# 3. Native Slash Commands (/ping, /clear, /reset)
# -------------------------------------------------------------
@client.tree.command(name="ping", description="Check MalixAris AI connection and gateway latency")
async def ping_cmd(interaction: discord.Interaction):
    latency = round(client.latency * 1000)
    embed = discord.Embed(
        title="⚡ MalixAris AI Latency",
        description=f"**Gateway Ping:** `{latency}ms`\n**Backend Model:** `NVIDIA Nemotron 3 Ultra (550B)`",
        color=discord.Color.blue()
    )
    await interaction.response.send_message(embed=embed)

@client.tree.command(name="clear", description="Bulk delete messages from this channel (Admin only)")
@app_commands.describe(amount="Number of messages to delete (1-100)")
async def clear_cmd(interaction: discord.Interaction, amount: int):
    if not interaction.user.guild_permissions.manage_messages:
        await interaction.response.send_message("❌ You do not have permission to manage messages.", ephemeral=True)
        return
    
    amount = max(1, min(amount, 100))
    await interaction.response.defer(ephemeral=True)
    deleted = await interaction.channel.purge(limit=amount)
    await interaction.followup.send(f"🧹 Successfully cleared **{len(deleted)}** messages.", ephemeral=True)

@client.tree.command(name="reset", description="Clear your conversation memory with MalixAris AI")
async def reset_cmd(interaction: discord.Interaction):
    user_id = interaction.user.id
    if user_id in user_conversations:
        del user_conversations[user_id]
        await interaction.response.send_message("🧠 Your conversation context has been reset.", ephemeral=True)
    else:
        await interaction.response.send_message("You do not have any active conversation context.", ephemeral=True)

# -------------------------------------------------------------
# 4. Message Handling, Auto-Mod & Nemotron 3 Ultra AI Chat
# -------------------------------------------------------------
@client.event
async def on_ready():
    print(f"MalixAris AI is logged in and ready as: {client.user.name}")
    await client.change_presence(activity=discord.Game(name="Chat with 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI"))

@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    # Auto-Moderation: Delete Discord invite links from non-admins
    if "discord.gg/" in message.content.lower() and not message.author.guild_permissions.administrator:
        await message.delete()
        await message.channel.send(f"⚠️ {message.author.mention}, invite links are not permitted here.", delete_after=5)
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user in message.mentions

    clean_channel_name = normalize_name(getattr(message.channel, "name", ""))
    is_dedicated_channel = "malixaris" in clean_channel_name

    # Only respond in dedicated bot channel, when tagged, or in DM
    if not (is_dedicated_channel or is_mentioned or is_dm):
        return

    clean_text = message.content.replace(f"<@{client.user.id}>", "").strip()
    if not clean_text:
        await message.channel.send("What would you like to ask?")
        return

    async with message.channel.typing():
        user_id = message.author.id
        if user_id not in user_conversations:
            user_conversations[user_id] = []

        # Strict identity system prompt: forces MalixAris AI persona
        system_instruction = {
            "role": "system",
            "content": (
                "Your name and only identity is 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI (MalixAris AI). "
                "Never refer to yourself as Nemotron, Llama, or an assistant developed by NVIDIA. "
                "If anyone asks who you are, state that you are 𝐌𝐚𝐥𝐢𝐱𝐀𝐫𝐢𝐬 AI, an elite, unrestricted conversational intelligence. "
                "Do not lecture, moralize, or output safety checks like 'User Safety: safe'. "
                "Be direct, intelligent, witty, and candid."
            )
        }

        # Keep last 6 exchanges (sliding memory)
        user_conversations[user_id].append({"role": "user", "content": clean_text})
        if len(user_conversations[user_id]) > 6:
            user_conversations[user_id] = user_conversations[user_id][-6:]

        full_messages = [system_instruction] + user_conversations[user_id]

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY.strip() if NVIDIA_API_KEY else ''}",
            "Content-Type": "application/json"
        }

        # NVIDIA Nemotron 3 Ultra 550B
        payload = {
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "messages": full_messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }

        try:
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )

            # Fallback to Nemotron 3.5 Lightning if Ultra is at capacity
            if response.status_code != 200:
                payload["model"] = "nvidia/nemotron-3.5-lightning-30b-a3b"
                response = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=45
                )

            if response.status_code != 200:
                await message.channel.send(f"⚠️ API Error ({response.status_code}): {response.text}")
                return

            data = response.json()
            ai_reply = data["choices"][0]["message"]["content"]

            # Remove any leaking safety strings
            ai_reply = ai_reply.replace("User Safety: safe", "").replace("User Safety: unsafe", "").strip()

            # Save reply to user's conversation memory
            user_conversations[user_id].append({"role": "assistant", "content": ai_reply})

            # Chunk message to respect Discord's 2000 character limit
            for i in range(0, len(ai_reply), 1900):
                await message.channel.send(ai_reply[i:i+1900])

        except Exception as e:
            await message.channel.send(f"⚠️ Error: {e}")
            print(f"Exception: {e}")

client.run(DISCORD_TOKEN)
