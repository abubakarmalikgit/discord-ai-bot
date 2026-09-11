import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import discord

# 1. Background web server so Render's Web Service stays happy
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Discord Bot is Online!")

    # Mute request logs to keep Render console clean
    def log_message(self, format, *args):
        return

def start_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

threading.Thread(target=start_server, daemon=True).start()

# 2. Discord Bot Configuration
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")

intents = discord.Intents.default()
intents.message_content = True  # Required to read channel text
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user.name}")

@client.event
async def on_message(message):
    # Prevent bot from responding to itself
    if message.author == client.user:
        return

    # Trigger typing indicator while calling the AI
    async with message.channel.typing():
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": [{"role": "user", "content": message.content}]
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            ai_reply = response.json()["choices"][0]["message"]["content"]
            
            # Discord has a 2000 character limit per message
            if len(ai_reply) > 2000:
                ai_reply = ai_reply[:1990] + "..."
            
            await message.channel.send(ai_reply)
        except Exception as e:
            await message.channel.send("Sorry, I had an issue contacting the AI.")
            print(f"Error: {e}")

client.run(DISCORD_TOKEN)