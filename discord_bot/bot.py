"""Bot discord provinding comedy tools"""

import io
import json
import logging
import requests
import numpy as np
import discord
from discord import Intents
from pyAudioAnalysis import audioBasicIO as aIO
from .configuration import TOKEN

GUILD = "Standup comedy tools"
intents = Intents.default()
intents.typing = True
intents.presences = False
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"{client.user.name} has connected to Discord!")


@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(f"Hi {member.name}, welcome to my Discord server!")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.attachments:
        attachment = message.attachments[0]

        if attachment.content_type.startswith("audio"):
            # Télécharger le fichier audio
            audio_bytes = await attachment.read()
            type_ = type(audio_bytes)
            logging.warning(
                f"Audio recieved ! type is {type_}, len is : {len(audio_bytes)}"
            )
            fs, signals = aIO.read_audio_file(io.BytesIO(audio_bytes))
            logging.warning(f"Signal: {signals}")
            max_amplitude = np.max(np.abs(signals))
            normalized_signals = signals / max_amplitude
            print("1", normalized_signals)
            max_value = min(len(normalized_signals), 100000)
            print("2", max_value)
            data = {
                "sampling_freq": fs,
                "signals": list(normalized_signals)[:max_value],
            }
            serialized_data = json.dumps(data)
            logging.warning("Data ready!")  # Sérialisez les données en JSON
            url = "https://laugh-detection-vgc33qfada-uc.a.run.app/kpis_2"  # L'URL correspondant à votre route Flask
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, data=serialized_data, headers=headers)
            logging.warning(f"Post on {url}")
            await message.channel.send(str(response.json()))


client.run(TOKEN)
