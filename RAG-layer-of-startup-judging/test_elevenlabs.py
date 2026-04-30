import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ELEVENLAB_API_KEY")
print(f"Testing API Key: {api_key[:5]}...{api_key[-5:] if api_key else ''}")

if not api_key:
    print("ERROR: ELEVENLAB_API_KEY not found in .env")
    exit(1)

url = "https://api.elevenlabs.io/v1/voices"
headers = {
    "Accept": "application/json",
    "xi-api-key": api_key
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    voices = response.json().get("voices", [])
    print(f"SUCCESS: Found {len(voices)} voices.")
    for v in voices[:5]:
        print(f" - {v.get('name')} ({v.get('voice_id')})")
else:
    print(f"FAILED: {response.status_code} - {response.text}")
