import requests

url = "https://api.elevenlabs.io/v1/voices"

headers = {
  "Accept": "application/json",
  "xi-api-key": "270f30c2d01d4529066bb6b97802b2df"
}

response = requests.get(url, headers=headers)

print(response.text)
