import requests
from time import sleep

try:
    resp = requests.post("http://localhost:8080/index4", files={'file': open('MutasiBCA_291123-061223.pdf', 'rb')})
    resp.raise_for_status()  # Raises an HTTPError for bad responses
    print(resp.json())
except requests.exceptions.HTTPError as errh:
    print("HTTP Error:", errh)
except requests.exceptions.RequestException as err:
    print("Error:", err)
except requests.exceptions.JSONDecodeError as errj:
    print("JSON Decode Error:", errj)
