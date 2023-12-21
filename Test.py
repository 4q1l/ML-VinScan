import requests

resp = requests.post("http://localhost:5000/", files={'file': open('1188-receipt.jpg', 'rb')})

print(resp.json())