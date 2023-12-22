import requests
from time import sleep

url = "https://getpredict2-r5ss4yyf6q-et.a.run.app"

# Retry up to 3 times with a delay between retries
for _ in range(3):
    try:
        resp = requests.post(
            url, files={'file': open('1188-receipt.jpg', 'rb')}
        )
        # resp = requests.post(
        #     url, files={'file': open('1188-receipt.jpg', 'rb')})
        if resp.status_code == 200:
            data = resp.json()
            print(data)
            break  # Break out of the loop if successful
        else:
            print(f"Request failed with status code {resp.status_code}")
            print("Response content:", resp.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    sleep(1)  # Add a delay between retries
