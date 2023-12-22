import requests

# Gantilah API_KEY dengan kunci API yang telah Anda dapatkan dari konsol GCP
API_KEY = "AIzaSyCaFhAur1bZI6aw5bLQ-fdeJ9PgTvS4IAg"

try:
    url = "https://getprediction1-r5ss4yyf6q-as.a.run.app"
    file_path = '1188-receipt.jpg'

    with open(file_path, 'rb') as file:
        files = {'file': (file_path, file)}

        # Gunakan parameter query untuk menyertakan kunci API
        params = {'key': API_KEY}

        resp = requests.post(url, files=files, params=params)
        resp.raise_for_status()

        print(resp.json())
except requests.exceptions.HTTPError as errh:
    print("HTTP Error:", errh)
except requests.exceptions.RequestException as err:
    print("Error:", err)
except requests.exceptions.JSONDecodeError as errj:
    print("JSON Decode Error:", errj)
