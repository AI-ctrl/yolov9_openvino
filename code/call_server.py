import requests

# URL of the Flask endpoint
url = 'http://172.17.0.2:8020/predict'

# Data to be sent in JSON format
data = {
    'image_path': '/home/ai-ctrl/matriceAI/data/val/images/011805.jpg',
}

# Making a POST request with JSON data
response = requests.post(url, json=data)

# Printing the response
print('Status Code:', response.status_code)
print('Response Body:', response.json())
