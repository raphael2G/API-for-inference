
import requests

output = requests.get('http://127.0.0.1:8000/classification/image.jpg')
print(output)
print(output.json())
