import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Crossing':2, 'Finishing':9, 'Free_Kick_Accuracy':6,'Heading_Accuracy':0})

print(r.json())
