import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'shop_id':5, 'item_id':1, 'month':11, 'year':2015})

print(r.json())