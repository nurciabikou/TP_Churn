import requests
url="http://127.0.0.1:5000/predict"
data = {
    "Age": 35,
    "Account_Manager": 1,  
    "Years": 5,
    "Num_Sites": 3
}
response = requests.post(url, data=data)
print(response.json())