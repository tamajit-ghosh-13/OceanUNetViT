import requests

payload = {
    "latitude": 15.0,
    "longitude": 85.0,
    "sst": 31.0,
    "sss": 35.0,
    "ssh": 0.5,
    "u_cur": 0.1,
    "v_cur": 0.1,
    "u_wind": 5.0,
    "v_wind": 5.0,
    "doy": 200
}

try:
    res = requests.post("http://localhost:8000/api/predict", json=payload)
    data = res.json()
    print(f"Status Code: {res.status_code}")
    print(f"Keys in response: {list(data.keys())}")
    if "visualizations" in data:
        print(f"Keys in visualizations: {list(data['visualizations'].keys())}")
        shallow = data['visualizations'].get('shallow_profile_image')
        print(f"Shallow image starts with: {shallow[:30] if shallow else 'None'}")
    else:
        print("No visualizations key!")
except Exception as e:
    print(f"Error: {e}")
