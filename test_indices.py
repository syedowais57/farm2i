import requests
import json

url = "http://localhost:8000/api/v1/calculate-indices"
data = {
    "coordinates": [
        {"lat": 34.02555224449088, "lng": 74.81327442321802},
        {"lat": 34.02566191157703, "lng": 74.81373755130355},
        {"lat": 34.02541293744765, "lng": 74.8139038482608},
        {"lat": 34.02524399030139, "lng": 74.81332091484}
    ],
    "start_date": "2025-07-01",
    "end_date": "2025-07-10",
    "field_id": "testing_scl_fix",
    "indices": ["NDVI", "OC"],
    "max_cloud_cover": 50
}

try:
    print("Testing API with new indices: RECI, PSRI, MCARI...")
    response = requests.post(url, json=data, timeout=300)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nIndices in response:")
        for idx_name, values in result.get('indices', {}).items():
            if values:
                print(f"- {idx_name}: Found data")
            else:
                print(f"- {idx_name}: No data")
                
        print("\nPNG paths in response:")
        for idx_name, paths in result.get('output_paths', {}).items():
            if paths:
                print(f"- {idx_name}: {paths}")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Test failed: {e}")
