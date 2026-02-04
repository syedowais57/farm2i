import ee
import json
import os

SERVICE_ACCOUNT_KEY = 'named-magnet-486409-f1-99092699381d.json'
GEE_PROJECT = 'named-magnet-486409-f1'

def test_init():
    try:
        if not os.path.exists(SERVICE_ACCOUNT_KEY):
            print(f"File not found: {SERVICE_ACCOUNT_KEY}")
            return
            
        with open(SERVICE_ACCOUNT_KEY, 'r') as f:
            key_data = json.load(f)
        
        service_account_email = key_data.get('client_email')
        print(f"Service Account Email: {service_account_email}")
        
        credentials = ee.ServiceAccountCredentials(
            email=service_account_email,
            key_file=SERVICE_ACCOUNT_KEY
        )
        
        print(f"Initializing for project: {GEE_PROJECT}...")
        ee.Initialize(credentials, project=GEE_PROJECT)
        print("✅ GEE initialized successfully!")
        
        # Try a small test query to be sure
        test_geom = ee.Geometry.Point([74.8133, 34.0255])
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(test_geom).limit(1)
        count = collection.size().getInfo()
        print(f"✅ Connection test: Found {count} images.")
        
    except Exception as e:
        import traceback
        print(f"❌ Initialization failed: {str(e)}")
        print("\n--- FULL STACKTRACE ---")
        traceback.print_exc()
        print("--- END STACKTRACE ---")
        
        if "not been used in project" in str(e) or "disabled" in str(e).lower():
            print("\nPROBABLE CAUSE: API enablement issue.")
            print(f"1. Visit: https://console.developers.google.com/apis/api/earthengine.googleapis.com/overview?project={GEE_PROJECT}")
            print("2. Ensure the status is 'API Enabled'.")
            print(f"3. Ensure the Service Account '{service_account_email}' has 'Service Usage Consumer' role in IAM for project '{GEE_PROJECT}'.")
            print(f"4. Ensure '{service_account_email}' is registered at https://signup.earthengine.google.com/.")

if __name__ == "__main__":
    test_init()
