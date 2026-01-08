"""
Sentinel Hub Python Service
Equivalent to your TypeScript SentinelServiceV2
"""
import requests
import os
from datetime import datetime
from typing import Optional
import json

# Evalscripts
NDVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  
  // Color gradient: red (low) -> yellow -> green (high)
  if (ndvi < 0) return [0.5, 0, 0, sample.dataMask];
  if (ndvi < 0.2) return [0.8, 0.2, 0, sample.dataMask];
  if (ndvi < 0.4) return [1, 0.8, 0, sample.dataMask];
  if (ndvi < 0.6) return [0.5, 0.8, 0.2, sample.dataMask];
  return [0, 0.5, 0, sample.dataMask];
}
"""

NDMI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B08", "B11", "dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(sample) {
  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11);
  
  // Color gradient: brown (dry) -> yellow -> green (moist)
  if (ndmi < -0.2) return [0.5, 0.3, 0.1, sample.dataMask];
  if (ndmi < 0) return [0.8, 0.6, 0.2, sample.dataMask];
  if (ndmi < 0.2) return [1, 0.9, 0.4, sample.dataMask];
  if (ndmi < 0.4) return [0.5, 0.8, 0.3, sample.dataMask];
  return [0, 0.6, 0.3, sample.dataMask];
}
"""

NDWI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B03", "B08", "dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(sample) {
  let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
  
  // Color gradient: brown (land) -> cyan -> blue (water)
  if (ndwi < 0) return [0.5, 0.3, 0.1, sample.dataMask];
  if (ndwi < 0.2) return [0.8, 0.9, 0.5, sample.dataMask];
  if (ndwi < 0.4) return [0, 0.8, 0.8, sample.dataMask];
  if (ndwi < 0.6) return [0, 0.5, 0.9, sample.dataMask];
  return [0, 0, 0.7, sample.dataMask];
}
"""

MSAVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(sample) {
  let msavi = (2 * sample.B08 + 1 - Math.sqrt(Math.pow(2 * sample.B08 + 1, 2) - 8 * (sample.B08 - sample.B04))) / 2;
  
  // Color gradient similar to NDVI
  if (msavi < 0) return [0.5, 0, 0, sample.dataMask];
  if (msavi < 0.2) return [0.8, 0.2, 0, sample.dataMask];
  if (msavi < 0.4) return [1, 0.8, 0, sample.dataMask];
  if (msavi < 0.6) return [0.5, 0.8, 0.2, sample.dataMask];
  return [0, 0.5, 0, sample.dataMask];
}
"""

class SentinelHubService:
    """Python equivalent of SentinelServiceV2"""
    
    SENTINEL_OAUTH_ENDPOINT = "https://services.sentinel-hub.com/oauth/token"
    SENTINEL_HUB_PROCESS_ENDPOINT = "https://services.sentinel-hub.com/api/v1/process"
    CREODIAS_PROCESS_ENDPOINT = "https://creodias.sentinel-hub.com/api/v1/process"
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id or os.environ.get('SENTINEL_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('SENTINEL_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET must be set")
    
    def generate_access_token(self) -> str:
        """Generate OAuth access token for Sentinel Hub"""
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }
        
        response = requests.post(
            self.SENTINEL_OAUTH_ENDPOINT,
            data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        
        if response.status_code != 200:
            raise Exception(f"Error generating access token: {response.text}")
        
        return response.json()['access_token']
    
    def process_image_request(
        self,
        polygon: dict,
        evalscript: str,
        data_type: str = "S2L2A",
        start_date: datetime = None,
        end_date: datetime = None,
        width: int = 512,
        height: int = 512,
        use_creodias: bool = False
    ) -> bytes:
        """
        Process image request to Sentinel Hub
        
        Args:
            polygon: GeoJSON Polygon geometry
            evalscript: Sentinel Hub evalscript for processing
            data_type: S2L2A, S2L1C, or S3SLSTR
            start_date: Start date for imagery
            end_date: End date for imagery
            width: Output image width
            height: Output image height
            use_creodias: Use CREODIAS endpoint
            
        Returns:
            Image bytes (PNG)
        """
        access_token = self.generate_access_token()
        endpoint = self.CREODIAS_PROCESS_ENDPOINT if use_creodias else self.SENTINEL_HUB_PROCESS_ENDPOINT
        
        # Default dates (last 30 days)
        if not start_date:
            start_date = datetime.now().replace(day=1)
        if not end_date:
            end_date = datetime.now()
        
        # Map data type
        data_type_map = {
            "S2L2A": "sentinel-2-l2a",
            "S2L1C": "sentinel-2-l1c",
            "S3SLSTR": "sentinel-3-slstr"
        }
        
        request_body = {
            "input": {
                "bounds": {"geometry": polygon},
                "data": [{
                    "type": data_type_map.get(data_type, data_type),
                    "dataFilter": {
                        "timeRange": {
                            "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                            "to": end_date.strftime("%Y-%m-%dT23:59:59Z")
                        },
                        "maxCloudCoverage": 20
                    }
                }]
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [
                    {"identifier": "default", "format": {"type": "image/png"}}
                ]
            },
            "evalscript": evalscript
        }
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "image/png"
        }
        
        response = requests.post(endpoint, json=request_body, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Error processing image: {response.text}")
        
        return response.content
    
    def get_ndvi_image(self, polygon: dict, start_date: datetime = None, end_date: datetime = None) -> bytes:
        """Get NDVI image for polygon"""
        return self.process_image_request(
            polygon=polygon,
            evalscript=NDVI_EVALSCRIPT,
            data_type="S2L2A",
            start_date=start_date,
            end_date=end_date
        )
    
    def get_ndmi_image(self, polygon: dict, start_date: datetime = None, end_date: datetime = None) -> bytes:
        """Get NDMI image for polygon"""
        return self.process_image_request(
            polygon=polygon,
            evalscript=NDMI_EVALSCRIPT,
            data_type="S2L2A",
            start_date=start_date,
            end_date=end_date
        )
    
    def get_ndwi_image(self, polygon: dict, start_date: datetime = None, end_date: datetime = None) -> bytes:
        """Get NDWI image for polygon"""
        return self.process_image_request(
            polygon=polygon,
            evalscript=NDWI_EVALSCRIPT,
            data_type="S2L2A",
            start_date=start_date,
            end_date=end_date
        )
    
    def get_msavi_image(self, polygon: dict, start_date: datetime = None, end_date: datetime = None) -> bytes:
        """Get MSAVI image for polygon"""
        return self.process_image_request(
            polygon=polygon,
            evalscript=MSAVI_EVALSCRIPT,
            data_type="S2L2A",
            start_date=start_date,
            end_date=end_date
        )
    
    def save_all_indices(
        self,
        polygon: dict,
        output_dir: str = "./output",
        start_date: datetime = None,
        end_date: datetime = None,
        polygon_id: str = "1"
    ):
        """
        Save all vegetation indices for a polygon
        
        Args:
            polygon: GeoJSON Polygon geometry
            output_dir: Directory to save images
            start_date: Start date
            end_date: End date
            polygon_id: ID for naming files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        indices = {
            'NDVI': self.get_ndvi_image,
            'NDMI': self.get_ndmi_image,
            'NDWI': self.get_ndwi_image,
            'MSAVI': self.get_msavi_image,
        }
        
        date_str = (start_date or datetime.now()).strftime("%Y-%m-%d")
        
        for index_name, func in indices.items():
            print(f"Processing {index_name}...")
            try:
                image_bytes = func(polygon, start_date, end_date)
                
                filename = f"{polygon_id}_{date_str}_{index_name}.png"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                print(f"  Saved: {filepath}")
            except Exception as e:
                print(f"  Error processing {index_name}: {e}")


# Example usage
if __name__ == "__main__":
    # Set your credentials
    # os.environ['SENTINEL_CLIENT_ID'] = 'your_client_id'
    # os.environ['SENTINEL_CLIENT_SECRET'] = 'your_client_secret'
    
    # Example polygon (GeoJSON format)
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [74.81199703093202, 34.02529834654321],
            [74.81198019953428, 34.02551805408936],
            [74.81337720549703, 34.02597839186359],
            [74.81397051224653, 34.0260307028166],
            [74.81363388430364, 34.02516931168813],
            [74.81199604386288, 34.025296237596336],
            [74.81199703093202, 34.02529834654321]  # Close the polygon
        ]]
    }
    
    try:
        service = SentinelHubService()
        
        service.save_all_indices(
            polygon=polygon,
            output_dir="./sentinel_hub_output",
            start_date=datetime(2025, 5, 1),
            end_date=datetime(2025, 5, 10),
            polygon_id="test_polygon"
        )
        
        print("\n✅ All indices saved successfully!")
        
    except ValueError as e:
        print(f"⚠️ {e}")
        print("\nTo use Sentinel Hub, set these environment variables:")
        print("  export SENTINEL_CLIENT_ID='your_client_id'")
        print("  export SENTINEL_CLIENT_SECRET='your_client_secret'")
        print("\nGet credentials at: https://www.sentinel-hub.com/")
