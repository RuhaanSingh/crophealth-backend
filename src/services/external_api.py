import requests
import os

class ExternalAPIService:
    """Service for fetching external API data (weather and soil)."""
    
    def __init__(self):
        # OpenWeatherMap API key (should be set as environment variable)
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY', 'your_api_key_here')
        self.openweather_base_url = 'https://api.openweathermap.org/data/2.5'
        
        # SoilGrids API (no authentication required)
        self.soilgrids_base_url = 'https://rest.isric.org/soilgrids/v2.0'
    
    def get_weather_data(self, latitude, longitude):
        """
        Fetch current weather data from OpenWeatherMap API.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            dict: Weather data or None if error
        """
        try:
            url = f"{self.openweather_base_url}/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.openweather_api_key,
                'units': 'metric'  # Use Celsius
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant weather features
            weather_features = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'wind_direction': data.get('wind', {}).get('deg', 0),
                'cloudiness': data['clouds']['all'],
                'visibility': data.get('visibility', 10000),  # Default 10km
                'weather_description': data['weather'][0]['description'],
                'precipitation_1h': data.get('rain', {}).get('1h', 0),  # Rain in last 1h
                'snow_1h': data.get('snow', {}).get('1h', 0),  # Snow in last 1h
                'timestamp': data['dt']
            }
            
            return weather_features
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except KeyError as e:
            print(f"Error parsing weather data: {e}")
            return None
    
    def get_soil_data(self, latitude, longitude):
        """
        Fetch soil data from SoilGrids API.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            dict: Soil data or None if error
        """
        try:
            url = f"{self.soilgrids_base_url}/properties/query"
            params = {
                'lon': longitude,
                'lat': latitude,
                'property': 'phh2o,oc,nitrogen,cec,sand,clay,silt',  # pH, organic carbon, nitrogen, etc.
                'depth': '0-30cm',  # Topsoil layer
                'value': 'mean'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant soil features
            soil_features = {}
            
            if 'properties' in data:
                properties = data['properties']
                
                # Extract values for 0-30cm depth
                for prop_name, prop_data in properties.items():
                    if 'depths' in prop_data and len(prop_data['depths']) > 0:
                        # Get the first depth layer (0-30cm)
                        depth_data = prop_data['depths'][0]
                        if 'values' in depth_data and len(depth_data['values']) > 0:
                            soil_features[prop_name] = depth_data['values'][0]
                
                # Rename properties to more readable names
                soil_features_readable = {
                    'ph_water': soil_features.get('phh2o', 0) / 10.0,  # Convert from pH*10 to pH
                    'organic_carbon': soil_features.get('oc', 0) / 10.0,  # Convert from g/kg*10 to g/kg
                    'nitrogen': soil_features.get('nitrogen', 0) / 100.0,  # Convert from cg/kg to g/kg
                    'cation_exchange_capacity': soil_features.get('cec', 0) / 10.0,  # Convert from cmol/kg*10
                    'sand_content': soil_features.get('sand', 0) / 10.0,  # Convert from g/kg*10 to g/kg
                    'clay_content': soil_features.get('clay', 0) / 10.0,  # Convert from g/kg*10 to g/kg
                    'silt_content': soil_features.get('silt', 0) / 10.0,  # Convert from g/kg*10 to g/kg
                }
                
                return soil_features_readable
            
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching soil data: {e}")
            return None
        except KeyError as e:
            print(f"Error parsing soil data: {e}")
            return None

