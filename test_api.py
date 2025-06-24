#!/usr/bin/env python3
"""
Comprehensive test suite for CropHealth AI+ Backend API
"""

import sys
import os
import requests
import json
import time
from io import BytesIO
from PIL import Image

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Test configuration
API_BASE_URL = 'http://localhost:5000/api'
TEST_USER = {
    'name': 'Test Farmer',
    'email': 'test@crophealth.com',
    'password': 'testpassword123'
}

class APITester:
    def __init__(self):
        self.session = requests.Session()
        self.token = None
        self.user_id = None
        self.field_id = None
        self.image_id = None
        
    def log(self, message, status="INFO"):
        print(f"[{status}] {message}")
    
    def test_health_check(self):
        """Test the health check endpoint."""
        try:
            response = self.session.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                self.log("✓ Health check passed")
                return True
            else:
                self.log(f"✗ Health check failed: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Health check error: {e}", "ERROR")
            return False
    
    def test_user_registration(self):
        """Test user registration."""
        try:
            response = self.session.post(f"{API_BASE_URL}/register", json=TEST_USER)
            if response.status_code == 201:
                self.log("✓ User registration passed")
                return True
            elif response.status_code == 409:
                self.log("✓ User already exists (expected for repeated tests)")
                return True
            else:
                self.log(f"✗ User registration failed: {response.status_code} - {response.text}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ User registration error: {e}", "ERROR")
            return False
    
    def test_user_login(self):
        """Test user login and token retrieval."""
        try:
            login_data = {
                'email': TEST_USER['email'],
                'password': TEST_USER['password']
            }
            response = self.session.post(f"{API_BASE_URL}/login", json=login_data)
            if response.status_code == 200:
                data = response.json()
                self.token = data.get('access_token')
                if self.token:
                    self.session.headers.update({'Authorization': f'Bearer {self.token}'})
                    self.log("✓ User login passed")
                    return True
                else:
                    self.log("✗ No access token received", "ERROR")
                    return False
            else:
                self.log(f"✗ User login failed: {response.status_code} - {response.text}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ User login error: {e}", "ERROR")
            return False
    
    def test_get_profile(self):
        """Test getting user profile."""
        try:
            response = self.session.get(f"{API_BASE_URL}/profile")
            if response.status_code == 200:
                data = response.json()
                self.user_id = data.get('id')
                self.log(f"✓ Get profile passed - User ID: {self.user_id}")
                return True
            else:
                self.log(f"✗ Get profile failed: {response.status_code} - {response.text}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Get profile error: {e}", "ERROR")
            return False
    
    def test_create_field(self):
        """Test creating a field."""
        try:
            field_data = {
                'name': 'Test Field',
                'crop_type': 'Corn',
                'polygon_geometry': json.dumps({
                    "type": "Polygon",
                    "coordinates": [[
                        [-74.0059, 40.7128],
                        [-74.0059, 40.7138],
                        [-74.0049, 40.7138],
                        [-74.0049, 40.7128],
                        [-74.0059, 40.7128]
                    ]]
                })
            }
            response = self.session.post(f"{API_BASE_URL}/fields", json=field_data)
            if response.status_code == 201:
                data = response.json()
                self.field_id = data.get('id')
                self.log(f"✓ Create field passed - Field ID: {self.field_id}")
                return True
            else:
                self.log(f"✗ Create field failed: {response.status_code} - {response.text}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Create field error: {e}", "ERROR")
            return False
    
    def test_get_fields(self):
        """Test getting user fields."""
        try:
            response = self.session.get(f"{API_BASE_URL}/fields")
            if response.status_code == 200:
                data = response.json()
                self.log(f"✓ Get fields passed - Found {len(data)} fields")
                return True
            else:
                self.log(f"✗ Get fields failed: {response.status_code} - {response.text}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Get fields error: {e}", "ERROR")
            return False
    
    def create_test_image(self):
        """Create a test image for upload."""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='green')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_image_upload(self):
        """Test image upload and prediction."""
        if not self.field_id:
            self.log("✗ No field ID available for image upload", "ERROR")
            return False
        
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Prepare form data
            files = {
                'image_file': ('test_crop.jpg', test_image, 'image/jpeg')
            }
            data = {
                'latitude': 40.7128,
                'longitude': -74.0059,
                'field_id': self.field_id
            }
            
            response = self.session.post(f"{API_BASE_URL}/upload", files=files, data=data)
            if response.status_code == 201:
                result = response.json()
                self.image_id = result.get('image_id')
                self.log(f"✓ Image upload passed - Image ID: {self.image_id}")
                self.log(f"  Stress scores: {result.get('stress_scores', {})}")
                self.log(f"  Recommendations: {len(result.get('recommendations', []))} generated")
                return True
            else:
                self.log(f"✗ Image upload failed: {response.status_code} - {response.text}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Image upload error: {e}", "ERROR")
            return False
    
    def test_get_predictions(self):
        """Test getting predictions for an image."""
        if not self.image_id:
            self.log("✗ No image ID available for predictions test", "ERROR")
            return False
        
        try:
            response = self.session.get(f"{API_BASE_URL}/images/{self.image_id}/predictions")
            if response.status_code == 200:
                data = response.json()
                self.log(f"✓ Get predictions passed - Found {len(data)} predictions")
                return True
            else:
                self.log(f"✗ Get predictions failed: {response.status_code} - {response.text}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Get predictions error: {e}", "ERROR")
            return False
    
    def test_get_field_stats(self):
        """Test getting field statistics."""
        if not self.field_id:
            self.log("✗ No field ID available for stats test", "ERROR")
            return False
        
        try:
            response = self.session.get(f"{API_BASE_URL}/field/{self.field_id}/stats")
            if response.status_code == 200:
                data = response.json()
                self.log("✓ Get field stats passed")
                self.log(f"  Total images: {data.get('summary', {}).get('total_images', 0)}")
                return True
            else:
                self.log(f"✗ Get field stats failed: {response.status_code} - {response.text}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Get field stats error: {e}", "ERROR")
            return False
    
    def test_get_overall_stats(self):
        """Test getting overall statistics."""
        try:
            response = self.session.get(f"{API_BASE_URL}/stats")
            if response.status_code == 200:
                data = response.json()
                self.log("✓ Get overall stats passed")
                self.log(f"  Total fields: {data.get('summary', {}).get('total_fields', 0)}")
                return True
            else:
                self.log(f"✗ Get overall stats failed: {response.status_code} - {response.text}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Get overall stats error: {e}", "ERROR")
            return False
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        self.log("Starting CropHealth AI+ Backend API Tests", "INFO")
        self.log("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("User Registration", self.test_user_registration),
            ("User Login", self.test_user_login),
            ("Get Profile", self.test_get_profile),
            ("Create Field", self.test_create_field),
            ("Get Fields", self.test_get_fields),
            ("Image Upload & Prediction", self.test_image_upload),
            ("Get Predictions", self.test_get_predictions),
            ("Get Field Stats", self.test_get_field_stats),
            ("Get Overall Stats", self.test_get_overall_stats),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"\nRunning: {test_name}")
            if test_func():
                passed += 1
            time.sleep(0.5)  # Small delay between tests
        
        self.log("\n" + "=" * 50)
        self.log(f"Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            self.log("🎉 All tests passed! Backend API is working correctly.", "SUCCESS")
            return True
        else:
            self.log(f"❌ {total - passed} tests failed. Please check the errors above.", "ERROR")
            return False

def main():
    """Main test runner."""
    tester = APITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())

