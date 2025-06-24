#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.user import User, db
from src.main import app

def test_backend():
    """Test basic backend functionality."""
    with app.app_context():
        try:
            # Test database connection
            db.create_all()
            print("✓ Database tables created successfully")
            
            # Test user creation
            test_user = User(name="Test User", email="test@example.com")
            test_user.set_password("testpassword")
            db.session.add(test_user)
            db.session.commit()
            print("✓ User creation works")
            
            # Test user authentication
            user = User.query.filter_by(email="test@example.com").first()
            if user and user.check_password("testpassword"):
                print("✓ User authentication works")
            else:
                print("✗ User authentication failed")
            
            print("\nBackend basic functionality test completed successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Backend test failed: {e}")
            return False

if __name__ == "__main__":
    test_backend()

