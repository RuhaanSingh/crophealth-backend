import os
import sys

# DON\'T CHANGE THE LINE BELOW. It is used for internal testing.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from datetime import timedelta

# Import models and routes
from src.models.user import db
from src.routes.auth import auth_bp
from src.routes.field import field_bp
from src.routes.upload import upload_bp
from src.routes.dashboard import dashboard_bp

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "super-secret")  # Change this in production!
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)

# Database configuration
# Use DATABASE_URL environment variable for production (e.g., PostgreSQL on Render)
# Fallback to SQLite for local development
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///src/database/app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix="/api")
app.register_blueprint(field_bp, url_prefix="/api")
app.register_blueprint(upload_bp, url_prefix="/api")
app.register_blueprint(dashboard_bp, url_prefix="/api")

# Create database tables if they don\'t exist
with app.app_context():
    db.create_all()

@app.route("/api/health")
def health_check():
    return {"status": "ok", "message": "CropHealth AI+ Backend is running!"}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=os.environ.get("PORT", 5000))
