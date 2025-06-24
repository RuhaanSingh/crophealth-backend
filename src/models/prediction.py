from src.models.user import db
import uuid
from sqlalchemy import func

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = db.Column(db.String(36), db.ForeignKey('images.id'), nullable=False)
    stress_scores = db.Column(db.JSON, nullable=False)  # e.g., {"drought": 0.8, "fungal": 0.1, "healthy": 0.1}
    timestamp = db.Column(db.DateTime, default=func.now())
    weather_data = db.Column(db.JSON)  # Cached OpenWeatherMap data
    soil_data = db.Column(db.JSON)  # Cached SoilGrids data
    
    # Relationships
    recommendations = db.relationship('Recommendation', backref='prediction', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Prediction {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'image_id': self.image_id,
            'stress_scores': self.stress_scores,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'weather_data': self.weather_data,
            'soil_data': self.soil_data
        }

