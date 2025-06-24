from src.models.user import db
import uuid
from sqlalchemy import func

class Image(db.Model):
    __tablename__ = 'images'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    field_id = db.Column(db.String(36), db.ForeignKey('fields.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=func.now())
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(512), nullable=False)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='image', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Image {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'field_id': self.field_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'image_path': self.image_path
        }

