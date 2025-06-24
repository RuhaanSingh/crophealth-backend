from src.models.user import db
import uuid
from sqlalchemy import func

class Field(db.Model):
    __tablename__ = 'fields'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    polygon_geometry = db.Column(db.Text, nullable=False)  # Store as GeoJSON text
    crop_type = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=func.now())
    
    # Relationships
    images = db.relationship('Image', backref='field', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Field {self.name}>'

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'polygon_geometry': self.polygon_geometry,
            'crop_type': self.crop_type,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

