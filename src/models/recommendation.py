from src.models.user import db
import uuid
from sqlalchemy import func

class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id = db.Column(db.String(36), db.ForeignKey('predictions.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=func.now())

    def __repr__(self):
        return f'<Recommendation {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'prediction_id': self.prediction_id,
            'text': self.text,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

