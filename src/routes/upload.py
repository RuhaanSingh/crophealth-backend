from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import os
import uuid
from src.models.user import db
from src.models.field import Field
from src.models.image import Image
from src.models.prediction import Prediction
from src.models.recommendation import Recommendation
from src.services.external_api import ExternalAPIService
from src.services.ml_prediction import MLPredictionService
from src.services.recommendation import RecommendationService

upload_bp = Blueprint('upload', __name__)

# Initialize services
external_api_service = ExternalAPIService()
ml_service = MLPredictionService()
recommendation_service = RecommendationService()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_folder():
    """Ensure upload folder exists."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

@upload_bp.route('/upload', methods=['POST'])
@jwt_required()
def upload_image():
    """Upload an image for analysis."""
    try:
        user_id = get_jwt_identity()
        
        # Check if image file is present
        if 'image_file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get form data
        latitude = request.form.get('latitude', type=float)
        longitude = request.form.get('longitude', type=float)
        field_id = request.form.get('field_id')
        
        if latitude is None or longitude is None:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        if not field_id:
            return jsonify({'error': 'Field ID is required'}), 400
        
        # Verify field belongs to user
        field = Field.query.filter_by(id=field_id, user_id=user_id).first()
        if not field:
            return jsonify({'error': 'Field not found or access denied'}), 404
        
        # Save uploaded file
        ensure_upload_folder()
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        # Create image record
        image = Image(
            field_id=field_id,
            user_id=user_id,
            latitude=latitude,
            longitude=longitude,
            image_path=file_path
        )
        db.session.add(image)
        db.session.flush()  # Get the image ID
        
        # Fetch external data
        weather_data = external_api_service.get_weather_data(latitude, longitude)
        soil_data = external_api_service.get_soil_data(latitude, longitude)
        
        # Run ML prediction
        prediction_result = ml_service.predict(file_path, weather_data, soil_data)
        
        if not prediction_result:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Create prediction record
        prediction = Prediction(
            image_id=image.id,
            stress_scores=prediction_result['stress_scores'],
            weather_data=weather_data,
            soil_data=soil_data
        )
        db.session.add(prediction)
        db.session.flush()  # Get the prediction ID
        
        # Generate recommendations
        recommendations = recommendation_service.generate_recommendations(
            prediction_result, weather_data, soil_data, field.crop_type
        )
        
        # Create recommendation records
        recommendation_objects = []
        for rec_text in recommendations:
            rec = Recommendation(
                prediction_id=prediction.id,
                text=rec_text
            )
            db.session.add(rec)
            recommendation_objects.append(rec)
        
        db.session.commit()
        
        # Prepare response
        response_data = {
            'image_id': image.id,
            'prediction_id': prediction.id,
            'stress_scores': prediction_result['stress_scores'],
            'dominant_stress': prediction_result.get('dominant_stress'),
            'confidence': prediction_result.get('confidence'),
            'recommendations': [rec.to_dict() for rec in recommendation_objects],
            'weather_data': weather_data,
            'soil_data': soil_data
        }
        
        return jsonify(response_data), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@upload_bp.route('/images/<image_id>/predictions', methods=['GET'])
@jwt_required()
def get_image_predictions(image_id):
    """Get predictions for a specific image."""
    try:
        user_id = get_jwt_identity()
        
        # Verify image belongs to user
        image = Image.query.filter_by(id=image_id, user_id=user_id).first()
        if not image:
            return jsonify({'error': 'Image not found or access denied'}), 404
        
        # Get predictions for this image
        predictions = Prediction.query.filter_by(image_id=image_id).all()
        
        return jsonify([pred.to_dict() for pred in predictions]), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@upload_bp.route('/fields/<field_id>/predictions', methods=['GET'])
@jwt_required()
def get_field_predictions(field_id):
    """Get all predictions for a specific field."""
    try:
        user_id = get_jwt_identity()
        
        # Verify field belongs to user
        field = Field.query.filter_by(id=field_id, user_id=user_id).first()
        if not field:
            return jsonify({'error': 'Field not found or access denied'}), 404
        
        # Get all images for this field
        images = Image.query.filter_by(field_id=field_id).all()
        image_ids = [img.id for img in images]
        
        # Get predictions for all images in this field
        predictions = Prediction.query.filter(Prediction.image_id.in_(image_ids)).all()
        
        # Include image data with predictions
        result = []
        for pred in predictions:
            pred_dict = pred.to_dict()
            pred_dict['image'] = pred.image.to_dict()
            result.append(pred_dict)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

