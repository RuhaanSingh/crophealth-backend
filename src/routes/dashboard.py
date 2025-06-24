from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from src.models.user import db
from src.models.field import Field
from src.models.image import Image
from src.models.prediction import Prediction
from src.models.recommendation import Recommendation

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/field/<field_id>/stats', methods=['GET'])
@jwt_required()
def get_field_stats(field_id):
    """Get aggregated statistics for a specific field."""
    try:
        user_id = get_jwt_identity()
        
        # Verify field belongs to user
        field = Field.query.filter_by(id=field_id, user_id=user_id).first()
        if not field:
            return jsonify({'error': 'Field not found or access denied'}), 404
        
        # Get date range from query parameters (default to last 30 days)
        days = request.args.get('days', 30, type=int)
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all images for this field within date range
        images = Image.query.filter(
            Image.field_id == field_id,
            Image.timestamp >= start_date
        ).all()
        
        if not images:
            return jsonify({
                'summary': {'total_images': 0, 'stress_distribution': {}},
                'trends': [],
                'recent_predictions': []
            }), 200
        
        image_ids = [img.id for img in images]
        
        # Get predictions for these images
        predictions = Prediction.query.filter(
            Prediction.image_id.in_(image_ids)
        ).order_by(desc(Prediction.timestamp)).all()
        
        # Calculate stress distribution
        stress_distribution = {'healthy': 0, 'drought': 0, 'fungal': 0, 'unknown': 0}
        total_predictions = len(predictions)
        
        for pred in predictions:
            if pred.stress_scores:
                dominant_stress = max(pred.stress_scores, key=pred.stress_scores.get)
                stress_distribution[dominant_stress] += 1
        
        # Convert to percentages
        if total_predictions > 0:
            stress_distribution = {k: (v / total_predictions) * 100 for k, v in stress_distribution.items()}
        
        # Calculate trends (daily averages)
        trends = []
        for i in range(days):
            day_start = start_date + timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            
            day_predictions = [p for p in predictions 
                             if day_start <= p.timestamp < day_end]
            
            if day_predictions:
                # Calculate average stress scores for the day
                avg_scores = {'healthy': 0, 'drought': 0, 'fungal': 0, 'unknown': 0}
                for pred in day_predictions:
                    if pred.stress_scores:
                        for stress_type, score in pred.stress_scores.items():
                            avg_scores[stress_type] += score
                
                # Average the scores
                num_preds = len(day_predictions)
                avg_scores = {k: v / num_preds for k, v in avg_scores.items()}
                
                trends.append({
                    'date': day_start.isoformat(),
                    'stress_scores': avg_scores,
                    'prediction_count': num_preds
                })
        
        # Get recent predictions (last 10)
        recent_predictions = []
        for pred in predictions[:10]:
            pred_dict = pred.to_dict()
            pred_dict['image'] = pred.image.to_dict()
            recent_predictions.append(pred_dict)
        
        response_data = {
            'summary': {
                'total_images': len(images),
                'total_predictions': total_predictions,
                'stress_distribution': stress_distribution,
                'field_info': field.to_dict()
            },
            'trends': trends,
            'recent_predictions': recent_predictions
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_overall_stats():
    """Get overall aggregated statistics for the user."""
    try:
        user_id = get_jwt_identity()
        
        # Get date range from query parameters (default to last 30 days)
        days = request.args.get('days', 30, type=int)
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all user's fields
        fields = Field.query.filter_by(user_id=user_id).all()
        field_ids = [field.id for field in fields]
        
        if not field_ids:
            return jsonify({
                'summary': {'total_fields': 0, 'total_images': 0, 'stress_distribution': {}},
                'field_summaries': [],
                'trends': []
            }), 200
        
        # Get all images for user's fields within date range
        images = Image.query.filter(
            Image.field_id.in_(field_ids),
            Image.timestamp >= start_date
        ).all()
        
        if not images:
            return jsonify({
                'summary': {'total_fields': len(fields), 'total_images': 0, 'stress_distribution': {}},
                'field_summaries': [field.to_dict() for field in fields],
                'trends': []
            }), 200
        
        image_ids = [img.id for img in images]
        
        # Get predictions for these images
        predictions = Prediction.query.filter(
            Prediction.image_id.in_(image_ids)
        ).order_by(desc(Prediction.timestamp)).all()
        
        # Calculate overall stress distribution
        stress_distribution = {'healthy': 0, 'drought': 0, 'fungal': 0, 'unknown': 0}
        total_predictions = len(predictions)
        
        for pred in predictions:
            if pred.stress_scores:
                dominant_stress = max(pred.stress_scores, key=pred.stress_scores.get)
                stress_distribution[dominant_stress] += 1
        
        # Convert to percentages
        if total_predictions > 0:
            stress_distribution = {k: (v / total_predictions) * 100 for k, v in stress_distribution.items()}
        
        # Calculate field summaries
        field_summaries = []
        for field in fields:
            field_images = [img for img in images if img.field_id == field.id]
            field_image_ids = [img.id for img in field_images]
            field_predictions = [p for p in predictions if p.image_id in field_image_ids]
            
            field_stress_dist = {'healthy': 0, 'drought': 0, 'fungal': 0, 'unknown': 0}
            for pred in field_predictions:
                if pred.stress_scores:
                    dominant_stress = max(pred.stress_scores, key=pred.stress_scores.get)
                    field_stress_dist[dominant_stress] += 1
            
            # Convert to percentages
            if field_predictions:
                field_stress_dist = {k: (v / len(field_predictions)) * 100 for k, v in field_stress_dist.items()}
            
            field_summary = field.to_dict()
            field_summary['image_count'] = len(field_images)
            field_summary['prediction_count'] = len(field_predictions)
            field_summary['stress_distribution'] = field_stress_dist
            field_summaries.append(field_summary)
        
        # Calculate overall trends (daily averages)
        trends = []
        for i in range(days):
            day_start = start_date + timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            
            day_predictions = [p for p in predictions 
                             if day_start <= p.timestamp < day_end]
            
            if day_predictions:
                # Calculate average stress scores for the day
                avg_scores = {'healthy': 0, 'drought': 0, 'fungal': 0, 'unknown': 0}
                for pred in day_predictions:
                    if pred.stress_scores:
                        for stress_type, score in pred.stress_scores.items():
                            avg_scores[stress_type] += score
                
                # Average the scores
                num_preds = len(day_predictions)
                avg_scores = {k: v / num_preds for k, v in avg_scores.items()}
                
                trends.append({
                    'date': day_start.isoformat(),
                    'stress_scores': avg_scores,
                    'prediction_count': num_preds
                })
        
        response_data = {
            'summary': {
                'total_fields': len(fields),
                'total_images': len(images),
                'total_predictions': total_predictions,
                'stress_distribution': stress_distribution
            },
            'field_summaries': field_summaries,
            'trends': trends
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/fields/<field_id>/recommendations', methods=['GET'])
@jwt_required()
def get_field_recommendations(field_id):
    """Get recommendations for a specific field."""
    try:
        user_id = get_jwt_identity()
        
        # Verify field belongs to user
        field = Field.query.filter_by(id=field_id, user_id=user_id).first()
        if not field:
            return jsonify({'error': 'Field not found or access denied'}), 404
        
        # Get recent images for this field (last 30 days)
        start_date = datetime.utcnow() - timedelta(days=30)
        images = Image.query.filter(
            Image.field_id == field_id,
            Image.timestamp >= start_date
        ).all()
        
        image_ids = [img.id for img in images]
        
        # Get predictions and their recommendations
        predictions = Prediction.query.filter(
            Prediction.image_id.in_(image_ids)
        ).order_by(desc(Prediction.timestamp)).all()
        
        all_recommendations = []
        for pred in predictions:
            recommendations = Recommendation.query.filter_by(prediction_id=pred.id).all()
            for rec in recommendations:
                rec_dict = rec.to_dict()
                rec_dict['prediction'] = pred.to_dict()
                all_recommendations.append(rec_dict)
        
        # Sort by timestamp (most recent first)
        all_recommendations.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify(all_recommendations), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/images/<image_id>/recommendations', methods=['GET'])
@jwt_required()
def get_image_recommendations(image_id):
    """Get recommendations for a specific image."""
    try:
        user_id = get_jwt_identity()
        
        # Verify image belongs to user
        image = Image.query.filter_by(id=image_id, user_id=user_id).first()
        if not image:
            return jsonify({'error': 'Image not found or access denied'}), 404
        
        # Get predictions for this image
        predictions = Prediction.query.filter_by(image_id=image_id).all()
        
        all_recommendations = []
        for pred in predictions:
            recommendations = Recommendation.query.filter_by(prediction_id=pred.id).all()
            for rec in recommendations:
                rec_dict = rec.to_dict()
                rec_dict['prediction'] = pred.to_dict()
                all_recommendations.append(rec_dict)
        
        # Sort by timestamp (most recent first)
        all_recommendations.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify(all_recommendations), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

