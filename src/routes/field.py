from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from src.models.user import db
from src.models.field import Field
import json

field_bp = Blueprint('field', __name__)

@field_bp.route('/fields', methods=['POST'])
@jwt_required()
def create_field():
    """Create a new field."""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Validate required fields
        if not data or not data.get('name') or not data.get('polygon_geometry'):
            return jsonify({'error': 'Name and polygon_geometry are required'}), 400
        
        # Validate GeoJSON format
        try:
            polygon_geom = data['polygon_geometry']
            if isinstance(polygon_geom, dict):
                polygon_geom = json.dumps(polygon_geom)
            else:
                # Validate it's valid JSON
                json.loads(polygon_geom)
        except (json.JSONDecodeError, TypeError):
            return jsonify({'error': 'Invalid GeoJSON format for polygon_geometry'}), 400
        
        # Create new field
        field = Field(
            user_id=user_id,
            name=data['name'],
            polygon_geometry=polygon_geom,
            crop_type=data.get('crop_type')
        )
        
        db.session.add(field)
        db.session.commit()
        
        return jsonify(field.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@field_bp.route('/fields', methods=['GET'])
@jwt_required()
def get_fields():
    """Get all fields for the authenticated user."""
    try:
        user_id = get_jwt_identity()
        fields = Field.query.filter_by(user_id=user_id).all()
        
        return jsonify([field.to_dict() for field in fields]), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@field_bp.route('/fields/<field_id>', methods=['GET'])
@jwt_required()
def get_field(field_id):
    """Get details of a specific field."""
    try:
        user_id = get_jwt_identity()
        field = Field.query.filter_by(id=field_id, user_id=user_id).first()
        
        if not field:
            return jsonify({'error': 'Field not found'}), 404
        
        return jsonify(field.to_dict()), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@field_bp.route('/fields/<field_id>', methods=['PUT'])
@jwt_required()
def update_field(field_id):
    """Update an existing field."""
    try:
        user_id = get_jwt_identity()
        field = Field.query.filter_by(id=field_id, user_id=user_id).first()
        
        if not field:
            return jsonify({'error': 'Field not found'}), 404
        
        data = request.get_json()
        
        # Update field attributes
        if data.get('name'):
            field.name = data['name']
        if data.get('polygon_geometry'):
            try:
                polygon_geom = data['polygon_geometry']
                if isinstance(polygon_geom, dict):
                    polygon_geom = json.dumps(polygon_geom)
                else:
                    # Validate it's valid JSON
                    json.loads(polygon_geom)
                field.polygon_geometry = polygon_geom
            except (json.JSONDecodeError, TypeError):
                return jsonify({'error': 'Invalid GeoJSON format for polygon_geometry'}), 400
        if data.get('crop_type'):
            field.crop_type = data['crop_type']
        
        db.session.commit()
        
        return jsonify({'message': 'Field updated successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@field_bp.route('/fields/<field_id>', methods=['DELETE'])
@jwt_required()
def delete_field(field_id):
    """Delete a field."""
    try:
        user_id = get_jwt_identity()
        field = Field.query.filter_by(id=field_id, user_id=user_id).first()
        
        if not field:
            return jsonify({'error': 'Field not found'}), 404
        
        db.session.delete(field)
        db.session.commit()
        
        return jsonify({'message': 'Field deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

