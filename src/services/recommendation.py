import random

class RecommendationService:
    """Service for generating farming recommendations based on predictions."""
    
    def __init__(self):
        # Rule-based recommendations
        self.recommendations_db = {
            'drought': [
                "Increase irrigation frequency in affected areas.",
                "Consider installing drip irrigation systems for water efficiency.",
                "Apply mulch around plants to retain soil moisture.",
                "Monitor soil moisture levels daily.",
                "Consider drought-resistant crop varieties for future planting."
            ],
            'fungal': [
                "Inspect plants for visible fungal symptoms (spots, wilting).",
                "Improve air circulation around plants by proper spacing.",
                "Apply appropriate fungicide treatment if infection is confirmed.",
                "Remove and dispose of infected plant material properly.",
                "Avoid overhead watering to reduce humidity around leaves."
            ],
            'unknown': [
                "Conduct detailed field inspection to identify the issue.",
                "Take additional photos from different angles for analysis.",
                "Consider consulting with a local agricultural extension agent.",
                "Monitor affected plants closely for symptom development.",
                "Document any changes in plant condition over time."
            ],
            'healthy': [
                "Continue current management practices.",
                "Maintain regular monitoring schedule.",
                "Ensure adequate nutrition and water supply.",
                "Keep records of successful practices for future reference.",
                "Consider this area as a reference for other field sections."
            ]
        }
        
        # Environmental condition modifiers
        self.weather_modifiers = {
            'high_temperature': "High temperatures detected. Consider providing shade or increasing irrigation.",
            'low_humidity': "Low humidity may increase drought stress. Monitor soil moisture closely.",
            'high_humidity': "High humidity increases fungal disease risk. Ensure good air circulation.",
            'recent_rain': "Recent rainfall detected. Monitor for fungal diseases and adjust irrigation accordingly.",
            'no_rain': "No recent rainfall. Increase irrigation frequency to prevent drought stress."
        }
    
    def generate_recommendations(self, prediction_result, weather_data=None, soil_data=None, crop_type=None):
        """
        Generate recommendations based on prediction results and environmental data.
        
        Args:
            prediction_result (dict): ML prediction results
            weather_data (dict): Weather data
            soil_data (dict): Soil data
            crop_type (str): Type of crop
            
        Returns:
            list: List of recommendation strings
        """
        recommendations = []
        
        if not prediction_result or 'stress_scores' not in prediction_result:
            return ["Unable to generate recommendations due to prediction error."]
        
        stress_scores = prediction_result['stress_scores']
        dominant_stress = prediction_result.get('dominant_stress', 'unknown')
        confidence = prediction_result.get('confidence', 0)
        
        # Base recommendations based on dominant stress type
        if dominant_stress in self.recommendations_db:
            base_recs = self.recommendations_db[dominant_stress]
            # Select 2-3 recommendations
            selected_recs = random.sample(base_recs, min(3, len(base_recs)))
            recommendations.extend(selected_recs)
        
        # Add confidence-based recommendations
        if confidence < 0.6:
            recommendations.append("Prediction confidence is low. Consider taking additional photos for better analysis.")
        
        # Add weather-based recommendations
        if weather_data:
            weather_recs = self._get_weather_recommendations(weather_data)
            recommendations.extend(weather_recs)
        
        # Add soil-based recommendations
        if soil_data:
            soil_recs = self._get_soil_recommendations(soil_data)
            recommendations.extend(soil_recs)
        
        # Add crop-specific recommendations
        if crop_type:
            crop_recs = self._get_crop_specific_recommendations(crop_type, dominant_stress)
            recommendations.extend(crop_recs)
        
        # Add stress-specific recommendations for secondary stresses
        for stress_type, score in stress_scores.items():
            if stress_type != dominant_stress and score > 0.3:  # Secondary stress threshold
                recommendations.append(f"Moderate {stress_type} risk detected (score: {score:.2f}). Monitor closely.")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _get_weather_recommendations(self, weather_data):
        """Generate weather-based recommendations."""
        recommendations = []
        
        temperature = weather_data.get('temperature', 20)
        humidity = weather_data.get('humidity', 50)
        precipitation = weather_data.get('precipitation_1h', 0)
        
        if temperature > 30:
            recommendations.append(self.weather_modifiers['high_temperature'])
        
        if humidity < 40:
            recommendations.append(self.weather_modifiers['low_humidity'])
        elif humidity > 80:
            recommendations.append(self.weather_modifiers['high_humidity'])
        
        if precipitation > 0:
            recommendations.append(self.weather_modifiers['recent_rain'])
        elif precipitation == 0:
            recommendations.append(self.weather_modifiers['no_rain'])
        
        return recommendations
    
    def _get_soil_recommendations(self, soil_data):
        """Generate soil-based recommendations."""
        recommendations = []
        
        ph = soil_data.get('ph_water', 7.0)
        organic_carbon = soil_data.get('organic_carbon', 0)
        
        if ph < 6.0:
            recommendations.append("Soil pH is acidic. Consider lime application to raise pH.")
        elif ph > 8.0:
            recommendations.append("Soil pH is alkaline. Consider sulfur application to lower pH.")
        
        if organic_carbon < 10:  # Low organic matter
            recommendations.append("Low soil organic matter detected. Consider adding compost or organic amendments.")
        
        return recommendations
    
    def _get_crop_specific_recommendations(self, crop_type, stress_type):
        """Generate crop-specific recommendations."""
        recommendations = []
        
        crop_specific = {
            'corn': {
                'drought': "Corn is particularly sensitive to drought during tasseling. Ensure adequate water supply.",
                'fungal': "Watch for common corn diseases like gray leaf spot and northern corn leaf blight."
            },
            'wheat': {
                'drought': "Wheat drought stress can significantly impact grain filling. Monitor soil moisture.",
                'fungal': "Common wheat diseases include rust and powdery mildew. Consider fungicide application."
            },
            'soybean': {
                'drought': "Soybeans are most sensitive to drought during pod filling stage.",
                'fungal': "Watch for soybean rust and white mold, especially in humid conditions."
            }
        }
        
        crop_lower = crop_type.lower() if crop_type else ''
        if crop_lower in crop_specific and stress_type in crop_specific[crop_lower]:
            recommendations.append(crop_specific[crop_lower][stress_type])
        
        return recommendations

