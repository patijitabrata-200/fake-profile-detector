# filepath: fake-profile-detector/src/project.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import numpy as np
from datetime import datetime
import random
import string
import logging
from src.utils import load_profile_data, save_analysis_report
from src.models.model_persistence import save_model, load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeProfileDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        self.train_model()
    
    def extract_features(self, profile_data):
        # Convert numeric fields to int/float
        username = profile_data.get('username', '')
        username_length = len(username)
        username_digits = sum(c.isdigit() for c in username)
        long_numbers = 1 if re.search(r'\d{3,}', username) else 0
        bio = profile_data.get('bio', '')
        bio_length = len(bio)
        has_picture = 1 if profile_data.get('profile_picture') else 0
        followers = int(profile_data.get('followers', 0))
        following = int(profile_data.get('following', 0))
        posts = int(profile_data.get('posts', 0))
        following_ratio = following / max(followers, 1)
        created_date = profile_data.get('created_date', '')
        if created_date:
            try:
                created = datetime.strptime(created_date, '%Y-%m-%d')
                account_age = (datetime.now() - created).days
            except:
                account_age = 0
        else:
            account_age = 0
        avg_likes = float(profile_data.get('avg_likes_per_post', 0))
        avg_comments = float(profile_data.get('avg_comments_per_post', 0))
        engagement_rate = avg_likes / max(followers, 1)
        suspicious = 1 if self.has_suspicious_patterns(profile_data) else 0
        name_randomness = self.calculate_name_randomness(username)

        features = [
            username_length, username_digits, long_numbers, bio_length, 
            has_picture, followers, following, posts, following_ratio,
            account_age, avg_likes, avg_comments, engagement_rate,
            suspicious, name_randomness
        ]
        return features
    
    def has_suspicious_patterns(self, profile_data):
        suspicious_indicators = 0
        bio = profile_data.get('bio', '').lower()
        generic_phrases = ['love life', 'living life', 'enjoying life', 'blessed', 'grateful']
        if any(phrase in bio for phrase in generic_phrases) and len(bio.split()) < 10:
            suspicious_indicators += 1

        # Convert to int before comparison
        followers = int(profile_data.get('followers', 0))
        following = int(profile_data.get('following', 0))
        if following > 1000 and followers < 100:
            suspicious_indicators += 1

        created_date = profile_data.get('created_date', '')
        if created_date:
            try:
                created = datetime.strptime(created_date, '%Y-%m-%d')
                account_age = (datetime.now() - created).days
                posts = int(profile_data.get('posts', 0))  # Convert to int
                if account_age < 30 and posts > 50:
                    suspicious_indicators += 1
            except:
                pass

        return suspicious_indicators >= 2
    
    def calculate_name_randomness(self, name):
        if not name:
            return 1
        
        consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{3,}', name)
        random_numbers = re.findall(r'\d{3,}', name)
        
        randomness_score = len(consonant_clusters) + len(random_numbers)
        return min(randomness_score / 3, 1)
    
    def generate_synthetic_data(self, n_samples=1000):
        data = []
        labels = []
        
        for i in range(n_samples):
            is_fake = random.choice([0, 1])
            if is_fake:
                username_length = random.randint(8, 15)
                username_digits = random.randint(3, 8)
                long_numbers = 1
                bio_length = random.randint(0, 50)
                has_picture = random.choice([0, 1])
                followers = random.randint(0, 500)
                following = random.randint(500, 3000)
                posts = random.randint(20, 200)
                account_age = random.randint(1, 90)
                avg_likes = random.randint(0, 20)
                avg_comments = random.randint(0, 5)
                suspicious = 1
                name_randomness = random.uniform(0.3, 1.0)
            else:
                username_length = random.randint(5, 12)
                username_digits = random.randint(0, 3)
                long_numbers = 0
                bio_length = random.randint(20, 200)
                has_picture = 1
                followers = random.randint(50, 5000)
                following = random.randint(50, 1000)
                posts = random.randint(5, 500)
                account_age = random.randint(90, 2000)
                avg_likes = random.randint(5, 200)
                avg_comments = random.randint(1, 50)
                suspicious = 0
                name_randomness = random.uniform(0.0, 0.3)
            
            following_ratio = following / max(followers, 1)
            engagement_rate = avg_likes / max(followers, 1)
            
            features = [
                username_length, username_digits, long_numbers, bio_length, 
                has_picture, followers, following, posts, following_ratio,
                account_age, avg_likes, avg_comments, engagement_rate,
                suspicious, name_randomness
            ]
            
            data.append(features)
            labels.append(is_fake)
        
        return np.array(data), np.array(labels)
    
    def train_model(self):
        print("Training fake profile detection model...")
        X, y = self.generate_synthetic_data(2000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        print(f"Training accuracy: {train_accuracy:.3f}")
        print(f"Testing accuracy: {test_accuracy:.3f}")
        self.trained = True
    
    def predict_fake_probability(self, profile_data):
        if not self.trained:
            return {"error": "Model not trained"}
        
        features = self.extract_features(profile_data)
        features_scaled = self.scaler.transform([features])
        fake_probability = self.model.predict_proba(features_scaled)[0][1]
        prediction = self.model.predict(features_scaled)[0]
        
        feature_names = [
            'Username Length', 'Username Digits', 'Long Numbers', 'Bio Length',
            'Has Profile Picture', 'Followers', 'Following', 'Posts', 
            'Following/Followers Ratio', 'Account Age', 'Avg Likes', 
            'Avg Comments', 'Engagement Rate', 'Suspicious Patterns', 'Name Randomness'
        ]
        
        feature_importance = list(zip(feature_names, self.model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'is_fake': bool(prediction),
            'fake_probability': float(fake_probability),
            'confidence': 'High' if fake_probability > 0.8 or fake_probability < 0.2 else 'Medium',
            'risk_level': self.get_risk_level(fake_probability),
            'top_indicators': feature_importance[:5],
            'features_analyzed': dict(zip(feature_names, features))
        }
    
    def get_risk_level(self, probability):
        if probability >= 0.8:
            return "High Risk"
        elif probability >= 0.6:
            return "Medium Risk"
        elif probability >= 0.4:
            return "Low Risk"
        else:
            return "Likely Genuine"

detector = FakeProfileDetector()

@app.route('/')
def index():
    return jsonify({
        "message": "Fake Social Media Profile Detection API",
        "version": "1.0",
        "endpoints": {
            "/api/analyze": "POST - Analyze a social media profile",
            "/api/health": "GET - Check API health",
            "/api/stats": "GET - Get detection statistics"
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_profile():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ['username']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        result = detector.predict_fake_probability(data)
        
        if "error" in result:
            return jsonify(result), 500
        
        analysis_report = {
            "profile_data": data,
            "detection_result": result,
            "timestamp": datetime.now().isoformat(),
            "analysis_id": generate_analysis_id()
        }
        
        return jsonify(analysis_report)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_trained": detector.trained,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "model_info": {
            "algorithm": "Random Forest Classifier",
            "features_count": 15,
            "trained": detector.trained
        },
        "detection_categories": {
            "High Risk": "80%+ probability of being fake",
            "Medium Risk": "60-79% probability of being fake", 
            "Low Risk": "40-59% probability of being fake",
            "Likely Genuine": "Less than 40% probability of being fake"
        }
    })

def generate_analysis_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

if __name__ == '__main__':
    print("Starting Fake Profile Detection API...")
    app.run(debug=True, port=5000)