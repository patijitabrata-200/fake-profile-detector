import joblib

def save_model(model, scaler, path='model.pkl'):
    joblib.dump({'model': model, 'scaler': scaler}, path)

def load_model(path='model.pkl'):
    data = joblib.load(path)
    return data['model'], data['scaler']