import json

def load_profile_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_analysis_report(report, filepath):
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)