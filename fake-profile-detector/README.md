# Fake Social Media Profile Detection System

## Overview
This project implements a fake social media profile detection system using machine learning techniques. The system analyzes social media profiles to determine the likelihood of them being fake based on various indicators.

## Project Structure
```
fake-profile-detector
├── src
│   ├── project.py          # Main application code
│   ├── utils
│   │   └── __init__.py     # Utility functions
│   └── models
│       └── __init__.py     # Machine learning models
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore file
```

## Setup Instructions
1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd fake-profile-detector
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```
   python src/project.py
   ```

## API Endpoints
- **GET /**: Returns a welcome message and available endpoints.
- **POST /api/analyze**: Analyzes a social media profile for fake indicators. Requires a JSON payload with at least a `username`.
- **GET /api/health**: Checks the health of the API and whether the model is trained.
- **GET /api/stats**: Provides statistics about the detection model.

## Usage Example
To analyze a profile, send a POST request to `/api/analyze` with the following JSON body:
```json
{
    "username": "example_user",
    "bio": "Just living life",
    "followers": 50,
    "following": 100,
    "posts": 10,
    "created_date": "2023-01-01",
    "avg_likes_per_post": 5,
    "avg_comments_per_post": 2,
    "profile_picture": "url_to_picture"
}
```

## Additional Components
- **Testing Framework**: Unit tests will be implemented using `pytest`.
- **Data Handling**: A module for handling data input/output will be added.
- **Model Persistence**: Functionality to save and load the trained model will be implemented.
- **Environment Configuration**: A `.env` file will be used for managing configurations.
- **API Documentation**: Tools like Swagger will be used for API documentation.
- **Frontend Interface**: A simple frontend will be created for user interaction.
- **Logging**: Logging will be implemented for better debugging.
- **Deployment Configuration**: Docker files and cloud service configurations will be prepared for deployment.

## License
This project is licensed under the MIT License. See the LICENSE file for details.