import pytest
from src.project import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200

def test_analyze_missing_username(client):
    response = client.post('/api/analyze', json={})
    assert response.status_code == 400