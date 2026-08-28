"""Test token authentication and gate verification."""

import pytest
from fastapi.testclient import TestClient
from transcribe.server import app
from transcribe.history import is_valid_token


client = TestClient(app)


def test_token_gate_valid_token():
    """Verify configured token passes authentication."""
    res = client.get('/api/auth/verify?token=DEMO')
    assert res.status_code == 200
    assert res.json()['status'] == 'valid'


def test_token_gate_invalid_token():
    """Verify invalid token is rejected with 401."""
    res = client.get('/api/auth/verify?token=INVALID_TOKEN_XYZ')
    assert res.status_code == 401


def test_token_gate_missing_token():
    """Verify missing token returns 401."""
    res = client.get('/api/auth/verify')
    assert res.status_code == 401


def test_token_gate_html_contains_lock_screen():
    """Verify HTML index contains the token-gate element."""
    res = client.get('/')
    assert res.status_code == 200
    assert 'id="token-gate"' in res.text
    assert 'id="gate-token-input"' in res.text
    assert 'id="btn-unlock-gate"' in res.text
