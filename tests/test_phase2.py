import os
import sys
import pytest
import pandas as pd
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_validator import validate_data
from data.feature_processor import process_features
from data.market_db import insert_features, get_latest, get_label_distribution, DB_PATH

@pytest.fixture(scope="module")
def validated_data():
    return validate_data()

@pytest.fixture(scope="module")
def processed_data(validated_data):
    return process_features(validated_data)

def test_data_validator(validated_data):
    assert not validated_data.empty
    # Allow a little buffer around 8640 because months have different days (e.g. 30 vs 31 days)
    assert 8000 < len(validated_data) < 12000
    assert list(validated_data.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert validated_data['timestamp'].is_monotonic_increasing

def test_feature_processor(processed_data):
    assert not processed_data.empty
    assert 'LABEL' in processed_data.columns
    
    # RSI limits
    rsi_valid = processed_data['rsi'].dropna()
    assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()
    
    # Check NaN limit (except first 200)
    assert processed_data['sma_200'].iloc[205:].isna().sum() == 0
    
    # Label check
    labels = processed_data['LABEL'].value_counts()
    assert set(labels.index).issubset({-1, 0, 1})

def test_market_db(processed_data):
    insert_features(processed_data)
    
    assert os.path.exists(DB_PATH)
    
    # Check rows
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM features')
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count == len(processed_data)
    
    # Check get_latest
    latest = get_latest(10)
    assert len(latest) == 10
    
    dist = get_label_distribution()
    assert not dist.empty
