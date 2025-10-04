import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def extract_advanced_features(df):
    """Erweiterte Features aus Chat-Daten extrahieren"""
    
    # Zeit-Features
    df['time_seconds'] = df['time'].apply(convert_time_to_seconds)
    df['game_phase'] = df['time_seconds'].apply(classify_game_phase)
    
    # Player-Features
    df['player_name'] = df['text'].str.extract(r'([^(]+)\s*\(')
    df['message_length'] = df['text'].str.len()
    df['caps_ratio'] = df['text'].apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0)
    
    # Behavioral Features
    df['ping_frequency'] = df.groupby('player_name')['text'].transform('count')
    df['toxicity_escalation'] = df.groupby('player_name')['toxic'].transform('cumsum')
    
    return df

def classify_game_phase(seconds):
    if seconds < 900:  # 0-15 min
        return 'early'
    elif seconds < 1800:  # 15-30 min
        return 'mid'
    else:
        return 'late'