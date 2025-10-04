import pandas as pd
import glob

def consolidate_matches():
    # Alle CSV-Dateien sammeln
    csv_files = glob.glob("data/annotations/chat_labeled_match_*.csv")
    
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        # Match-ID hinzufügen
        match_id = file.split('_')[-1].replace('.csv', '')
        df['match_id'] = match_id
        all_data.append(df)
    
    # Zusammenführen
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv("data/processed/all_matches_labeled.csv", index=False)
    
    return combined_df