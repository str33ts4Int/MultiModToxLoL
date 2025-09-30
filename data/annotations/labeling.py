import pandas as pd
import re

# Datei laden
file_path = "chat_output_match_1.csv"
df = pd.read_csv(file_path)

# Remove unwanted columns
columns_to_remove = ['first_frame', 'avg_confidence']
df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

# Function to check if a row is valid chat message
def is_valid_chat_message(text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return False
    
    # Remove rows that are just numbers or contain mostly numbers/symbols
    if re.match(r'^[\d\s,.;"\']+$', text.strip()):
        return False
    
    # Remove rows that don't have proper structure (player name and message)
    # Valid format should have time and player info or just player message
    if not (re.search(r'\[.*?\].*\(.*\):', text) or  # Has time and player
            re.search(r'\(.*\):', text) or  # Has player without time
            re.search(r'\[.*?\]', text)):  # Has time stamp
        # If no player format, check if it's a meaningful message
        if len(text.strip()) < 10 or text.strip().count(' ') < 2:
            return False
    
    return True

# Filter out invalid entries
df = df[df['text'].apply(is_valid_chat_message)].copy()

# Function to extract time and clean text
def extract_time_and_clean_text(text):
    if not isinstance(text, str):
        return None, text
    
    # Pattern to match time formats like [00:16], [C0:16], [00.37], [0 /.46] 
    time_patterns = [
        r'\[([CO0-9]{1,2})[:\.\s\/]*([0-9I]{1,2})\]',  
        r'\[([CO0-9]{1,2})[;\.\s]*([0-9I]{1,2})\]'     
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, text)
        if match:
            minutes = match.group(1).replace('O', '0').replace('C', '0')  # Replace O,C with 0
            seconds = match.group(2).replace('I', '1')  # Replace I with 1
            
            # Clean up any remaining non-digits
            minutes = re.sub(r'[^0-9]', '', minutes)
            seconds = re.sub(r'[^0-9]', '', seconds)
            
            if minutes and seconds:
                # Ensure proper formatting
                minutes = minutes.zfill(2)
                seconds = seconds.zfill(2)
                
                # Remove the time from text
                clean_text = re.sub(pattern, '', text).strip()
                
                return f'{minutes}:{seconds}', clean_text
    
    # If no time found, return None for time and original text
    return None, text

# Apply time extraction
df[['time', 'clean_text']] = df['text'].apply(
    lambda x: pd.Series(extract_time_and_clean_text(x))
)

# Replace the text column with clean_text and drop the temporary column
df['text'] = df['clean_text']
df = df.drop('clean_text', axis=1)

# Create empty columns for manual toxicity labeling
df['toxic'] = ''  # Empty column for you to fill with 0 or 1
df['toxicity_type'] = ''  # Empty column for you to fill with toxicity type

# Reorder columns
columns_order = ['time', 'text', 'toxic', 'toxicity_type']
df = df[columns_order]

# Remove rows where text is too short or meaningless after cleaning
df = df[df['text'].str.len() > 3].copy()

# Remove rows where time is empty, None, or NaN
df = df.dropna(subset=['time']).copy()  # Remove NaN/None values
df = df[df['time'].str.strip() != ''].copy()  # Remove empty strings
df = df[df['time'].notna()].copy()  

# Neues CSV speichern
df.to_csv("chat_labeled_match_1.csv", index=False)

print("Fertig! Clean chat data saved as chat_labeled_match_1.csv")
print(f"Removed columns: {columns_to_remove}")

