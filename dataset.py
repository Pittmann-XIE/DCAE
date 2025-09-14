import pandas as pd
import csv

def sample_csv_entries(input_file, output_file, n_samples):
    # Try different encodings and separators
    encodings = ['utf-8', 'latin-1', 'cp1252']
    separators = [',', '\t', '|', ';']
    
    df = None
    for encoding in encodings:
        for sep in separators:
            try:
                print(f"Trying encoding: {encoding}, separator: '{sep}'")
                df = pd.read_csv(input_file, 
                               sep=sep, 
                               encoding=encoding, 
                               on_bad_lines='skip',
                               low_memory=False)
                print(f"Success! Loaded {len(df)} rows")
                print(f"Columns: {list(df.columns)}")
                break
            except Exception as e:
                continue
        if df is not None:
            break
    
    if df is None:
        raise ValueError("Could not read the CSV file")
    
    # Filter for train subset entries
    if 'Subset' in df.columns:
        train_df = df[df['Subset'] == 'validation']
        print(f"Train subset rows: {len(train_df)}")
    else:
        print("No 'Subset' column found, using all rows")
        train_df = df
    
    # Convert n_samples to int
    n_samples = int(n_samples)
    
    # Randomly sample N entries with seed 43
    if n_samples > len(train_df):
        print(f"Warning: Requested {n_samples} samples, but only {len(train_df)} available")
        sampled_df = train_df
    else:
        sampled_df = train_df.sample(n=n_samples, random_state=43)
    
    # Create new format: "train/imageID"
    new_data = []
    for _, row in sampled_df.iterrows():
        new_data.append(f"validation/{row['ImageID']}")
    
    # Save to new CSV file
    with open(output_file, 'w') as f:
        for entry in new_data:
            f.write(entry + '\n')
    
    print(f"Successfully created {output_file} with {len(new_data)} entries")

# Usage
sample_csv_entries('./validation-images-with-rotation.csv', 'Images_to_download_valid.csv', 100)
