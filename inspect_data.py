import pandas as pd
import os

file_path = 'Final_data.csv'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    try:
        df = pd.read_csv(file_path, encoding='latin1') # Try latin1 just in case
        print("Successful load.")
        print(f"Shape: {df.shape}")
        print("\nTypes:")
        print(df.dtypes)
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nHead:")
        print(df.head())
        
        if 'Disease' in df.columns:
            print("\nTop 10 Diseases by Record Count:")
            top_diseases = df['Disease'].value_counts().head(10)
            print(top_diseases)
            
            print("\nYear coverage for top 5 diseases:")
            for disease in top_diseases.index[:5]:
                years = sorted(df[df['Disease'] == disease]['year'].unique())
                print(f"{disease}: {len(years)} years ({years[0]} - {years[-1]})")
        
    except Exception as e:
        print(f"Error reading csv: {e}")
