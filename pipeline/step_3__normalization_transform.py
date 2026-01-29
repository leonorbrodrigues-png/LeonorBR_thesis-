import os
import pandas as pd
import numpy as np

# Define paths for input and output files
input_file = "/home/students/l.rodrigues/pipeline/outputs/first_steps_po_mapped_matrix.csv"
output_dir = "/home/students/l.rodrigues/pipeline/outputs"
output_file = os.path.join(output_dir, "step_2_normalized.csv")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the data
print("Loading data...")
print(f"Input file: {input_file}")
print(f"Output file: {output_file}")

# Check for input file 
if not os.path.exists(input_file):
    print(f" ERROR: Input file not found: {input_file}")
    print("Available files in output directory:")
    for file in os.listdir(output_dir):
        if file.endswith('.csv'):
            print(f"  - {file}")
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pd.read_csv(input_file, index_col=0)
print(f"Original matrix shape: {df.shape}")
print(f"Rows (HOGs): {len(df)}")
print(f"Columns: {len(df.columns)}")

# Create a copy as instructed
mat = df.copy()

# Check data distribution before normalization
print("\n Data distribution BEFORE normalization:")
print(f"  Minimum value: {mat.values[mat.values > 0].min():.4f}")
print(f"  Maximum value: {mat.values.max():.2f}")
print(f"  Mean intensity: {mat.values[mat.values > 0].mean():.2f}")
print(f"  Median intensity: {np.median(mat.values[mat.values > 0]):.2f}")
print(f"  Zero values: {np.sum(mat.values == 0):,} ({np.sum(mat.values == 0)/mat.size*100:.1f}%)")

# Apply total-sum normalization
print("\nApplying total-sum normalization...")
col_sums = mat.sum(axis=0, skipna=True)
target = col_sums.median()
scaling = target / col_sums

print(f"  Target sum (median): {target:.2f}")
print(f"  Column sums range: {col_sums.min():.2f} to {col_sums.max():.2f}")
print(f"  Scaling factors range: {scaling.min():.4f} to {scaling.max():.4f}")

mat_norm = mat * scaling

# Log2-transform
print("\nApplying log2 transformation (log2(x+1))...")
mat_log = np.log2(mat_norm + 1)

# Check data distribution after normalization
print("\n Data distribution AFTER normalization:")
print(f"  Minimum value: {mat_log.values[mat_log.values > 0].min():.4f}")
print(f"  Maximum value: {mat_log.values.max():.2f}")
print(f"  Mean intensity: {mat_log.values[mat_log.values > 0].mean():.2f}")
print(f"  Median intensity: {np.median(mat_log.values[mat_log.values > 0]):.2f}")
print(f"  Zero values: {np.sum(mat_log.values == 0):,} ({np.sum(mat_log.values == 0)/mat_log.size*100:.1f}%)")

# Save the normalized data
print(f"\nSaving normalized data to {output_file}...")
mat_log.to_csv(output_file)
print("Normalization complete!")

# Show summary of saved file
print(f"\n Normalized matrix saved:")
print(f"   Shape: {mat_log.shape}")
print(f"   Size: {os.path.getsize(output_file)/1024/1024:.2f} MB")

# Show sample of normalized data
print("\n Sample of normalized data (first 5 rows, first 5 columns):")
sample_df = mat_log.iloc[:5, :5]
print(sample_df)

# Check for any NaN values
print(f"\n Quality check:")
print(f"   NaN values in output: {mat_log.isna().sum().sum()}")
print(f"   Infinite values in output: {np.isinf(mat_log.values).sum()}")
