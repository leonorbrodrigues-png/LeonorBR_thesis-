#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path

PIPELINE_DIR = Path("/home/students/l.rodrigues/pipeline")
OUTPUT_DIR = PIPELINE_DIR / "outputs"
LOG_DIR = PIPELINE_DIR / "logs"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging with correct path
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "step3_progress.log"),
        logging.StreamHandler()
    ]
)

def step3_split_by_po_tissue(step2_matrix_path, tissue_ontology_file):
    logging.info("=== STEP 3: CREATING SEPARATE TABLES FOR EACH PO TISSUE ===")
    logging.info("Note: UNMAPPED/nan columns will be filtered out")
    
    # Check if file exists first
    if not os.path.exists(step2_matrix_path):
        logging.error(f" PO-mapped matrix not found: {step2_matrix_path}")
        logging.error("Please run first_steps.py first")
        sys.exit(1)
    
    logging.info("Loading  matrix...")
    step2_matrix = pd.read_csv(step2_matrix_path, index_col=0)
    logging.info(f"Input matrix: {step2_matrix.shape}")
    logging.info(f"Rows (HOGs): {len(step2_matrix)}")
    logging.info(f"Columns: {len(step2_matrix.columns)}")
    
    # Show matrix statistics (generic, not normalization-specific)
    logging.info("\n-> MATRIX STATISTICS:")
    
    # Get non-zero values safely
    non_zero_mask = step2_matrix.values > 0
    non_zero_values = step2_matrix.values[non_zero_mask]
    
    if len(non_zero_values) > 0:
        logging.info(f"  Minimum non-zero value: {non_zero_values.min():.4f}")
        logging.info(f"  Maximum value: {step2_matrix.values.max():.2f}")
        logging.info(f"  Mean non-zero intensity: {non_zero_values.mean():.2f}")
        logging.info(f"  Median non-zero intensity: {np.median(non_zero_values):.2f}")
    else:
        logging.info("  No non-zero values found!")
    
    # Count zero values safely
    zero_count = np.sum(step2_matrix.values == 0)
    logging.info(f"  Zero values: {zero_count:,} ({zero_count/step2_matrix.size*100:.1f}%)")
    
    # Check for NaN values
    nan_count = np.sum(np.isnan(step2_matrix.values))
    if nan_count > 0:
        logging.warning(f"  ! NaN values: {nan_count:,} ({nan_count/step2_matrix.size*100:.1f}%) !")
    
    # FILTER OUT UNMAPPED/INVALID COLUMNS
    logging.info("\n -> Filtering out unmapped/invalid columns...")
    
    valid_columns = []
    unmapped_columns = []
    
    for col in step2_matrix.columns:
        col_str = str(col)
        
        # Check if column has the expected format
        if '_' not in col_str:
            unmapped_columns.append(col_str)
            continue
            
        # Split into species and PO term
        try:
            species, po_term = col_str.split('_', 1)
            
            # Check for invalid PO terms
            if ('UNMAPPED' in po_term.upper() or 
                po_term.lower() == 'nan' or 
                pd.isna(po_term) or
                not po_term.startswith('PO:')):
                unmapped_columns.append(col_str)
                continue
                
            # Column is valid
            valid_columns.append(col_str)
                
        except ValueError:
            unmapped_columns.append(col_str)
            logging.debug(f"Could not split column: {col_str}")
    
    # Log filtering results
    logging.info(f"- Valid PO-mapped columns: {len(valid_columns)}")
    logging.info(f"-  Unmapped/invalid columns filtered out: {len(unmapped_columns)}")
    
    if unmapped_columns:
        logging.warning(f"Filtered columns (first 10):")
        for col in unmapped_columns[:10]:
            logging.warning(f"  - {col}")
        if len(unmapped_columns) > 10:
            logging.warning(f"  ... and {len(unmapped_columns) - 10} more")
    
    # Create filtered matrix with only valid columns
    if not valid_columns:
        logging.error("! No valid PO-mapped columns found in the matrix!")
        logging.error("Please check Step 2 output for proper PO term mapping")
        sys.exit(1)
    
    filtered_matrix = step2_matrix[valid_columns].copy()
    logging.info(f"Filtered matrix shape: {filtered_matrix.shape}")
    
    # Load tissue ontology to get tissue names
    logging.info("\nLoading tissue ontology...")
    tissue_ontology = pd.read_csv(tissue_ontology_file, sep='\t')
    
    # Create PO to tissue name mapping
    po_to_tissue = {}
    for _, row in tissue_ontology.iterrows():
        po_term = str(row.get('PO_1', '')).strip()
        tissue_name = str(row.get('PO_term_1', '')).strip()
        if (po_term and po_term.lower() != 'nan' and 
            tissue_name and tissue_name.lower() != 'nan'):
            po_to_tissue[po_term] = tissue_name
    
    logging.info(f"Mapped {len(po_to_tissue)} PO terms to tissue names")
    
    # Create output directory 
    output_dir = OUTPUT_DIR / "tissue_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Group columns by PO term (using filtered matrix)
    logging.info("\nGrouping columns by PO term...")
    po_groups = {}
    valid_po_terms = set()
    invalid_po_format = []
    
    for col in filtered_matrix.columns:
        try:
            species, po_term = col.split('_', 1)
            
            # Additional validation
            if not po_term.startswith('PO:'):
                invalid_po_format.append(col)
                continue
                
            if po_term not in po_groups:
                po_groups[po_term] = {}
                valid_po_terms.add(po_term)
            po_groups[po_term][species] = col
            
        except ValueError as e:
            logging.warning(f"Unexpected error parsing column {col}: {e}")
    
    if invalid_po_format:
        logging.warning(f"! Found {len(invalid_po_format)} columns with invalid PO format (should not happen after filtering):")
        for col in invalid_po_format[:5]:
            logging.warning(f"  - {col}")
    
    logging.info(f"Found {len(po_groups)} unique valid PO terms")
    
    # Check which PO terms are in ontology
    mapped_po_count = 0
    unmapped_po_count = 0
    
    for po_term in po_groups.keys():
        if po_term in po_to_tissue:
            mapped_po_count += 1
        else:
            unmapped_po_count += 1
    
    logging.info(f"PO terms with ontology mapping: {mapped_po_count}")
    if unmapped_po_count > 0:
        logging.warning(f"PO terms WITHOUT ontology mapping: {unmapped_po_count}")
        # Show first few unmapped PO terms
        unmapped_list = [po for po in po_groups.keys() if po not in po_to_tissue]
        for po_term in unmapped_list[:5]:
            logging.warning(f"  - {po_term}")
        if len(unmapped_list) > 5:
            logging.warning(f"  ... and {len(unmapped_list) - 5} more")
    
    # Create separate table for each PO term
    created_tables = {}
    
    for po_term, species_cols in po_groups.items():
        # Get tissue name from ontology, or use PO term as fallback
        tissue_name = po_to_tissue.get(po_term)
        
        if tissue_name:
            clean_name = tissue_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace("'", "")
            status = "mapped"
        else:
            tissue_name = f"PO_{po_term.replace(':', '_')}"
            clean_name = tissue_name
            status = "unmapped"
        
        filename = output_dir / f"{clean_name}.tsv"
        
        # Extract columns for this PO term
        po_columns = list(species_cols.values())
        po_df = filtered_matrix[po_columns].copy()
        
        # Rename columns to just species names
        species_names = list(species_cols.keys())
        po_df.columns = species_names
        
        # Sort by maximum intensity (descending) to see most expressed HOGs first
        if not po_df.empty:
            po_df = po_df.loc[po_df.max(axis=1).sort_values(ascending=False).index]
        
        # Reset index to have HOG IDs as a column
        po_df_reset = po_df.reset_index()
        po_df_reset = po_df_reset.rename(columns={'index': 'HOG'})
        
        # Create metadata row with correct format
        metadata_data = {}
        metadata_data['HOG'] = 'METADATA'
        
        # Fill metadata for each species column
        for i, species in enumerate(species_names):
            if i == 0:
                # First species column gets the PO term
                metadata_data[species] = po_term
            elif i == 1 and tissue_name and status == "mapped":
                # Second species column gets the tissue name (if we have it)
                metadata_data[species] = tissue_name
            else:
                # Other columns get empty or the tissue name
                metadata_data[species] = tissue_name if status == "mapped" else ''
        
        metadata_row = pd.DataFrame([metadata_data])
        
        # Combine metadata with data
        final_df = pd.concat([metadata_row, po_df_reset], ignore_index=True)
        
        # Save to file
        final_df.to_csv(filename, sep='\t', index=False)

        created_tables[po_term] = po_df  
        
        # Get sample count for logging
        species_count = len(species_names)
        hog_count = len(po_df)
        
        logging.info(f"{status} Created: {clean_name}.tsv")
        logging.info(f"   └─ {species_count} species, {hog_count} HOGs")
        
        if species_count > 0:
            logging.info(f"   └─ Species: {', '.join(sorted(species_names)[:5])}{'...' if len(species_names) > 5 else ''}")
            
        if not po_df.empty:
            max_intensity = po_df.max().max()
            non_zero_mask = po_df.values > 0
            if np.any(non_zero_mask):
                mean_intensity = po_df.values[non_zero_mask].mean()
                logging.info(f"   └─ Max intensity: {max_intensity:.2f}")
                logging.info(f"   └─ Mean intensity: {mean_intensity:.2f}")
            else:
                logging.info(f"   └─ All values are zero or NaN")
        
        if status == "unmapped":
            logging.info(f"   └─ WARNING: PO term '{po_term}' not found in ontology")
        
        # Show file format
        logging.info(f"   └─ File format: METADATA row with PO term in first column")
    
    logging.info(f"\n=> Created {len(created_tables)} tissue tables in '{output_dir}/'")
    logging.info(f"- {mapped_po_count} tables with proper tissue names")
    if unmapped_po_count > 0:
        logging.info(f"!  {unmapped_po_count} tables with PO-term-only names (not in ontology) !")
    
    # Show summary of created tables
    if created_tables:
        logging.info("\n TISSUE TABLE SUMMARY (first 15):")
        for i, (po_term, df) in enumerate(list(created_tables.items())[:15]):
            tissue_name = po_to_tissue.get(po_term, f"PO_{po_term}")
            species_count = len(df.columns)
            hog_count = len(df)
            status = "mapped" if po_term in po_to_tissue else "unmapped"
            logging.info(f"  {status} {tissue_name}: {species_count} species, {hog_count} HOGs")
    
    # Save list of filtered columns for reference
    if unmapped_columns:
        filtered_report_path = output_dir / "FILTERED_UNMAPPED_COLUMNS.txt"
        with open(filtered_report_path, 'w') as f:
            f.write(f"Total columns in input: {len(step2_matrix.columns)}\n")
            f.write(f"Valid columns kept: {len(valid_columns)}\n")
            f.write(f"Unmapped columns filtered out: {len(unmapped_columns)}\n\n")
            f.write("FILTERED COLUMNS:\n")
            f.write("=" * 80 + "\n")
            for col in sorted(unmapped_columns):
                f.write(f"{col}\n")
        
        logging.info(f"\n- Filtered columns report saved to: {filtered_report_path}")
    
    # Show example of file format
    logging.info("\n=> EXAMPLE FILE FORMAT (for Step 4):")
    logging.info("First few lines of a sample file:")
    if created_tables:
        first_po = list(created_tables.keys())[0]
        if first_po in po_to_tissue:
            tissue_name = po_to_tissue[first_po].replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace("'", "")
        else:
            tissue_name = f"PO_{first_po.replace(':', '_')}"
        
        first_file = output_dir / f"{tissue_name}.tsv"
        try:
            with open(first_file, 'r') as f:
                lines = f.readlines()[:4]
                for line in lines:
                    logging.info(f"  {line.strip()}")
        except Exception as e:
            logging.info(f"  (Could not read sample file: {e})")
    
    return created_tables

def main():
    step2_matrix_path = OUTPUT_DIR / "first_steps_po_mapped_matrix.csv"
    tissue_ontology_file = "/home/students/l.rodrigues/Shared/High_quality_species/Tissue_ontology/ENB_TissueOntology_long.tsv"
    
    logging.info("=" * 60)
    logging.info("STEP 3: Split  PO-mapped matrix into tissue-specific tables")
    logging.info(f"Input matrix: {step2_matrix_path}")
    logging.info(f"Output directory: {OUTPUT_DIR / 'tissue_tables'}")
    logging.info("=" * 60)
    
    try:
        # First check if the input file exists
        if not os.path.exists(step2_matrix_path):
            logging.error(f" PO-mapped matrix not found: {step2_matrix_path}")
            logging.info("\n Checking for available output files:")
            for file in OUTPUT_DIR.glob("*.csv"):
                logging.info(f"  - {file.name}")
            
            # Check if normalized exists
            normalized_file = OUTPUT_DIR / "step_2_normalized.csv"
            if os.path.exists(normalized_file):
                logging.info(f"  3. To use normalized matrix instead:")
                logging.info(f"     Change line in this script: step2_matrix_path = OUTPUT_DIR / 'step_2_normalized.csv'")
                logging.info(f"     Also change output directory name to 'tissue_tables_normalized'")
            
            sys.exit(1)
        
        tissue_tables = step3_split_by_po_tissue(step2_matrix_path, tissue_ontology_file)
        logging.info("\n - STEP 3 COMPLETED SUCCESSFULLY!")
        logging.info(" The NON-NORMALIZED tissue tables are now in CORRECT FORMAT for Step 4")
        logging.info(" Each file has: METADATA row with PO term in first species column")
        
        # Final summary
        total_tissues = len(tissue_tables)
        if total_tissues > 0:
            total_hogs = len(next(iter(tissue_tables.values())))
            avg_species = sum(len(df.columns) for df in tissue_tables.values()) / total_tissues
            
            logging.info(f"\n=> FINAL SUMMARY:")
            logging.info(f"   Total tissue tables: {total_tissues}")
            logging.info(f"   HOGs per table: {total_hogs}")
            logging.info(f"   Average species per tissue: {avg_species:.1f}")
        
    except Exception as e:
        logging.error(f"\n STEP 3 FAILED: {e}")
        logging.error("Full error details:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()