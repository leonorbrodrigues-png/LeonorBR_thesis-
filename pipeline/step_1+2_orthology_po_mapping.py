#!/usr/bin/env python3
"""
FIRST_STEPS: Combined pipeline for heat stress proteins with PO mapping
Steps: 
1. Filter orthogroups for heat stress proteins only
2. Create HOG × Individual Protein matrix
3. Map p-numbers to PO terms
"""
import pandas as pd
import os
import numpy as np
import time
import logging
import re
from tqdm import tqdm
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path("/home/students/l.rodrigues/pipeline/scripts")
OUTPUT_DIR = Path("/home/students/l.rodrigues/pipeline/outputs")
LOG_DIR = Path("/home/students/l.rodrigues/pipeline/logs")

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "first_steps_progress.log"),
        logging.StreamHandler()
    ]
)

# SPECIES MAPPING: GCF accession names → Common names used in intensity files
SPECIES_MAPPING = {
    'GCF_042453785.1_Malus_domestica.ref': 'Apple',
    'GCF_034140825.1_Oryza_sativa.ref': 'Rice', 
    'GCF_902167145.1_Zea_mays.ref': 'Maize',
    'GCF_904849725.1_Hordeum_vulgare.ref': 'Barley',
    'GCF_036512215.1_Solanum_lycopersicum.ref': 'Tomato',
    'GCF_030704535.1_Vitis_vinifera.ref': 'Grapevine',
    'GCF_025177605.1_Cucumis_melo.ref': 'Melon',
    'GCF_022201045.2_Citrus_sinensis.ref': 'Orange',
    'GCF_020520425.1_Spinacia_oleracea.ref': 'Spinach',
    'GCF_011075055.1_Mangifera_indica.ref': 'Mango',
    'GCF_002870075.4_Lactuca_sativa.ref': 'Lettuce',
    'GCF_001659605.2_Manihot_esculenta.ref': 'Cassava',
    'GCF_001625215.2_Daucus_carota.ref': 'Carrot',
    'GCF_000499845.2_Phaseolus_vulgaris.ref': 'CommonBean',
    'GCF_024323335.1_Pisum_sativum.ref': 'Gardenpea',
    'GCF_963583255.1_Pyrus_communis.ref': 'Pear',
    'GCF_901000735.1_Corylus_avellana.ref': 'Hazelnut',
    'GCF_963169125.1_Humulus_lupulus.ref': 'Hops'
}

def _normalize_pnumber(p):
    #Strip replicate suffixes (e.g. _1) and version suffixes (.v1) from p-number
    if p is None:
        return p
    s = str(p).strip()
    s = re.sub(r'_[0-9]+$', '', s)    # strip trailing _1, _2, ...
    s = re.sub(r'\.v\d+$', '', s)     # strip .v1, .v2, ...
    return s

def _normalize_protein_id(protein_id):
    #Strip version suffix from protein ID (e.g., .1, .2, .3) for matching
    if not isinstance(protein_id, str):
        return protein_id
    # Remove version suffix like .1, .2, .3, etc.
    return re.sub(r'\.\d+$', '', protein_id.strip())

def _normalize_protein_col(col):
    if not isinstance(col, str) or '_' not in col:
        return col
    species, rest = col.split('_', 1)
    # Normalize protein ID part
    rest = _normalize_protein_id(rest)
    return f"{species}_{rest}"

def map_column(col, protein_to_tissue, p_to_po):
    #Map Species_ProteinID -> Species_POterm. Normalize both protein key and p-number before lookup.
    # try exact key
    tissue = protein_to_tissue.get(col)
    if tissue is None:
        # try normalized protein column
        norm_col = _normalize_protein_col(col)
        tissue = protein_to_tissue.get(norm_col)
    if tissue is None:
        return col  # leave as-is (unmapped)
    
    # normalize p-number before lookup
    tissue_norm = _normalize_pnumber(tissue)
    
    po_term = None
    if tissue_norm is not None:
        po_term = p_to_po.get(tissue_norm)
    
    # If not found with normalized, try original (with suffix)
    if po_term is None:
        po_term = p_to_po.get(str(tissue))
    
    if po_term is None:
        po_term = f"UNMAPPED_{tissue_norm or tissue}"
    
    species = col.split('_')[0] if isinstance(col, str) and '_' in col else 'UNKNOWN'
    return f"{species}_{po_term}"

def create_heat_stress_matrix():
    
    logging.info("=== FIRST STEPS: HEAT STRESS PROTEIN PIPELINE ===")
    logging.info(f"Script location: {SCRIPT_DIR}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info(f"Log directory: {LOG_DIR}")
    start_time = time.time()
    
    # File paths
    orthogroups_file = "/home/students/l.rodrigues/Shared/High_quality_species/orthofinder/Results_Oct27_w_heat/Orthogroups/Orthogroups.tsv"
    protein_files_path = "/home/students/l.rodrigues/Shared/High_quality_species/Intensities"
    tissue_file = "/home/students/l.rodrigues/Shared/High_quality_species/Tissue_ontology/ENB_TissueOntology_long.tsv"
    output_file = OUTPUT_DIR / "first_steps_po_mapped_matrix.csv"
    
    #  Filter orthogroups for heat stress proteins 
    logging.info("1. Loading and filtering orthogroups for heat stress proteins...")
    
    # Load orthogroups
    orthogroups = pd.read_csv(orthogroups_file, sep='\t')
    
    # Filter for rows with heat_stress_protein_sequences
    heat_mask = orthogroups['heat_stress_protein_sequences'].notna()
    heat_orthogroups = orthogroups[heat_mask].copy()
    
    logging.info(f"   Total orthogroups: {len(orthogroups):,}")
    logging.info(f"   Heat stress orthogroups: {len(heat_orthogroups):,}")
    logging.info(f"   Percentage: {len(heat_orthogroups)/len(orthogroups)*100:.1f}%")
    
    # Create protein→HOG mapping for heat stress proteins only
    logging.info("2. Creating protein→HOG mapping for heat stress proteins...")
    logging.info(f"   Using species mapping for {len(SPECIES_MAPPING)} species")
    
    protein_to_hog = {}
    protein_version_map = {}  # Map normalized protein ID → original protein ID
    
    # Process only heat stress orthogroups
    for _, row in tqdm(heat_orthogroups.iterrows(), 
                      total=len(heat_orthogroups),
                      desc="Processing heat stress orthogroups"):
        hog_id = row['Orthogroup']
        # Process all species columns except metadata
        metadata_cols = ['Orthogroup', 'heat_stress_protein_sequences']
        species_cols = [col for col in orthogroups.columns if col not in metadata_cols]
        
        for gcf_species in species_cols:
            if pd.notna(row[gcf_species]):
                # Convert GCF species name to common name
                common_species = SPECIES_MAPPING.get(gcf_species)
                if not common_species:
                    logging.debug(f"   Skipping unmapped species: {gcf_species}")
                    continue
                
                proteins = str(row[gcf_species]).split(', ')
                for protein in proteins:
                    protein = protein.strip()
                    if protein:
                        # Normalize protein ID (remove version suffix)
                        normalized_protein = _normalize_protein_id(protein)
                        
                        # Create lookup key with common species name
                        lookup_key = f"{common_species}_{normalized_protein}"
                        protein_to_hog[lookup_key] = hog_id
                        
                        # Store original protein ID for reference
                        protein_version_map[normalized_protein] = protein
    
    logging.info(f"   Total protein-HOG pairs: {len(protein_to_hog):,}")
    logging.info(f"   Unique HOGs: {len(set(protein_to_hog.values())):,}")
    logging.info(f"   Unique proteins: {len(protein_to_hog):,}")
    
    # Log species mapping statistics
    gcf_species_in_heat = [col for col in heat_orthogroups.columns 
                          if col not in ['Orthogroup', 'heat_stress_protein_sequences']]
    mapped_count = sum(1 for gcf in gcf_species_in_heat if gcf in SPECIES_MAPPING)
    logging.info(f"   Species in heat orthogroups: {len(gcf_species_in_heat)}")
    logging.info(f"   Species mapped to common names: {mapped_count}")
    
    #  Create HOG × Individual Protein matrix 
    logging.info("3. Processing species files and creating protein-tissue matrix...")
    
    protein_data = []
    intensity_files = [f for f in os.listdir(protein_files_path) if f.endswith('.tsv')]
    
    # Track mapping statistics
    proteins_found = 0
    proteins_not_found = 0
    
    # Process each species file
    for file_idx, file in enumerate(tqdm(intensity_files, 
                                       desc="Processing species files")):
        species = file.replace('.proteins.tsv', '')
        
        # Read protein intensity file
        file_path = os.path.join(protein_files_path, file)
        df = pd.read_csv(file_path, sep='\t')
        tissue_cols = [col for col in df.columns if 'MaxLFQ Intensity' in col]
        
        # Process each protein
        for _, row in tqdm(df.iterrows(), 
                          total=len(df),
                          desc=f"Processing {species} proteins",
                          leave=False):
            protein_id = row['Protein ID']
            
            # Normalize protein ID for lookup
            normalized_id = _normalize_protein_id(str(protein_id))
            lookup_key = f"{species}_{normalized_id}"
            
            if lookup_key in protein_to_hog:
                proteins_found += 1
                hog_id = protein_to_hog[lookup_key]
                
                max_intensity = 0
                best_tissue_col = None
                
                for tissue_col in tissue_cols:
                    intensity = row[tissue_col]
                    if pd.notna(intensity) and intensity > max_intensity:
                        max_intensity = intensity
                        best_tissue_col = tissue_col
                
                if best_tissue_col is not None:
                    p_number = best_tissue_col.replace(' MaxLFQ Intensity', '')
                    # Use original protein ID with version for column name
                    original_protein = protein_version_map.get(normalized_id, protein_id)
                    column_name = f"{species}_{original_protein}"
                    protein_data.append({
                        'HOG': hog_id,
                        'Protein_Column': column_name,
                        'Intensity': max_intensity,
                        'P_Number': p_number
                    })
            else:
                proteins_not_found += 1
        
        # Log progress
        if (file_idx + 1) % 5 == 0:
            elapsed_time = time.time() - start_time
            logging.info(
                f"   Progress: {file_idx + 1}/{len(intensity_files)} species "
                f"({(file_idx + 1)/len(intensity_files)*100:.1f}%) | "
                f"Time: {elapsed_time/60:.1f} minutes"
            )
    
    logging.info(f"   Proteins found in orthogroups: {proteins_found:,}")
    logging.info(f"   Proteins not found in orthogroups: {proteins_not_found:,}")
    logging.info(f"   Mapping success rate: {proteins_found/(proteins_found+proteins_not_found)*100:.1f}%")
    
    if len(protein_data) == 0:
        logging.error(" ERROR: No protein intensity data found for heat stress proteins!")
        return None
    
    # Create initial matrix
    logging.info("4. Building initial matrix...")
    long_df = pd.DataFrame(protein_data)
    
    # Pivot to wide format
    wide_df = long_df.pivot_table(
        index='HOG',
        columns='Protein_Column', 
        values='Intensity',
        aggfunc='first'
    ).fillna(0)
    
    # Create protein→tissue mapping from our collected data
    protein_to_tissue = dict(zip(long_df['Protein_Column'], long_df['P_Number']))
    
    logging.info(f"   Initial matrix shape: {wide_df.shape}")
    logging.info(f"   Protein→tissue mappings: {len(protein_to_tissue):,}")
    
    #  Map to PO terms 
    logging.info("5. Loading tissue ontology for PO mapping...")
    
    # Load PO mappings
    tissue_df = pd.read_csv(tissue_file, sep='\t', dtype=str)
    
    #  Create p_to_po dictionary with BOTH original and normalized p-numbers
    p_to_po = {}
    for _, row in tissue_df.iterrows():
        p_num = str(row['pNumber']) if pd.notna(row['pNumber']) else None
        po_term = str(row['PO_1']) if pd.notna(row['PO_1']) else None
        
        if p_num and po_term:
            # Add original p-number
            p_to_po[p_num] = po_term
            
            norm_pnum = _normalize_pnumber(p_num)
            if norm_pnum != p_num:
                p_to_po[norm_pnum] = po_term
    
    logging.info(f"   Loaded {len(p_to_po):,} p-number → PO mappings (including normalized versions)")
    
    # Precompute column mappings
    logging.info("6. Precomputing PO mappings for columns...")
    orig_cols = [c for c in wide_df.columns if c != 'HOG']
    precomputed = [map_column(c, protein_to_tissue, p_to_po) for c in orig_cols]
    
    # Calculate mapping statistics
    unmapped_mask = [((m == o) or (isinstance(m, str) and m.startswith("UNMAPPED_"))) 
                    for o, m in zip(orig_cols, precomputed)]
    unmapped_count = sum(unmapped_mask)
    mapped_count = len(orig_cols) - unmapped_count
    
    logging.info(f"   Column mapping: {mapped_count:,} mapped, {unmapped_count:,} unmapped")
    
    # Show sample mappings
    if mapped_count > 0:
        mapped_samples = [m for o, m, u in zip(orig_cols, precomputed, unmapped_mask) 
                         if not u][:5]
        logging.info(f"   Sample mapped columns: {mapped_samples}")
    
    # Apply mappings
    col_mapping = dict(zip(orig_cols, precomputed))
    wide_df.columns = [col_mapping.get(col, col) for col in wide_df.columns]
    
    # Aggregate duplicate columns (same PO term)
    logging.info("7. Aggregating duplicate PO terms...")
    final_df = wide_df.groupby(wide_df.columns, axis=1).max()
    
    # Save final output
    logging.info("8. Saving final matrix...")
    final_df.to_csv(output_file)
    
    total_time = time.time() - start_time
    
    # Final statistics
    logging.info(f"\n => PIPELINE COMPLETED in {total_time/60:.1f} minutes!")
    logging.info(f"- Final matrix: {final_df.shape}")
    logging.info(f"   {len(final_df):,} HOGs × {len(final_df.columns):,} PO-mapped columns")
    logging.info(f" Saved to: {output_file}")
    
    # Show example of the transformation
    if len(orig_cols) > 0 and len(precomputed) > 0:
        logging.info("\n-> Example transformations:")
        for i in range(min(5, len(orig_cols))):
            if orig_cols[i] != precomputed[i]:
                logging.info(f"   {orig_cols[i]} → {precomputed[i]}")
    
    # Save a small sample for inspection
    sample_file = OUTPUT_DIR / "first_steps_sample.csv"
    if len(final_df) > 0:
        final_df.head(20).to_csv(sample_file)
        logging.info(f" Sample saved to: {sample_file}")
    
    # Save mapping statistics
    stats_file = OUTPUT_DIR / "mapping_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("=== PIPELINE MAPPING STATISTICS ===\n")
        f.write(f"Total orthogroups: {len(orthogroups):,}\n")
        f.write(f"Heat stress orthogroups: {len(heat_orthogroups):,}\n")
        f.write(f"Protein-HOG pairs created: {len(protein_to_hog):,}\n")
        f.write(f"Proteins found in intensity files: {proteins_found:,}\n")
        f.write(f"Proteins not found: {proteins_not_found:,}\n")
        f.write(f"Mapping success rate: {proteins_found/(proteins_found+proteins_not_found)*100:.1f}%\n")
        f.write(f"Final matrix shape: {final_df.shape}\n")
        f.write(f"Columns mapped to PO terms: {mapped_count:,}\n")
        f.write(f"Columns unmapped: {unmapped_count:,}\n")
        f.write(f"p_to_po dictionary size (with normalized): {len(p_to_po):,}\n")
    
    logging.info(f"-> Statistics saved to: {stats_file}")
    
    return final_df

if __name__ == "__main__":
    try:
        logging.info("=" * 60)
        logging.info("Starting first_steps.py - Heat Stress Protein Pipeline")
        logging.info("=" * 60)
        result = create_heat_stress_matrix()
        if result is not None:
            logging.info("\n" + "=" * 60)
            logging.info(" PIPELINE FINISHED SUCCESSFULLY!")
            logging.info("=" * 60)
        else:
            logging.error("\n PIPELINE FAILED")
    except Exception as e:
        logging.error(f"\n ERROR in pipeline: {str(e)}", exc_info=True)
        raise
