#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import json

def parse_po_ontology(po_file):
    #Parse the PO OBO file, ONLY keeping PO terms (no BFO).
    ontology = {}
    current_term = None
    
    print(f"Parsing ontology file: {po_file}")
    with open(po_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('[Term]'):
                if current_term and 'id' in current_term:
                    # ONLY store PO terms
                    if current_term['id'].startswith('PO:'):
                        ontology[current_term['id']] = current_term
                current_term = {'parents': []}
            
            elif current_term is not None:
                if line.startswith('id:'):
                    current_term['id'] = line.split(': ')[1]
                
                elif line.startswith('name:'):
                    current_term['name'] = line.split(': ')[1]
                
                elif line.startswith('is_a:'):
                    parent_id = line.split(': ')[1].split(' ! ')[0]
                    # ONLY track PO parents
                    if parent_id.startswith('PO:'):
                        current_term['parents'].append(parent_id)
                
                elif line.startswith('relationship: part_of'):
                    parent_id = line.split(' ')[2].split(' ! ')[0]
                    if parent_id.startswith('PO:'):
                        current_term['parents'].append(parent_id)
    
    if current_term and 'id' in current_term and current_term['id'].startswith('PO:'):
        ontology[current_term['id']] = current_term
    
    print(f"Parsed {len(ontology)} PO terms (filtered out BFO/non-PO terms)")
    return ontology

def count_real_species(data_df):
    # Count columns with at least some non-NaN values
    species_with_data = 0
    for col in data_df.columns:
        if not data_df[col].isna().all():
            species_with_data += 1
    return species_with_data

def load_tissue_tables(tissue_dir):
    #Load ALL tissue tables, counting REAL species with data.
    
    tissue_tables = {}
    
    print(f"Loading tissue tables from: {tissue_dir}")
    
    tsv_files = [f for f in os.listdir(tissue_dir) if f.endswith('.tsv')]
    print(f"Found {len(tsv_files)} TSV files")
    
    for filename in tsv_files:
        filepath = os.path.join(tissue_dir, filename)
        try:
            df = pd.read_csv(filepath, sep='\t')
            
            # Validate format
            if df.columns[0] != 'HOG' or df.iloc[0, 0] != 'METADATA':
                print(f"  !!  {filename}: Wrong format - skipping")
                continue
            
            # Extract PO term from metadata
            metadata_row = df.iloc[0]
            po_term = str(metadata_row[1]).strip() if len(df.columns) > 1 else None
            
            # Validate PO term
            if not po_term or not po_term.startswith('PO:'):
                print(f"  !!  {filename}: Invalid PO term '{po_term}' - skipping")
                continue
            
            # Get species columns
            species_cols = df.columns[1:].tolist()
            
            # Extract data
            data_df = df.iloc[1:].copy()
            for col in species_cols:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
            
            data_df = data_df.set_index('HOG')
            
            # Count REAL species with data
            real_species_count = count_real_species(data_df)
            
            # Get tissue name from filename (temporary)
            tissue_name = os.path.splitext(filename)[0].replace('_', ' ')
            
            tissue_tables[filename] = {
                'filename': filename,
                'data_df': data_df,
                'po_term': po_term,
                'tissue_name': tissue_name,
                'species_cols': species_cols,
                'real_species_count': real_species_count,
                'total_rows': data_df.shape[0]
            }
            
            print(f"  {filename}: {po_term}, {real_species_count} real species, {data_df.shape[0]} HOGs")
            
        except Exception as e:
            print(f"  ! Error loading {filename}: {e}!")
    
    print(f"\n Successfully loaded {len(tissue_tables)} tissue tables")
    return tissue_tables

def find_parent_in_hierarchy(po_term, ontology, min_species, current_species):
    
    #Find parent in hierarchy that would give ≥5 species.
    #Returns (parent_term and collapse_levels)
    
    if not po_term or po_term not in ontology:
        return po_term, 0
    
    # If already has enough species, don't move
    if current_species >= min_species:
        return po_term, 0
    
    current_term = po_term
    collapse_levels = 0
    max_levels = 10  # Safety limit
    
    while current_species < min_species and collapse_levels < max_levels:
        if current_term in ontology:
            parents = ontology[current_term].get('parents', [])
            
            # Filter to only PO parents
            po_parents = [p for p in parents if p.startswith('PO:')]
            
            if not po_parents:
                break  # No more PO parents
            
            # Move to first PO parent
            current_term = po_parents[0]
            collapse_levels += 1
            
            # Each level up adds at least 1 species (conservative estimate)
            current_species += 1
        else:
            break
    
    return current_term, collapse_levels

def process_tissues_for_grouping(tissue_tables, ontology, min_species=5):
    
    print(f"\n -> Analyzing {len(tissue_tables)} tissues for grouping...")
    
    # First pass: find optimal parent for each tissue
    tissue_info = {}
    
    for filename, info in tissue_tables.items():
        po_term = info['po_term']
        species_count = info['real_species_count']
        
        parent_term, collapse_levels = find_parent_in_hierarchy(
            po_term, ontology, min_species, species_count
        )
        
        # Get parent name
        parent_name = None
        if parent_term in ontology:
            parent_name = ontology[parent_term].get('name', parent_term)
        else:
            parent_name = parent_term
        
        tissue_info[filename] = {
            'original_po': po_term,
            'original_name': info['tissue_name'],
            'species_count': species_count,
            'parent_po': parent_term,
            'parent_name': parent_name,
            'collapse_levels': collapse_levels,
            'needs_grouping': species_count < min_species
        }
        
        status = "yes" if species_count >= min_species else "no"
        print(f"  {status} {filename}: {species_count} species → {parent_term} "
              f"({collapse_levels} levels)")
    
    # Group tissues by their parent
    parent_groups = defaultdict(list)
    for filename, info in tissue_info.items():
        parent_groups[info['parent_po']].append({
            'filename': filename,
            'info': info
        })
    
    return tissue_info, parent_groups

def merge_tissues_under_parent(filenames_info, tissue_tables, parent_po, parent_name):
    #Merge multiple tissues under a common parent.
    print(f"\n -> Merging {len(filenames_info)} tissues under {parent_name} ({parent_po})")
    
    all_data = []
    all_species = set()
    total_collapses = 0
    
    for file_info in filenames_info:
        filename = file_info['filename']
        info = tissue_tables[filename]
        collapse_info = file_info['info']
        
        # Add species
        for species in info['species_cols']:
            all_species.add(species)
        
        # Add data
        all_data.append(info['data_df'])
        
        # Track collapses
        total_collapses += collapse_info['collapse_levels']
        
        print(f"  - {filename}: {collapse_info['original_po']} → {parent_po} "
              f"({collapse_info['collapse_levels']} levels, {collapse_info['species_count']} species)")
    
    # Merge data
    if all_data:
        # Start with first dataframe
        merged_df = all_data[0].copy()
        
        # Merge others
        for i in range(1, len(all_data)):
            for col in all_data[i].columns:
                if col not in merged_df.columns:
                    merged_df[col] = all_data[i][col]
                else:
                    # Keep maximum value
                    merged_df[col] = merged_df[col].combine(all_data[i][col], max)
        
        # Add missing columns as NaN
        for species in all_species:
            if species not in merged_df.columns:
                merged_df[species] = np.nan
        
        # Count REAL species in merged data
        real_species = count_real_species(merged_df)
        
        merged_info = {
            'po_term': parent_po,
            'tissue_name': parent_name,
            'data_df': merged_df,
            'species_cols': list(all_species),
            'real_species_count': real_species,
            'source_files': [f['filename'] for f in filenames_info],
            'total_collapses': total_collapses,
            'merged': True
        }
        
        print(f"   Merged: {real_species} real species, {merged_df.shape[0]} HOGs, "
              f"{total_collapses} total collapses")
        
        return merged_info
    
    return None

def create_output_tables(processed_tissues, output_dir):
    
    print("\n" + "="*60)
    print("CREATING FINAL OUTPUT TABLES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for tissue_info in processed_tissues:
        # Create filename from tissue name
        safe_name = tissue_info['tissue_name'].replace(' ', '_').replace('/', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '._-')
        if not safe_name.endswith('.tsv'):
            safe_name = f"{safe_name}.tsv"
        
        output_path = os.path.join(output_dir, safe_name)
        
        print(f"\n-> Creating: {safe_name}")
        print(f"  Tissue: {tissue_info['tissue_name']}")
        print(f"  PO term: {tissue_info['po_term']}")
        print(f"  Real species: {tissue_info['real_species_count']}")
        
        if tissue_info.get('merged', False):
            print(f"  -> Merged from {len(tissue_info['source_files'])} files")
            print(f"  -> Total collapses: {tissue_info['total_collapses']}")
        
        # Prepare data for output
        data_df = tissue_info['data_df'].copy()
        output_df = data_df.reset_index()  # HOG becomes column
        
        # Sort columns
        if 'HOG' in output_df.columns:
            other_cols = [c for c in output_df.columns if c != 'HOG']
            output_df = output_df[['HOG'] + sorted(other_cols)]
        
        # Create metadata row
        metadata_data = {'HOG': 'METADATA'}
        for col in output_df.columns[1:]:  # Skip HOG column
            metadata_data[col] = tissue_info['po_term']
        
        metadata_df = pd.DataFrame([metadata_data])
        
        final_df = pd.concat([metadata_df, output_df], ignore_index=True)
        
        # Save
        final_df.to_csv(output_path, sep='\t', index=False)
        print(f"  ✅ Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Group tissue tables by Plant Ontology hierarchy')
    parser.add_argument('--po_file', default='/home/students/l.rodrigues/slurm-jobs/plant_ontology.obo',
                       help='Path to PO OBO file')
    parser.add_argument('--tissue_dir', default='/home/students/l.rodrigues/slurm-jobs/outputs/tissue_tables',
                       help='Directory containing tissue tables')
    parser.add_argument('--output_dir', default='/home/students/l.rodrigues/slurm-jobs/outputs/PO_overlord',
                       help='Output directory for processed files')
    parser.add_argument('--min_species', type=int, default=5,
                       help='Minimum number of species required per tissue group')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PLANT ONTOLOGY TISSUE GROUPING - CORRECTED")
    print("=" * 80)
    print(f"Input: {args.tissue_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Min species: {args.min_species}")
    print("Requirements:")
    print("1. ALL output tables ≥ {args.min_species} species")
    print("2. Merge tissues with <{args.min_species} species under common parents")
    print("3. Track collapse levels for each tissue")
    print("4. Filter out BFO terms")
    print("=" * 80)

    # Load ontology (filtered for PO only)
    print("\n  Loading Plant Ontology (PO terms only)")
    ontology = parse_po_ontology(args.po_file)
    
    if not ontology:
        print(" Error: No PO terms found in ontology")
        return
    
    #  Load ALL tissue tables
    print("\n  Loading ALL tissue tables")
    tissue_tables = load_tissue_tables(args.tissue_dir)
    
    if not tissue_tables:
        print(" Error: No tissue tables found")
        return
    
    #  Process tissues for grouping
    print("\n  Processing tissues for grouping")
    tissue_info, parent_groups = process_tissues_for_grouping(tissue_tables, ontology, args.min_species)
    
    #  Create final tissues (merge where needed)
    print("\n  Creating final tissue groups")
    final_tissues = []
    total_collapses = 0
    
    for parent_po, tissues in parent_groups.items():
        parent_name = None
        if parent_po in ontology:
            parent_name = ontology[parent_po].get('name', parent_po)
        else:
            parent_name = parent_po
        
        # Check if any tissue in this group needs grouping (<5 species)
        needs_grouping = any(t['info']['needs_grouping'] for t in tissues)
        
        if needs_grouping:
            # Merge tissues under parent
            merged_info = merge_tissues_under_parent(tissues, tissue_tables, parent_po, parent_name)
            if merged_info:
                final_tissues.append(merged_info)
                total_collapses += merged_info['total_collapses']
        else:
            # All tissues have ≥5 species - keep them separate
            for tissue in tissues:
                filename = tissue['filename']
                info = tissue_tables[filename]
                
                # Update with parent info (may have moved up hierarchy)
                final_info = {
                    'po_term': parent_po,
                    'tissue_name': parent_name,
                    'data_df': info['data_df'],
                    'species_cols': info['species_cols'],
                    'real_species_count': info['real_species_count'],
                    'source_files': [filename],
                    'total_collapses': tissue['info']['collapse_levels'],
                    'merged': False
                }
                final_tissues.append(final_info)
                total_collapses += tissue['info']['collapse_levels']
    
    #  Filter out tissues that still don't have enough species
    print(f"\n Filtering tissues (must have ≥{args.min_species} species)")
    filtered_tissues = []
    for tissue in final_tissues:
        if tissue['real_species_count'] >= args.min_species:
            filtered_tissues.append(tissue)
        else:
            print(f"   Removing: {tissue['tissue_name']} ({tissue['real_species_count']} species)")

    #  Create output tables
    print("\n Creating output tables")
    create_output_tables(filtered_tissues, args.output_dir)
    
    #  Generate summary
    print("\n" + "=" * 80)
    print(" PROCESS COMPLETED!")
    print("=" * 80)
    
    # Calculate statistics
    input_count = len(tissue_tables)
    output_count = len(filtered_tissues)
    merged_count = sum(1 for t in filtered_tissues if t.get('merged', False))
    
    print(f"\n -> FINAL STATISTICS:")
    print(f"  Input files: {input_count}")
    print(f"  Output files: {output_count}")
    print(f"  Merged tissues: {merged_count}")
    print(f"  Total collapses: {total_collapses}")
    print(f"  Removed tissues (insufficient species): {len(final_tissues) - len(filtered_tissues)}")
    
    print(f"\n-> Output directory: {args.output_dir}")
    print("  All files have:")
    print(f"  • ≥ {args.min_species} REAL species with data")
    print(f"  • Proper tissue names")
    print(f"  • No BFO terms")
    
    # Show some examples
    print(f"\n Example output files:")
    for i, tissue in enumerate(filtered_tissues[:5]):
        status = "merged" if tissue.get('merged') else "good"
        print(f"  {status} {tissue['tissue_name']}: {tissue['real_species_count']} species")

if __name__ == "__main__":
    main()