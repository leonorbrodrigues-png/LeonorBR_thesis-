#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse

def parse_po_ontology(po_file):
    
    ontology = {}
    current_term = None
    
    print(f"Parsing ontology file: {po_file}")
    
    # Define STRUCTURAL relationship types we care about (NO is_a)
    structural_relationship_types = {
        'part_of': [],
        'has_part': [],
        'located_in': [],
        'adjacent_to': []
    }
    
    with open(po_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('[Term]'):
                if current_term and 'id' in current_term:
                    if current_term['id'].startswith('PO:'):
                        ontology[current_term['id']] = current_term
                
                # Initialize new term with structural relationship types only
                current_term = {'id': '', 'name': '', 'parents': [], 'is_a': []}
                for rel_type in structural_relationship_types:
                    current_term[rel_type] = []
            
            elif current_term is not None:
                if line.startswith('id:'):
                    current_term['id'] = line.split(': ')[1]
                
                elif line.startswith('name:'):
                    current_term['name'] = line.split(': ')[1]
                
                elif line.startswith('is_a:'):
                    # Store is_a separately but don't use it for structural merging
                    parent_id = line.split(': ')[1].split(' ! ')[0]
                    if parent_id.startswith('PO:'):
                        current_term['is_a'].append(parent_id)
                
                elif line.startswith('relationship:'):
                    parts = line.split(' ')
                    if len(parts) >= 3:
                        rel_type = parts[1]
                        parent_id = parts[2].split(' ! ')[0]
                        
                        if rel_type in structural_relationship_types and parent_id.startswith('PO:'):
                            current_term[rel_type].append(parent_id)
                            current_term['parents'].append((rel_type, parent_id))
    
    if current_term and 'id' in current_term and current_term['id'].startswith('PO:'):
        ontology[current_term['id']] = current_term
    
    print(f"Parsed {len(ontology)} PO terms")
    
    # Count relationship types
    rel_counts = defaultdict(int)
    for term in ontology.values():
        for rel_type in structural_relationship_types:
            rel_counts[rel_type] += len(term.get(rel_type, []))
    
    print("STRUCTURAL Relationship counts (is_a excluded from merging):")
    for rel_type in structural_relationship_types:
        print(f"  {rel_type}: {rel_counts[rel_type]}")
    
    # Count is_a separately
    is_a_count = sum(len(term.get('is_a', [])) for term in ontology.values())
    print(f"  is_a (NOT used for structural merging): {is_a_count}")
    
    return ontology

def count_real_species(data_df):
    
    species_with_data = 0
    for col in data_df.columns:
        if not data_df[col].isna().all():
            species_with_data += 1
    return species_with_data

def load_tissue_tables(tissue_dir):
    
    tissue_tables = {}
    
    print(f"Loading tissue tables from: {tissue_dir}")
    
    tsv_files = [f for f in os.listdir(tissue_dir) if f.endswith('.tsv')]
    print(f"Found {len(tsv_files)} TSV files")
    
    for filename in tsv_files:
        filepath = os.path.join(tissue_dir, filename)
        try:
            df = pd.read_csv(filepath, sep='\t')
            
            if df.columns[0] != 'HOG' or df.iloc[0, 0] != 'METADATA':
                print(f"  !  {filename}: Wrong format - skipping !")
                continue
            
            metadata_row = df.iloc[0]
            po_term = str(metadata_row[1]).strip() if len(df.columns) > 1 else None
            
            if not po_term or not po_term.startswith('PO:'):
                print(f"  !  {filename}: Invalid PO term '{po_term}' - skipping !")
                continue
            
            species_cols = df.columns[1:].tolist()
            
            data_df = df.iloc[1:].copy()
            for col in species_cols:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
            
            data_df = data_df.set_index('HOG')
            
            real_species_count = count_real_species(data_df)
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
            
            print(f"  - {filename}: {po_term}, {real_species_count} species")
            
        except Exception as e:
            print(f"   Error loading {filename}: {e}")
    
    print(f"\n Successfully loaded {len(tissue_tables)} tissue tables")
    return tissue_tables

def get_all_ancestors_structural(po_term, ontology, visited=None):
    
    if visited is None:
        visited = set()
    
    if po_term not in ontology:
        return visited
    
    visited.add(po_term)
    
    for rel_type, parent in ontology[po_term].get('parents', []):
        if parent not in visited:
            get_all_ancestors_structural(parent, ontology, visited)
    
    return visited

def is_under_parent_structural(child_po, parent_po, ontology):
    
    if child_po == parent_po:
        return True
    
    if child_po not in ontology:
        return False
    
    ancestors = get_all_ancestors_structural(child_po, ontology)
    return parent_po in ancestors

def count_species_under_parent(parent_po, ontology, tissue_tables):
    
    print(f"    -> Counting species under {parent_po}")
    
    species_with_data = set()
    
    for filename, info in tissue_tables.items():
        if is_under_parent_structural(info['po_term'], parent_po, ontology):
            for species in info['species_cols']:
                if not info['data_df'][species].isna().all():
                    species_with_data.add(species)
    
    count = len(species_with_data)
    print(f"    - {parent_po} has {count} species with data")
    return count

def get_ontology_depth_structural(po_term, ontology, root_terms=None):
    
    if root_terms is None:
        root_terms = ['PO:0025131', 'PO:0009011', 'PO:0009003']
    
    if po_term in root_terms:
        return 0
    
    if po_term not in ontology:
        return 10
    
    depth = 1
    parents = ontology[po_term].get('parents', [])
    
    if not parents:
        return depth
    
    min_parent_depth = float('inf')
    for rel_type, parent in parents:
        parent_depth = get_ontology_depth_structural(parent, ontology, root_terms)
        min_parent_depth = min(min_parent_depth, parent_depth)
    
    return min_parent_depth + 1

def get_structural_parents(po_term, ontology):
    
    if po_term not in ontology:
        return []
    
    # Define priority order for STRUCTURAL relationships only
    priority_order = ['part_of', 'has_part', 'located_in', 'adjacent_to']
    
    # Group parents by relationship type
    parents_by_type = defaultdict(list)
    for rel_type, parent in ontology[po_term].get('parents', []):
        if rel_type in priority_order:  # Only include structural relationships
            parents_by_type[rel_type].append(parent)
    
    # Return parents in priority order
    prioritized_parents = []
    for rel_type in priority_order:
        if rel_type in parents_by_type:
            prioritized_parents.extend(parents_by_type[rel_type])
    
    return prioritized_parents

def find_parent_structural_only(current_po, current_species, ontology, tissue_tables, min_species):
    
    print(f"    -> Finding STRUCTURAL parent for {current_po} ({current_species} species)")
    print(f"    !  is_a relationships are IGNORED for structural merging !")
    
    if current_species >= min_species:
        parent_name = ontology.get(current_po, {}).get('name', current_po)
        print(f"    - Already has enough species at {current_po}")
        return current_po, 0, parent_name, current_po
    
    best_parent = current_po
    best_collapses = 0
    best_name = ontology.get(current_po, {}).get('name', current_po)
    best_species = current_species
    best_lca = current_po
    
    current_term = current_po
    collapse_levels = 0
    visited_terms = set([current_term])
    path = [current_term]
    path_relationships = []
    
    # Priority order for STRUCTURAL relationships only (no is_a)
    priority_order = ['part_of', 'has_part', 'located_in', 'adjacent_to']
    
    while collapse_levels < 10:
        if current_term not in ontology:
            print(f"    !  {current_term} not in ontology !")
            break
        
        structural_parents = get_structural_parents(current_term, ontology)
        
        if not structural_parents:
            print(f"    !  No STRUCTURAL parents for {current_term} !")
            print(f"    - Considered is_a parents: {ontology.get(current_term, {}).get('is_a', [])}")
            
            # If no structural parents and we haven't reached minimum, we have to stop
            # OR consider generic structural root if absolutely necessary
            if collapse_levels == 0:
                # No structural parents at all for this term
                print(f"    ! No structural relationships found for {current_term} !")
                print(f"    - Staying at original term")
                return current_po, 0, best_name, current_po
            
            break
        
        print(f"    -> Level {collapse_levels}: {current_term} has {len(structural_parents)} structural parents")
        if len(structural_parents) <= 10:
            print(f"       Parents: {structural_parents}")
        
        # Try each structural parent in priority order
        for parent in structural_parents:
            if parent in visited_terms:
                continue
            
            species_count = count_species_under_parent(parent, ontology, tissue_tables)
            
            # Check if this parent alone has enough species
            if species_count >= min_species:
                parent_name = ontology.get(parent, {}).get('name', parent)
                # Find relationship type
                rel_type = None
                for rt, p in ontology[current_term].get('parents', []):
                    if p == parent:
                        rel_type = rt
                        break
                
                relationship_path = ' → '.join(path + [parent])
                if path_relationships:
                    rel_path = ' → '.join([f"{path[i]}-{path_relationships[i]}" for i in range(len(path_relationships))] + [parent])
                    print(f"    - Found suitable parent {parent} ({rel_type}) with {species_count} species!")
                    print(f"       Path: {rel_path}")
                else:
                    print(f"    - Found suitable parent {parent} with {species_count} species!")
                return parent, collapse_levels + 1, parent_name, parent
            
            # Track best parent found so far
            elif species_count > best_species:
                best_parent = parent
                best_collapses = collapse_levels + 1
                best_name = ontology.get(parent, {}).get('name', parent)
                best_species = species_count
                best_lca = parent
                print(f"    -> New best structural parent: {parent} with {species_count} species")
        
        # If we're at root or generic terms, stop
        root_terms = {'PO:0009011', 'PO:0025131', 'PO:0009003'}
        if current_term in root_terms:
            print(f"    ! Reached root term {current_term}, stopping !")
            break
        
        # Move to the structural parent with highest priority (already sorted)
        next_parent = None
        next_rel_type = None
        
        for parent in structural_parents:
            if parent not in visited_terms:
                next_parent = parent
                # Find the relationship type
                for rel_type, p in ontology[current_term].get('parents', []):
                    if p == parent:
                        next_rel_type = rel_type
                        break
                break
        
        if next_parent:
            species_count = count_species_under_parent(next_parent, ontology, tissue_tables)
            print(f"    ⬆  Moving to highest priority structural parent: {next_parent} "
                  f"({next_rel_type}, {species_count} species)")
            current_term = next_parent
            collapse_levels += 1
            visited_terms.add(current_term)
            path.append(current_term)
            if next_rel_type:
                path_relationships.append(next_rel_type)
        else:
            print(f"    !  No unvisited structural parents for {current_term} !")
            break
    
    if best_parent != current_po:
        print(f"    - Using best structural parent found: {best_parent} ({best_species} species)")
        return best_parent, best_collapses, best_name, best_lca

    print(f"    ! No suitable structural parent found, staying at {current_po} !")
    parent_name = ontology.get(current_po, {}).get('name', current_po)
    return current_po, 0, parent_name, current_po

def merge_tissues_data(tissues_to_merge, tissue_tables, parent_po, parent_name):
    
    print(f"  - Merging {len(tissues_to_merge)} tissues under {parent_name}")
    
    all_data = []
    all_species = set()
    total_collapses = 0
    
    for tissue_info in tissues_to_merge:
        filename = tissue_info['filename']
        info = tissue_tables[filename]
        
        for species in info['species_cols']:
            all_species.add(species)
        
        all_data.append(info['data_df'])
        total_collapses += tissue_info['collapse_levels']
        
        print(f"    - {filename}: {tissue_info['original_po']} → {parent_po} "
              f"({tissue_info['collapse_levels']} levels)")
    
    # Merge data
    if all_data:
        merged_df = all_data[0].copy()
        
        for i in range(1, len(all_data)):
            for col in all_data[i].columns:
                if col not in merged_df.columns:
                    merged_df[col] = all_data[i][col]
                else:
                    merged_df[col] = merged_df[col].combine(all_data[i][col], max)
        
        # Add missing columns
        for species in all_species:
            if species not in merged_df.columns:
                merged_df[species] = np.nan
        
        real_species = count_real_species(merged_df)
        
        merged_info = {
            'po_term': parent_po,
            'tissue_name': parent_name,
            'data_df': merged_df,
            'species_cols': list(all_species),
            'real_species_count': real_species,
            'source_files': [t['filename'] for t in tissues_to_merge],
            'total_collapses': total_collapses,
            'merged': True
        }
        
        print(f"    -> Result: {real_species} species, {total_collapses} collapses")
        return merged_info
    
    return None

def find_lowest_common_ancestor_structural(tissues, ontology):
    
    print(f"    -> Finding STRUCTURAL LCA for {len(tissues)} tissues")
    
    if not tissues:
        return None
    
    # Get all ancestors for each tissue using structural relationships only
    all_ancestors = []
    for tissue in tissues:
        ancestors = get_all_ancestors_structural(tissue['original_po'], ontology)
        all_ancestors.append(ancestors)
    
    # Find common ancestors
    common = set.intersection(*[set(anc) for anc in all_ancestors])
    
    if not common:
        print(f"    !  No common structural ancestors found! !")
        
        # Try to find if they share any is_a relationships (for logging only)
        is_a_common = set()
        first_tissue = tissues[0]['original_po']
        if first_tissue in ontology:
            first_is_a = set(ontology[first_tissue].get('is_a', []))
            is_a_common.update(first_is_a)
        
        for tissue in tissues[1:]:
            po_term = tissue['original_po']
            if po_term in ontology:
                tissue_is_a = set(ontology[po_term].get('is_a', []))
                is_a_common = is_a_common.intersection(tissue_is_a)
        
        if is_a_common:
            print(f"    - Note: Tissues share is_a ancestors: {list(is_a_common)[:5]}")
            print(f"    ! But is_a relationships are NOT used for structural merging !")

        return None
    
    # Remove very generic terms
    generic_terms = {
        'PO:0009011',  # plant structure
        'PO:0025131',  # plant anatomical entity
        'PO:0009003',  # whole plant
        'PO:0000003',  # plant anatomy
    }
    
    meaningful = common - generic_terms
    
    # Score ancestors by structural relationship quality
    ancestor_scores = {}
    for ancestor in meaningful:
        if ancestor not in ontology:
            continue
        
        # Score based on structural relationship quality
        score = 0
        for tissue in tissues:
            # Check if reached via strong structural relationships
            if is_under_parent_structural(tissue['original_po'], ancestor, ontology):
                # Higher score for closer structural relationships
                depth_diff = get_ontology_depth_structural(ancestor, ontology) - \
                           get_ontology_depth_structural(tissue['original_po'], ontology)
                if depth_diff <= 2:
                    score += 20  # Close structural ancestor
                elif depth_diff <= 4:
                    score += 10  # Medium structural distance
                else:
                    score += 5   # Distant structural ancestor
        
        ancestor_scores[ancestor] = score
    
    if ancestor_scores:
        # Find ancestor with highest score
        best_ancestor = max(ancestor_scores.items(), key=lambda x: x[1])[0]
        print(f"    - Found meaningful STRUCTURAL LCA: {best_ancestor} (score: {ancestor_scores[best_ancestor]})")
        return best_ancestor
    
    # If no meaningful structural common ancestor
    print(f"    ! No meaningful structural LCA found !")
    return None

def estimate_species_count(tissues, tissue_tables):
    
    all_species = set()
    for tissue in tissues:
        info = tissue_tables[tissue['filename']]
        all_species.update(info['species_cols'])
    
    # Count species with actual data
    species_with_data = 0
    for species in all_species:
        for tissue in tissues:
            info = tissue_tables[tissue['filename']]
            if species in info['data_df'].columns and not info['data_df'][species].isna().all():
                species_with_data += 1
                break
    
    return species_with_data

def force_min_species_groups_structural(groups, tissue_tables, ontology, min_species):
    
    print(f"\n- FORCING all groups to have ≥{min_species} species")
    print(f"  Strategy: Merge groups using ONLY structural relationships")
    print(f"  !  is_a relationships are IGNORED for merging !")
    
    # Convert to list for processing
    group_list = [(po, tissues) for po, tissues in groups.items()]
    
    # Calculate species for each group
    print(f"  -> Calculating species for {len(group_list)} groups")
    group_stats = []
    for po, tissues in group_list:
        species_count = estimate_species_count(tissues, tissue_tables)
        depth = get_ontology_depth_structural(po, ontology)
        group_stats.append((po, tissues, species_count, depth))
        print(f"    - {po}: {species_count} species, structural depth {depth} ({len(tissues)} tissues)")
    
    # Sort by species count (ascending), then by depth (descending - more specific first)
    group_stats.sort(key=lambda x: (x[2], -x[3]))
    
    # Merge undersized groups
    merged_groups = {}
    merged_indices = set()
    
    print(f"\n  -> Targeting groups with <{min_species} species")
    
    for i, (po, tissues, species_count, depth) in enumerate(group_stats):
        if i in merged_indices:
            continue
            
        if species_count >= min_species:
            # Group already has enough species
            merged_groups[po] = tissues
            print(f"     {po}: Already has {species_count} species")
            continue
        
        print(f"\n    -> Processing undersized group: {po} ({species_count} species, structural depth {depth})")
        
        # Try to merge with another undersized group that shares structural relationships
        merged = False
        
        for j in range(i + 1, len(group_stats)):
            if j in merged_indices:
                continue
                
            other_po, other_tissues, other_species, other_depth = group_stats[j]
            
            # Check if these groups share any structural relationships
            combined_tissues = tissues + other_tissues
            combined_species = estimate_species_count(combined_tissues, tissue_tables)
            
            # Find STRUCTURAL LCA for merged group
            lca = find_lowest_common_ancestor_structural(combined_tissues, ontology)
            
            if lca and combined_species >= min_species:
                # Check if LCA is meaningful (not too generic)
                lca_depth = get_ontology_depth_structural(lca, ontology)
                if lca_depth >= 1:  # Not root
                    print(f"     Structurally merging {po} ({species_count} sp, depth {depth}) + "
                          f"{other_po} ({other_species} sp, depth {other_depth})")
                    print(f"       → {lca} ({combined_species} sp, structural depth {lca_depth})")
                    
                    merged_groups[lca] = combined_tissues
                    merged_indices.add(j)
                    merged = True
                    break
        
        if not merged:
            # Try to merge with any existing merged group that has structural similarity
            best_merge_key = None
            best_lca = None
            best_combined = 0
            best_depth = -1
            
            for merged_po, merged_tissues in merged_groups.items():
                combined_tissues = tissues + merged_tissues
                combined_species = estimate_species_count(combined_tissues, tissue_tables)
                
                # Find STRUCTURAL LCA
                lca = find_lowest_common_ancestor_structural(combined_tissues, ontology)
                
                if lca and combined_species >= min_species:
                    lca_depth = get_ontology_depth_structural(lca, ontology)
                    # Prefer deeper (more specific) structural LCAs
                    if lca_depth > best_depth or (lca_depth == best_depth and combined_species > best_combined):
                        best_merge_key = merged_po
                        best_lca = lca
                        best_combined = combined_species
                        best_depth = lca_depth
            
            if best_merge_key:
                # Merge with existing group
                print(f"     Structurally merging {po} ({species_count} sp) into {best_merge_key}")
                print(f"       → {best_lca} ({best_combined} sp, structural depth {best_depth})")
                
                new_tissues = merged_groups.pop(best_merge_key) + tissues
                merged_groups[best_lca] = new_tissues
                merged = True
        
        if not merged:
            # Last resort: check if tissues have ANY structural parent in common
            print(f"    !  No direct structural LCA found for {po} !")
            
            # Try to find if all tissues in this group share any structural ancestor
            if len(tissues) > 1:
                lca = find_lowest_common_ancestor_structural(tissues, ontology)
                if lca:
                    combined_species = estimate_species_count(tissues, tissue_tables)
                    if combined_species >= min_species:
                        print(f"     Using existing group's structural LCA: {lca}")
                        merged_groups[lca] = tissues
                        merged = True
        
        if not merged:
            # Can't find structural relationships - create isolated group
            print(f"    !  No structural relationships found for merging !")
            print(f"     Keeping as isolated group: {po}")
            
            merged_groups[po] = tissues
            merged = True
    
    print(f"\n  -> Group reduction: {len(group_list)} → {len(merged_groups)} structural groups")
    
    # Log final groups
    print(f"\n  -> Final structural groups:")
    for po, tissues in merged_groups.items():
        species_count = estimate_species_count(tissues, tissue_tables)
        depth = get_ontology_depth_structural(po, ontology)
        has_structural = any(tissue['collapse_levels'] > 0 for tissue in tissues)
        status = "passt" if has_structural else "isol"
        print(f"    {status} {po}: {species_count} species, depth {depth} ({len(tissues)} tissues)")
    
    return merged_groups

def process_all_tissues_structural(tissue_tables, ontology, min_species):
    
    print(f"\n => Initial structural grouping of {len(tissue_tables)} tissues")
    print(f"   Strategy: Use ONLY structural relationships")
    print(f"   Priority: part_of > has_part > located_in > adjacent_to")
    print(f"   !  is_a relationships are IGNORED for structural merging !")
    
    tissue_list = []
    
    for filename, info in tissue_tables.items():
        po_term = info['po_term']
        species_count = info['real_species_count']
        
        print(f"\n  Processing {filename}:")
        print(f"    Original: {po_term} ({species_count} species)")
        
        # Find parent using STRUCTURAL relationships only (no is_a)
        parent_po, collapse_levels, parent_name, lca = find_parent_structural_only(
            po_term, species_count, ontology, tissue_tables, min_species
        )
        
        # Check if any structural relationships were found
        has_structural_parent = (parent_po != po_term or collapse_levels > 0)
        
        tissue_list.append({
            'filename': filename,
            'original_po': po_term,
            'original_name': info['tissue_name'],
            'species_count': species_count,
            'parent_po': parent_po,
            'parent_name': parent_name,
            'collapse_levels': collapse_levels,
            'original_species': species_count,
            'lca': lca,
            'has_structural': has_structural_parent
        })
        
        if species_count >= min_species:
            status = "good"
            reason = "Enough species"
        elif has_structural_parent:
            status = "passt"
            reason = f"Structural parent found ({collapse_levels} levels)"
        else:
            status = "isol"
            reason = "No structural parent, staying put"
        
        print(f"  {status} {filename}: {species_count} species → {parent_po} ({reason})")
    
    # Group by structural LCA
    print(f"\n => Grouping by structural LCA terms")
    groups = defaultdict(list)
    for tissue in tissue_list:
        # Use structural LCA if available, otherwise original PO
        group_key = tissue.get('lca', tissue['parent_po'])
        groups[group_key].append(tissue)
    
    # Count groups with/without structural relationships
    structural_groups = sum(1 for tissues in groups.values() 
                          if any(t.get('has_structural', False) for t in tissues))
    isolated_groups = len(groups) - structural_groups
    
    print(f"Created {len(groups)} groups")
    print(f"  - {structural_groups} groups with structural relationships")
    print(f"  - {isolated_groups} groups without structural relationships (isolated)")
    
    for po, tissues in groups.items():
        total_species = sum(t['species_count'] for t in tissues)
        depth = get_ontology_depth_structural(po, ontology)
        has_struct = any(t.get('has_structural', False) for t in tissues)
        status = "passt" if has_struct else "isol"
        print(f"  {status} {po}: {len(tissues)} tissues, {total_species} species, depth {depth}")
    
    # ENSURE ALL GROUPS HAVE ≥min_species using structural relationships only
    final_groups = force_min_species_groups_structural(groups, tissue_tables, ontology, min_species)
    
    # Create final tissues
    print(f"\n => Creating final tissues")
    final_tissues = []
    
    for parent_po, tissues in final_groups.items():
        print(f"\n  Processing group {parent_po} ({len(tissues)} tissues):")
        parent_name = ontology.get(parent_po, {}).get('name', parent_po)
        
        # Check if this is a structural group
        has_structural = any(t.get('has_structural', False) for t in tissues)
        
        if len(tissues) == 1:
            # Single tissue
            tissue = tissues[0]
            info = tissue_tables[tissue['filename']]
            
            status = "passt" if tissue.get('has_structural', False) else "isol"
            
            final_info = {
                'po_term': parent_po,
                'tissue_name': parent_name,
                'data_df': info['data_df'],
                'species_cols': info['species_cols'],
                'real_species_count': info['real_species_count'],
                'source_files': [tissue['filename']],
                'total_collapses': tissue['collapse_levels'],
                'merged': False,
                'structural': tissue.get('has_structural', False)
            }
            final_tissues.append(final_info)
            print(f"    {status} Single tissue: {tissue['original_name']} → {parent_po}")
        else:
            # Multiple tissues - merge them
            merged_info = merge_tissues_data(tissues, tissue_tables, parent_po, parent_name)
            if merged_info:
                merged_info['structural'] = has_structural
                final_tissues.append(merged_info)
    
    return final_tissues

def create_output_tables(processed_tissues, output_dir):
    
    print("\n" + "="*60)
    print("CREATING FINAL OUTPUT TABLES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for tissue_info in processed_tissues:
        safe_name = tissue_info['tissue_name'].replace(' ', '_').replace('/', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '._-')
        if not safe_name.endswith('.tsv'):
            safe_name = f"{safe_name}.tsv"
        
        output_path = os.path.join(output_dir, safe_name)
        
        print(f"\n Creating: {safe_name}")
        print(f"  Tissue: {tissue_info['tissue_name']}")
        print(f"  PO term: {tissue_info['po_term']}")
        print(f"  Species: {tissue_info['real_species_count']}")
        
        if tissue_info.get('structural', False):
            print(f"  - Structural relationship used")
        
        if tissue_info.get('merged', False):
            print(f"  - Merged from {len(tissue_info['source_files'])} files")
            print(f"  - Total collapses: {tissue_info['total_collapses']}")
        
        # Prepare data
        data_df = tissue_info['data_df'].copy()
        output_df = data_df.reset_index()
        
        if 'HOG' in output_df.columns:
            other_cols = [c for c in output_df.columns if c != 'HOG']
            output_df = output_df[['HOG'] + sorted(other_cols)]
        
        # Metadata
        metadata_data = {'HOG': 'METADATA'}
        for col in output_df.columns[1:]:
            metadata_data[col] = tissue_info['po_term']
        
        metadata_df = pd.DataFrame([metadata_data])
        final_df = pd.concat([metadata_df, output_df], ignore_index=True)
        
        # Save
        final_df.to_csv(output_path, sep='\t', index=False)
        print(f"   Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Merge tissue tables using ONLY structural Plant Ontology relationships (NO is_a)'
    )
    parser.add_argument('--po_file', default='/home/students/l.rodrigues/slurm-jobs/plant_ontology.obo',
                       help='Path to PO OBO file')
    parser.add_argument('--tissue_dir', default='/home/students/l.rodrigues/slurm-jobs/outputs/tissue_tables',
                       help='Directory containing tissue tables')
    parser.add_argument('--output_dir', default='/home/students/l.rodrigues/slurm-jobs/outputs/PO_structural_only',
                       help='Output directory for processed files')
    parser.add_argument('--min_species', type=int, default=5,
                       help='Minimum number of species required per tissue group')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PLANT ONTOLOGY TISSUE GROUPING - STRUCTURAL RELATIONSHIPS ONLY")
    print("=" * 80)
    print(f"Input: {args.tissue_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Min species: {args.min_species}")
    print("Strategy: Use ONLY structural relationships (NO is_a)")
    print("Priority: part_of > has_part > located_in > adjacent_to")
    print("Goal: Force ≥5 species using ONLY biologically meaningful structural relationships")
    print("=" * 80)
    
    # Load ontology
    print("\n📖 Loading Plant Ontology")
    ontology = parse_po_ontology(args.po_file)
    
    if not ontology:
        print(" Error: No PO terms found")
        return
    
    # Load tissue tables
    print("\n Loading tissue tables")
    tissue_tables = load_tissue_tables(args.tissue_dir)
    
    if not tissue_tables:
        print(" Error: No tissue tables found")
        return
    
    # Process tissues with structural relationships only
    print("\n Processing tissues with STRUCTURAL relationships only")
    final_tissues = process_all_tissues_structural(tissue_tables, ontology, args.min_species)
    
    # Create output tables
    print("\n Creating output tables")
    create_output_tables(final_tissues, args.output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print(" PROCESS COMPLETED!")
    print("=" * 80)
    
    # Statistics
    input_count = len(tissue_tables)
    output_count = len(final_tissues)
    merged_count = sum(1 for t in final_tissues if t.get('merged', False))
    structural_count = sum(1 for t in final_tissues if t.get('structural', False))
    total_collapses = sum(t.get('total_collapses', 0) for t in final_tissues)
    
    print(f"\n-> FINAL STATISTICS:")
    print(f"  Input files: {input_count}")
    print(f"  Output files: {output_count}")
    print(f"  Merged tissues: {merged_count}")
    print(f"  Tissues using structural relationships: {structural_count}")
    print(f"  Isolated tissues (no structural relationships): {output_count - structural_count}")
    print(f"  Total collapses: {total_collapses}")
    
    # Structural relationship analysis
    print(f"\n-> STRUCTURAL RELATIONSHIP ANALYSIS:")
    rel_counts = defaultdict(int)
    for tissue in final_tissues:
        if tissue.get('structural', False):
            po_term = tissue['po_term']
            if po_term in ontology:
                term = ontology[po_term]
                for rel_type in ['part_of', 'has_part', 'located_in', 'adjacent_to']:
                    if rel_type in term and term[rel_type]:
                        rel_counts[rel_type] += 1
                        break  # Count only primary relationship
    
    if rel_counts:
        print("  Final tissues by primary structural relationship:")
        for rel_type in ['part_of', 'has_part', 'located_in', 'adjacent_to']:
            count = rel_counts.get(rel_type, 0)
            if count > 0:
                percentage = (count / structural_count * 100) if structural_count > 0 else 0
                print(f"    - {rel_type}: {count} ({percentage:.1f}% of structural tissues)")
    else:
        print("  No structural relationships used in final tissues")
    
    # Check all have enough species
    insufficient = [t for t in final_tissues if t['real_species_count'] < args.min_species]
    if insufficient:
        print(f"\n!  WARNING: {len(insufficient)} tissues still have <{args.min_species} species:")
        for t in insufficient:
            structural_status = "struct" if t.get('structural', False) else "good"
            print(f"  {structural_status} {t['tissue_name']}: {t['real_species_count']} species")
    else:
        print(f"\n SUCCESS: All {output_count} output tissues have ≥{args.min_species} species!")
    
    print(f"\n Output directory: {args.output_dir}")
    
    # Show examples
    print(f"\n Example output files (first 10):")
    for i, tissue in enumerate(final_tissues[:10]):
        if tissue.get('merged', False):
            status = "bad"
        elif tissue.get('structural', False):
            status = "struct"
        else:
            status = "good"
        
        print(f"  {status} {tissue['tissue_name']}: {tissue['real_species_count']} species")

if __name__ == "__main__":
    main()