# Comparative Heat Stress Proteomics Pipeline

Cross-species comparative analysis of heat stress response proteins across 18 plant species using orthology inference and tissue ontology mapping.

## Pipeline Scripts

**step1_orthology_po_mapping.py** - Integrates protein intensity data with heat stress orthogroups and maps to Plant Ontology terms.

**step2_normalization_transform.py** - Applies total-sum normalization and log2 transformation to intensity matrices.

**step3_tissue_table_generator.py** - Generates separate tissue-specific expression tables with metadata formatting.

## Tissue Merging Approaches (Experimental)

**merge_approach1_statistical.py** - Hierarchical merging prioritizing statistical threshold (≥5 species per group).

**merge_approach2_biological_lca.py** - Evidence-based merging using Lowest Common Ancestor with biological coherence constraints.

**merge_approach3_structural_enforced.py** - structural Relationship-prioritized merging with two-phase group enforcement.

**merge_approach3_structural_strict.py** - Strict structural relationship merging excluding classificatory (is_a) relationships.

## Visualization

**visualizations_python.ipynb** - Python-based plots for data exploration and quality control analysis including heatmaps, PCA plots, and comparative figures.

**visualizations_R_code.ipynb** - R-based visualizations including UpSet plot, stacked bar plots, and species trees. Produced on Google Collab

## Input Data
- OrthoFinder results 
- MaxLFQ intensity files (.tsv per species)
- Plant Ontology mapping file (ENB_TissueOntology_long.tsv and PO.obo)
