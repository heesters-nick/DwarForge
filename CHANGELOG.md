# Changelog

All notable changes to DwarForge will be documented in this file.

## [0.1.0] - 2025-10-27

### Added

#### Core Pipeline
- Automated preprocessing and 4x4 pixel rebinning of survey images
- Automated detection pipeline for dwarf galaxy candidates using MTObjects (MTO)
- Multi-band image processing and combination module for cross-matching detections
- Deep learning classification using fine-tuned Zoobot model

#### Data Integration
- Integration with UNIONS survey data via CANFAR VOSpace
- Support for CADC X509/SSL certificate authentication
- Automatic tile downloading with dedicated worker thread
- Multi-band filter support: cfis-u, whigs-g, cfis_lsb-r, ps-i, wishes-z

#### Processing Features
- Parallel processing with configurable number of worker cores
- Queue-based task management for efficient processing
- False detection filtering through multi-band cross-matching
- RGB cutout generation (256x256 pixels) from multi-band data
- HDF5 file format for storing cutouts and metadata

#### Configuration System
- YAML-based configuration files for detection, combination, and aggregation
- Support for multiple computing environments (local, CANFAR, Narval)
- Flexible input modes:
  - Individual tile numbers
  - RA/Dec coordinates
  - DataFrame input
  - All available tiles
- KD-tree spatial indexing with build/update options for efficient tile queries

#### Scripts
- `detection.py` - Main detection pipeline script
- `combination.py` - Multi-band cross-matching and combination
- `inference.py` - Deep learning model inference
- `h5_aggregation.py` - Aggregate cutouts from different image tiles for ML/DL training
- `file_transfer.py` - Upload utilities for Google Drive integration

#### Documentation
- Installation instructions with pip editable mode
- CANFAR authentication and setup guide
- Quickstart workflow for running the pipeline
- Configuration file documentation
- Citation information for GOBLIN catalog and dependencies

### Scientific Validation
- Pipeline used to produce GOBLIN catalog (43,000 dwarf galaxy candidates)
- Methodology published in Heesters et al. 2025, A&A, 699, A232
- Results available in CDS catalog J/A+A/699/A232

### Dependencies
- Python 3.11.5+ support
- MTObjects integration for classical detection
- Zoobot fine-tuned model for classification
- HDF5 development libraries
- CANFAR VOSpace client tools
- If working with UNIONS data: UNIONS collaboration membership + CANFAR account

---

## Notes

This is the initial public release of DwarForge, providing a complete pipeline for detecting and classifying dwarf galaxy candidates in wide-field imaging surveys. The pipeline has been tuned specifically for UNIONS survey data but can be adapted for other surveys.

### Citation

If you use DwarForge or results produced with it, please cite:
- The GOBLIN catalog paper: Heesters et al. 2025, A&A, 699, A232
- This repository: https://github.com/heesters-nick/DwarForge

[0.1.0]: https://github.com/heesters-nick/DwarForge/releases/tag/v0.1.0
