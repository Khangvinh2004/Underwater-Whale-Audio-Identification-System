Share Checklist For This Project

Goal
- Package this project so another person can run it on Windows with minimal setup.

What To Include In The Zip
- app.py
- Model and encoder artifacts:
  - whale_gate_classifier.pth
  - whale_species_classifier.pth
  - whale_species_label_encoder.pkl
  - diffusion_unet.pth
  - full_label_encoder.pkl
- Notebook file:
  - DiffusionWhaleClassifier (1).ipynb
- Metrics/history files you want to keep:
  - training_history_diffusion.csv
  - gate_training_history.csv
  - species_confusion_matrix.npy
  - species_confusion_matrix_normalized.npy
  - gate_confusion_matrix.npy
  - gate_confusion_matrix_normalized.npy
- Frontend app folder:
  - whales_website
- Dataset folder only if recipient needs retraining or local data exploration:
  - whalesoundsong-main
- Setup documentation:
  - RUN_ME.md
  - SHARE_CHECKLIST.md

What To Exclude From The Zip
- .venv
- __pycache__
- .ipynb_checkpoints
- whales_website/node_modules
- whales_website/dist
- Any temporary logs, cache files, or local editor folders

Recommended Zip Procedure
1. In File Explorer, copy the project folder to a clean staging folder.
2. Delete excluded folders from staging copy.
3. Right click staging folder and choose Send to > Compressed (zipped) folder.
4. Name it something clear, for example Whale_final_share_2026-04-18.zip.

Quick Validation Before Sending
1. Open terminal in the staging folder.
2. Follow RUN_ME.md exactly.
3. Confirm backend starts and frontend loads.
4. Confirm one audio prediction request works.

Optional But Strongly Recommended
- Put the project on GitHub and share the repository URL instead of a zip.
- Keep large model files in release assets or LFS if the repository gets too big.
