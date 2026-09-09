# Synthetic ToF Data Pipeline

This project renders synthetic time-of-flight (ToF) data to train a Vision
Transformer (ViT).

It focuses on rendering this specific dataset so the training data better
matches the target domain and helps reduce the sim-to-real performance
degradation that can occur when models trained on generic synthetic data are
deployed on real ToF measurements.

The long-term goal is an automated pipeline that prepares scenes, renders the
dataset, and processes the resulting ToF outputs with minimal manual work.
That automation is currently a work in progress.
