# Self-DistilT

This repository contains the implementation and research report for **SelfDistil-T: Self-Distilling Transformers via EMA Teachers, Layer-wise Predictive Alignment, Progressive Freezing, and LayerDrop**.

---

## 📂 Project Structure

```
.
├── SelfDistil-T.pdf       # Final project report (PhD-level analysis)
├── config.py              # Configuration file (hyperparameters, paths, settings)
├── data.py                # Dataset utilities (loading, preprocessing)
├── distillation.py        # Core distillation methods (losses, alignment)
├── ema.py                 # EMA teacher implementation
├── example.ipynb          # Example Jupyter notebook (training & evaluation demo)
├── model.py               # Transformer model definition
├── scheduler.py           # Learning rate and freezing schedule logic
├── train.py               # Training script (integrating all components)
├── utils.py               # Helper functions (logging, metrics, etc.)
```

---

## 📖 Contents

- **Report (`SelfDistil-T.pdf`)**  
  Full description of methodology, theoretical analysis, experiments, and results.

- **Source Code**  
  - `model.py`: Defines the causal decoder-only transformer architecture.  
  - `ema.py`: Maintains an exponential moving average (EMA) teacher.  
  - `distillation.py`: Implements LM loss, KD loss, and representation alignment loss.  
  - `scheduler.py`: Progressive freezing and LayerDrop schedule.  
  - `train.py`: Main entry point for training.  
  - `data.py`: Data preprocessing and loaders.  
  - `config.py`: Centralized hyperparameters and experiment settings.  
  - `utils.py`: Utilities for metrics, logging, and checkpointing.  
  - `example.ipynb`: Quick-start example to reproduce experiments.

---

## 🔧 Requirements

- Python 3.9+
- PyTorch 2.0+
- HuggingFace Transformers (optional for tokenizer)
- NumPy, SciPy, Matplotlib
- Jupyter (for running `example.ipynb`)

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

Run training with:

```bash
python train.py --config config.py
```

---

## 📊 Report Highlights

- EMA teachers reduce variance and stabilize optimization.  
- BYOL-style predictive alignment avoids collapse.  
- Progressive freezing improves efficiency and curriculum learning.  
- LayerDrop increases pruning robustness.  
- Empirical gains on WikiText-2, WikiText-103, and OpenWebText.

---

## 📚 References

Key references included in the bibliography:
- Hinton et al. (2015) — *Distilling the Knowledge in a Neural Network*
- Tarvainen & Valpola (2017) — *Mean Teacher*
- Grill et al. (2020) — *Bootstrap Your Own Latent (BYOL)*
- Fan et al. (2020) — *LayerDrop*
- Sanh et al. (2019) — *DistilBERT*
- Furlanello et al. (2018) — *Born-Again Neural Networks*

---

## ✨ Author

- **Khalil Braham** (France)  
  📧 khalilbrahem.kb@gmail.com
