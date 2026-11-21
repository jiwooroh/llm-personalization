# LLM Personalization: Understanding User Preferences in Language Models

> Analyzing how user preferences are represented in LLM internal activations and leveraging them for personalization

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project explores how Large Language Models (LLMs) internally represent user preferences and how we can extract and leverage these representations for personalization. We analyze two complementary datasets:

1. **PRISM**: 1,500 users rating AI assistant preferences across 8 dimensions
2. **LiteraryTaste**: 60 users making 6,300 binary choices between text pairs

### Key Research Questions

1. **Where are preferences encoded?** Which layers contain the strongest user preference signals?
2. **Can we predict preferences?** Using only internal activations, can we predict user choices?
3. **Cross-domain transfer?** Do general AI preferences correlate with literary preferences?
4. **Steering effectiveness?** Can we adjust model behavior toward specific user preferences?

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/llm-personalization.git
cd llm-personalization

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download LiteraryTaste data
git clone https://github.com/mj-storytelling/LiteraryTaste.git
```

### Running Analysis

```bash
# 1. Exploratory Data Analysis (2 minutes)
python literary_taste_eda.py

# 2. Extract activations (5 minutes on CPU)
python demo_with_progress.py

# 3. Train probes (1 minute)
python literary_probe.py

# 4. Visualize results (1 minute)
python visualize_results.py
```

### Interactive Analysis

```bash
jupyter notebook notebooks/04_literary_taste_analysis.ipynb
```

## 📊 Results

### LiteraryTaste Dataset

- **60 users**, **2,000 text pairs**, **6,300 judgments**
- Nearly balanced preferences: 46.2% Text A, 45.0% Text B
- Text sources: Project Gutenberg, Reddit WritingPrompts, Poetry
- Top genres: Contemporary fiction, Social & Cultural, Crime/Mystery

### Key Findings (Expected)

1. **Middle layers (5-8) show strongest preference signals** (~70% accuracy)
2. **Genre preferences form distinct clusters** in activation space
3. **Human vs LLM text distinguishable** in late layers (~65% accuracy)

## 🔬 Methodology

### 1. Activation Extraction

```python
# Extract hidden states from each layer
outputs = model(**inputs, output_hidden_states=True)
activations = [h[0, -1, :] for h in outputs.hidden_states]  # Last token
```

### 2. Linear Probing

```python
# Train classifiers per layer
for layer in range(n_layers):
    X = activations[:, layer, :]
    y = labels  # Binary: Text A vs Text B
    
    probe = LogisticRegression()
    probe.fit(X_train, y_train)
    accuracy[layer] = probe.score(X_test, y_test)
```

### 3. Visualization

- **PCA & t-SNE**: 2D projection of activation space
- **Heatmaps**: Layer-wise activation patterns
- **Comparison plots**: Text A vs Text B differences

### 4. Steering (Planned)

```python
# Calculate genre-specific direction
literary_vec = activations[literary_users].mean(axis=0)
action_vec = activations[action_users].mean(axis=0)
steering_vector = literary_vec - action_vec

# Apply during generation
model.generate(..., steering_vector=steering_vector)
```

## 📁 Project Structure

```
llm-personalization/
├── README.md                    # This file
├── PROJECT_STATUS.md            # Detailed status (Obsidian-friendly)
├── requirements.txt             # Dependencies
│
├── LiteraryTaste/              # Dataset
│   ├── annotated_instances.csv
│   ├── preference_tasks.csv
│   └── ANALYSIS_GUIDE.md       # Complete guide
│
├── runs/                       # Outputs
│   ├── qwen7b/                 # PRISM results
│   └── literary_taste/         # LiteraryTaste results
│       ├── eda_summary.json
│       ├── demo/               # Activations
│       └── figures/            # Visualizations
│
├── notebooks/                  # Jupyter notebooks
│   └── 04_literary_taste_analysis.ipynb
│
├── src/                        # PRISM codebase
│
├── literary_taste_eda.py       # EDA script
├── demo_with_progress.py       # Activation extraction
├── literary_probe.py           # Probe training
├── visualize_results.py        # Visualization
└── compare_datasets.py         # Dataset comparison
```

## 🛠️ Technical Stack

- **Python 3.12**
- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Pre-trained models
- **scikit-learn** - Machine learning (probes)
- **Pandas/NumPy** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **Jupyter** - Interactive analysis

## 📚 Datasets

### LiteraryTaste

- **Source**: [mj-storytelling/LiteraryTaste](https://github.com/mj-storytelling/LiteraryTaste)
- **Size**: 60 users, 2,000 text pairs
- **Task**: Binary preference (Text A vs Text B)
- **Features**: Genre preferences, reading habits, demographics

### PRISM

- **Source**: [HannahKirk/prism-alignment](https://github.com/HannahKirk/prism-alignment)
- **Size**: 1,500 users
- **Task**: Rating 8 AI attributes (0-100)
- **Features**: values, fluency, factuality, safety, diversity, creativity, helpfulness, personalisation

## 🎯 Roadmap

### Completed ✅
- [x] Data collection and EDA
- [x] Analysis pipeline implementation
- [x] Activation extraction scripts
- [x] Probe training framework
- [x] Visualization tools
- [x] Documentation

### In Progress 🚧
- [ ] Full dataset activation extraction
- [ ] Layer-wise accuracy analysis
- [ ] Genre clustering experiments

### Planned 🔜
- [ ] Steering vector generation
- [ ] Text generation experiments
- [ ] Cross-domain (PRISM ↔ LiteraryTaste) analysis
- [ ] Multi-model comparison (GPT-2, LLaMA, Qwen)
- [ ] Web interface for personalized recommendations
- [ ] Paper writing

## 📖 Documentation

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current status and detailed progress (Obsidian-friendly)
- **[LiteraryTaste/ANALYSIS_GUIDE.md](LiteraryTaste/ANALYSIS_GUIDE.md)** - Complete analysis guide
- **[runs/literary_taste/README.md](runs/literary_taste/README.md)** - Experiment documentation

## 🤝 Contributing

This is currently a research project. For questions or collaboration inquiries, please open an issue.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **PRISM Dataset**: Hannah Kirk et al.
- **LiteraryTaste Dataset**: MJ Storytelling
- **TransformerLens**: Neel Nanda
- **Hugging Face**: Transformers library

## 📞 Contact

**Lucy Roh**
- Project: [GitHub](https://github.com/yourusername/llm-personalization)
- Status: Active Development

---

## 🔖 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{roh2026llmpersonalization,
  title={Understanding User Preferences in LLM Internal Representations},
  author={Roh, Lucy},
  year={2026},
  url={https://github.com/yourusername/llm-personalization}
}
```

---

**Last Updated**: 2026-11-20
**Status**: Phase 1 Complete, Phase 2 In Progress
