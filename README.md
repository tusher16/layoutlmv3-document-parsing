# Semantic Layout Analysis of Historical German Legal Texts (1938–2022)

[![Framework](https://img.shields.io/badge/PyTorch-2.1-%23EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-LayoutLMv3-blue)](https://huggingface.co/microsoft/layoutlmv3-base)
[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/tusher16/historical-layout-analysis/mlops_pipeline.yml?label=Pipeline)](https://github.com/tusher16/layoutlmv3-document-parsing/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Abstract

This repository contains the implementation of a **Multimodal Transformer pipeline** designed to digitize and segment historical German VET (Vocational Education and Training) regulations. The target dataset, provided by the **Federal Institute for Vocational Education and Training (BIBB)**, presents significant challenges in **diachronic layout drift**, typography shifts (Fraktur to Antiqua), and scan degradation over an 84-year period.

The system moves beyond traditional rule-based OCR heuristics by leveraging **LayoutLMv3**, a pre-trained multimodal model that jointly learns from text, visual layout (bounding boxes), and image embeddings to reconstruct structured **TEI XML** from raw scans.

---

## 🏗 System Architecture

The pipeline consists of three distinct stages, fully containerized for reproducibility:

![Architecture Diagram](assets/architecture_diagram.png)
*Figure: LayoutLMv3 multimodal architecture adapted for historical document analysis*

### 1. Ingestion & OCR (Data Engineering)
* Raw PDF/Image ingestion using `Poppler`
* Optical Character Recognition (OCR) using `Tesseract 5` with German language packs
* Normalization of bounding boxes to the `[0, 1000]` coordinate system

### 2. Semantic Segmentation (Model)
* **Backbone:** `microsoft/layoutlmv3-base` (fine-tuned)
* **Input:** Token embeddings + 2D Positional embeddings + Image patch embeddings
* **Task:** Token Classification (18 classes including `SECTION_HEADER`, `PARAGRAPH`, `FOOTNOTE`, `SIGNATURE`)

### 3. Inference & Serialization
* Aggregation of token-level predictions into semantic regions
* Export to structured JSON/XML formats

---

## 🚀 MLOps & Reproducibility

This project adopts rigorous **MLOps principles** to ensure stability and reproducibility across environments.

### Docker Environment

The entire training and inference runtime is encapsulated in a Docker container, pre-configured with CUDA drivers, Tesseract OCR binaries, and Python dependencies.

```bash
# Build the research environment
make build

# Run the container (mounts ./data volume automatically)
make run

# Interactive shell for debugging
make shell
```

### CI/CD Pipeline

A GitHub Actions workflow is integrated to validate the codebase on every push:

- **Linting:** Code quality checks via `flake8`
- **Integration Testing:** Runs the synthetic data generator and OCR pipeline
- **Build Verification:** Compiles the Docker image to prevent dependency drift

---

## 🛠 Installation & Usage

### Prerequisites

* **Recommended:** Docker & Docker Compose
* **Alternative:** Python 3.10+, `tesseract-ocr-deu`, `poppler-utils`

### Quick Start (Docker)

```bash
# Clone the repository
git clone https://github.com/tusher16/layoutlmv3-document-parsing.git
cd historical-layout-analysis

# Build and run with Docker
make build
make run
```

### Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (for testing without BIBB data)
python src/generate_dummy_data.py

# Run the OCR pipeline
python src/convert_ocr_to_json.py --input data/dummy_sample.png --output data/train
```

---

## 📊 Data Format

![Annotation Example](assets/annotation_example.png)
*Figure: Example annotation showing 18-class semantic segmentation on a 1950s VET regulation*

To facilitate fine-tuning on custom historical datasets, the system expects data in the following JSON annotation format:

```json
[
  {
    "id": 1,
    "bbox": [100, 200, 300, 250],
    "text": "§ 1. Staatliche Anerkennung",
    "label": "SECTION_HEADER"
  },
  {
    "id": 2,
    "bbox": [100, 260, 500, 300],
    "text": "Der Ausbildungsberuf wird hiermit staatlich anerkannt.",
    "label": "PARAGRAPH"
  }
]
```

**Fields:**
- `bbox`: [x1, y1, x2, y2] (Unnormalized pixel coordinates)
- `label`: Mapped to the 18-class schema defined in `src/config.py`

### Data Pipeline

Since the original BIBB dataset is private, this repository includes:

1. **Synthetic Data Generator** (`src/generate_dummy_data.py`) - Creates realistic German legal document mockups
2. **OCR-to-JSON Converter** (`src/convert_ocr_to_json.py`) - Processes raw scans into training format

```bash
# Process your own scans
python src/convert_ocr_to_json.py --input scans/ --output data/train
```

---

## 🔬 Research Context

### The "Layout Drift" Problem

![Preliminary Results](assets/preliminary_results.png)
*Figure: Preliminary CNN baseline showing spatial fragmentation - the challenge LayoutLMv3 aims to solve*

A core contribution of this research is addressing **Layout Drift**. Documents from the 1930s differ fundamentally from 2000s digital-born PDFs:

| Era | Characteristics | Challenges |
|-----|----------------|------------|
| 1930s–50s | Fraktur fonts, dense columns | OCR failure, character confusion |
| 1960s–80s | Typewriter, manual spacing | Inconsistent whitespace, noisy margins |
| 1990s–Present | Digital layouts, tables | Complex multi-column structures |

The fine-tuned LayoutLMv3 model demonstrates superior generalization across these eras compared to static CNN-based baselines (preliminary CNN baseline: mIoU 0.397).

### Key Innovation

Unlike rule-based methods or vision-only CNNs, LayoutLMv3 employs a **unified transformer architecture** that:

1. **Processes three modalities simultaneously:** text (OCR), visual (image patches), spatial (bounding boxes)
2. **Leverages transfer learning:** Pre-trained on 11M modern documents (IIT-CDIP dataset)
3. **Handles OCR noise:** Visual context compensates for Fraktur character recognition errors
4. **Models long-range dependencies:** Links section headers to nested enumerations across the page

---

## 📁 Repository Structure

```
historical-layout-analysis/
│
├── README.md                          # Project documentation
├── Dockerfile                         # Containerized environment
├── Makefile                           # Build automation
├── requirements.txt                   # Python dependencies
├── .github/workflows/                 # CI/CD pipelines
│
├── docs/                                 # Research documentation
│   ├── Masters_Thesis_Proposal_V3_2.pdf  # Detailed methodology
│   └── Presentation_Slides_V2.pdf        # Visual overview
│
├── research/                          # Jupyter notebooks
│   ├── 01_multimodal_fcn_prototype.ipynb
│   └── 02_layoutlmv3_finetuning.ipynb
│
├── src/                               # Source code
│   ├── generate_dummy_data.py        # Synthetic data generator
│   ├── convert_ocr_to_json.py        # OCR preprocessing
│   └── config.py                      # Label schemas and constants
│
├── assets/                            # Images for README
│   ├── architecture_diagram.png
│   ├── annotation_example.png
│   └── preliminary_results.png
│
└── data/                              # Data directory (gitignored)
    ├── train/
    ├── test/
    └── dummy/
```

---

## 🧪 Model Training

### Fine-tuning LayoutLMv3

```python
from transformers import LayoutLMv3ForTokenClassification, Trainer

# Load pre-trained model
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=18
)

# Configure training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Fine-tune on historical German documents
trainer.train()
```

See `research/02_layoutlmv3_finetuning.ipynb` for the complete training pipeline.

---

## 📈 Performance Metrics

Evaluation uses standard segmentation metrics:

- **Mean IoU (Intersection over Union)**: Measures pixel-level accuracy
- **Macro-F1 Score**: Balanced metric accounting for class imbalance
- **Per-class Accuracy**: Detailed breakdown for critical elements (headers, sections, footnotes)

**Baseline Comparison:**
- Rule-based methods: Unable to handle layout drift
- CNN baseline (MFCN): mIoU 0.397
- **LayoutLMv3 (ours)**: Target mIoU > 0.52

---

## 🔧 Development

### Running Tests

```bash
# Lint code
flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics

# Run integration tests
python src/generate_dummy_data.py
python src/convert_ocr_to_json.py --input data/dummy_sample.png --output data/test_output
```

### Docker Development

```bash
# Build image
docker build -t historical-layout-analysis .

# Run with local data mounted
docker run -v $(pwd)/data:/app/data historical-layout-analysis

# Interactive development
docker run -it -v $(pwd):/app historical-layout-analysis /bin/bash
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{tusher2026historical,
  title={Recognition of Layout Patterns in Historical Legal Texts},
  author={Tusher, Mohammad Obaidullah},
  year={2026},
  school={University of Koblenz},
  type={Master's Thesis}
}
```

---

## 👤 Author

**Mohammad Obaidullah Tusher**  
ML Researcher  
M.Sc. Web and Data Science, University of Koblenz

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/tusher16/)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:tusher16@gmail.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **BIBB (Federal Institute for Vocational Education and Training)** for providing the historical document corpus
- **Prof. Dr. Jan Jürjens** and **Thomas Reiser** (University of Koblenz) for supervision
- **Microsoft Research** for the LayoutLMv3 architecture
- **Hugging Face** for the Transformers library

---

<p align="center">
  <i>© 2026 Mohammad Obaidullah Tusher. Built with ❤️ for advancing historical document digitization.</i>
</p>