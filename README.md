# P15: Multi-Label Node Classification using Graph Representation Learning

## Team
| Name | Role |
|------|------|
| Harsha Koushik Teja Aila | Graph Embedding Design, Integration |
| Prashant Rathod | Dataset Acquisition, Graph Construction |
| Shaman Kanapathy | DeepWalk Implementation |
| Priyanshu Gupta | Node2Vec Implementation |
| Aditya Khurana | Classification Models & Training Pipeline |
| Sai Sagar Galli Raghu | Evaluation Metrics & Visualization |

## Project Description
Multi-label node classification on social and PPI networks using DeepWalk and Node2Vec embeddings.

## Milestones
- [x] M1: Dataset Setup and Exploration
- [x] M2: Baseline Model Implementation
- [x] M3: DeepWalk Implementation
- [x] M4: Node2Vec Implementation
- [x] M5: Multi-Label Learning and Evaluation
- [ ] M6: Analysis and Reporting

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/HarshaKoushikTeja/MultiLabel-NodeClassification.git
cd MultiLabel-NodeClassification
```

### 2. Create and activate virtual environment
```bash
#Create
python -m venv venv

#Activate — Windows
.\venv\Scripts\activate     #For CMD
OR
.\venv\Scripts\Activate.ps1     #For Powershell
OR
source venv/Scripts/activate        #For GitBash

#Activate — Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create your branch (each teammate does this once)
```bash
git checkout -b feature/your-name
git push -u origin feature/your-name
```

## Branch Naming Convention
| Branch | Owner |
|--------|-------|
| `feature/embeddings` | Harsha |
| `feature/data` | Prashant |
| `feature/deepwalk` | Shaman |
| `feature/node2vec` | Priyanshu |
| `feature/classification` | Aditya |
| `feature/evaluation` | Sagar |

## Project Structure
```
MultiLabel-NodeClassification/
├── data/
│   ├── raw/              ← downloaded datasets go here
│   └── processed/        ← cleaned graphs and labels
├── notebooks/            ← EDA and visualization notebooks
├── reports/              ← final report drafts
├── results/
│   ├── figures/          ← plots and charts
│   └── tables/           ← metric comparison tables
├── src/
│   ├── classification/   ← Aditya's classifiers
│   ├── deepwalk/         ← Shaman's DeepWalk implementation
│   ├── embeddings/       ← Harsha's interface and pipeline
│   ├── evaluation/       ← Sagar's metrics module
│   └── node2vec/         ← Priyanshu's Node2Vec implementation
├── requirements.txt
└── README.md
```