# 👁️ AI Cataract Diagnosis Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-success?style=flat-square)](https://streamlit.io/)
[![AI](https://img.shields.io/badge/AI-YOLOv8-red?style=flat-square)]()
[![Semantic Search](https://img.shields.io/badge/Semantic%20Search-Endee%20Inspired-orange?style=flat-square)]()

> 🚀 **AI-powered cataract detection assistant** that combines **YOLOv8 deep learning** with a **semantic search pipeline inspired by Endee vector database** for intelligent medical Q&A.

---

## 🎯 Project Overview

The **AI Cataract Diagnosis Assistant** is an end-to-end AI application designed for:

- 👁️ Early cataract screening  
- 🧠 AI-powered medical Q&A  
- 🏥 Patient awareness & education  
- 💡 Telemedicine support systems  

Users can upload eye images, receive AI-based diagnosis, and ask natural language questions about cataracts.

---

## ✨ Key Features

### 🔍 1. Cataract Detection (Computer Vision)
- Upload eye images (JPG, PNG, JPEG)
- Detect **Cataract vs Normal**
- Confidence score prediction
- Bounding box visualization

---

### 🧠 2. Semantic Medical Q&A
- Ask questions like:
  - *What are symptoms of cataract?*
  - *When should I consult a doctor?*
- Context-aware answers using semantic similarity

---

### 📊 3. Visual Feedback
- Annotated images with bounding boxes
- Easy-to-understand UI for non-technical users

---

### 💡 4. Medical Guidance
- Suggested next steps
- Patient-friendly explanations

---

### 🛡️ 5. Fallback Knowledge System
- Local medical knowledge backup
- Ensures answers even if semantic match fails

---

User Input (Image / Question)
        ↓
Streamlit Frontend
        ↓
 ┌───────────────┬────────────────┐
 │ YOLOv8 Model  │ Endee Vector DB │
 │ (Detection)   │ (Semantic Q&A)  │
 └───────────────┴────────────────┘
        ↓
Results + Recommendations
---

## 🧠 Semantic Search (Endee-Inspired Design)

This project implements a **semantic retrieval pipeline inspired by Endee vector database architecture**:

### Workflow:
1. Medical knowledge → converted into embeddings  
2. User query → transformed into embedding  
3. Cosine similarity → finds closest match  
4. Best response returned  

👉 This mimics **RAG (Retrieval-Augmented Generation)** systems used in production AI.

---

## 🛠️ Technology Stack

| Category | Tools Used |
|--------|-----------|
| Language | Python 3.10+ |
| UI | Streamlit |
| AI Model | YOLOv8 (Ultralytics) |
| NLP | Sentence Transformers |
| Search | Cosine Similarity |
| Libraries | NumPy, Scikit-learn, Pillow |

---

cataract-assistant/
│
├── app.py
├── model_utils.py
├── endee_setup.py
├── medical_data.txt
├── requirements.txt
├── README.md
├── temp/
└── docs/


---

## ⚡ Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/cataract-assistant.git
cd cataract-assistant

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
venv\Scripts\activate
# or
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
streamlit run app.py
