<p align="center">
  <img src="https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=256&h=256&fit=crop" alt="SentinelForensicsAI Logo" width="128" style="border-radius: 24px;" />
</p>

<h1 align="center">🛡️ SentinelForensicsAI</h1>

<p align="center">
  <b>Multimodal Deepfake Detection & Explainable AI Forensic Station</b>
</p>

<p align="center">
  An enterprise-grade, research-level AI Digital Media Forensics Platform designed to verify and audit digital content. It features multimodal verification modules (Video CNN + Audio CNN) with comprehensive explainability layers (Grad-CAM), and scales to high-throughput containerized environments.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="UI Streamlit" />
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="Framework FastAPI" />
  <img src="https://img.shields.io/badge/AI-PyTorch%20%7C%20Scikit--Learn-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="AI Framework" />
  <img src="https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge" alt="Status" />
</p>

<p align="center">
  🚀 <a href="#-quick-start">Quick Start</a> • 🔍 <a href="#-key-features-at-a-glance">Key Features</a> • 📖 <a href="#-documentation-center">Documentation Docs</a> • 🏛️ <a href="#%EF%B8%8F-platform-architecture-overview">Architecture</a> • 🛡️ <a href="#-security-roadmap--known-gaps">Security Roadmap</a>
</p>

---

## 🎯 Key Features at a Glance

| 🔍 Multimodal Fusion | 🧠 Offline RAG Agent | 🎯 Explainable AI | 📈 Evaluation Metrics |
| :--- | :--- | :--- | :--- |
| **Dual-Branch Forensics**<br>Combines 2D-CNN Face crop analysis with 2D Mel-Spectrogram Audio classifiers. | **Offline Semantic Search**<br>~200 curated variations with cosine-similarity matching. | **Explainable AI (XAI)**<br>Real-time Grad-CAM heatmaps showing exact forgery points. | **Diagnostics Hub**<br>Interactive ROC-AUC curves, Confusion Matrices & dataset audits. |

---

## 📋 Table of Contents

| | |
| :--- | :--- |
| **1.** [Key Features](#-key-features-at-a-glance) | **6.** [Security Roadmap](#-security-roadmap--known-gaps) |
| **2.** [Architecture Overview](#%EF%B8%8F-platform-architecture-overview) | **7.** [Contributing](#-contributing--feedback) |
| **3.** [Documentation Docs](#-documentation-center) | **8.** [Show Your Support](#-show-your-support) |
| **4.** [Repository Structure](#-project-structure) | **9.** [Author & Contact](#-author--contact) |
| **5.** [Quickstart](#-quick-start) | |

---

## 🏛️ Platform Architecture Overview

```mermaid
graph TD
    %% Define Styles
    classDef client fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    classDef ui fill:#16a085,stroke:#1abc9c,stroke-width:2px,color:#fff;
    classDef gateway fill:#2980b9,stroke:#3498db,stroke-width:2px,color:#fff;
    classDef core fill:#d35400,stroke:#e67e22,stroke-width:2px,color:#fff;
    classDef rag fill:#8e44ad,stroke:#9b59b6,stroke-width:2px,color:#fff;
    classDef db fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;

    %% Nodes
    Client[User / Forensic Analyst]:::client
    
    subgraph UI [Streamlit Forensic Workstation UI]
        direction TB
        Tab1["Analysis Hub: Upload & Grad-CAM Visualization"]
        Tab2["Dataset & Performance Registry Dashboard"]
        Tab3["Interactive Forensic RAG Chat Interface"]
    end
    class UI ui;

    subgraph Gateway [FastAPI Gateway & Security Middleware]
        direction TB
        RoutePredict["/predict endpoint"]
        RouteAgent["/agent/query endpoint"]
        Middleware["Input Validation & Prompt Shield"]
    end
    class Gateway gateway;

    subgraph Core [Multimodal Late Fusion Engine]
        direction TB
        subgraph Video [Video Branch]
            V1["OpenCV Frame & MTCNN Face Cropping"] --> V2["ResNet-18 Deep Feature Extractor"]
        end
        subgraph Audio [Audio Branch]
            A1["Librosa Mel-Spectrogram Extraction"] --> A2["2D CNN Audio Feature Extractor"]
        end
        Video --> Fusion["Late Fusion Concatenation Layer"]
        Audio --> Fusion
        Fusion --> Classifier["BCE Sigmoid Classifier - Threshold 0.2"]
        Classifier --> XAI["Grad-CAM Heatmap Visualizer"]
    end
    class Core core;

    subgraph RAG [Offline Forensic RAG Agent]
        direction TB
        KB["200+ Curated Offline Knowledge Base Queries"]
        Vec["TF-IDF Vectorizer & Cosine Similarity"]
        Fallback["Dynamic File Parser: SQLite / metadata.csv"]
        Vec --> KB
        Vec --> Fallback
    end
    class RAG rag;

    subgraph Storage [Data & Storage Layer]
        direction TB
        SQLite[("SQLite DB: Users, Prediction Logs, Reports")]
        Disk[("Local Disk Cache: face_crops/, audio_data/, checkpoints/")]
    end
    class Storage db;

    %% Connections
    Client -->|Upload Media / Ask Query| UI
    UI -->|API POST Request| Gateway
    Gateway -->|Verify & Filter| Middleware
    Middleware -->|Process Video/Audio| Core
    Middleware -->|Execute Query| RAG
    Core -->|Log Results & Read Weights| Storage
    RAG -->|Query Stats & Metrics| Storage
```

---

## 📖 Documentation Center

To learn about specific parts of the platform infrastructure, refer to these sub-documents:

1. **[System Architecture & Design Patterns (docs/ARCHITECTURE.md)](docs/ARCHITECTURE.md)**: Deep dive into the Bounded Contexts, Bounded Domain schemas, Clean Architecture design, and Late Fusion algorithm specs.
2. **[Installation & Local/Docker Setup (docs/INSTALLATION.md)](docs/INSTALLATION.md)**: Detailed configuration guides for setting up virtual environments, installing FFmpeg, and running CPU/GPU Docker Swarms with Grafana telemetry.
3. **[Developer Guide & Code Standards (docs/DEVELOPER_GUIDE.md)](docs/DEVELOPER_GUIDE.md)**: Code formatting rules, auto-format commands (`make format`), and a step-by-step tutorial on registering new deep learning models.
4. **[Production Readiness & Security (docs/PRODUCTION_READINESS.md)](docs/PRODUCTION_READINESS.md)**: Security validations checklist, horizontal auto-scaling design, and the long-term project development roadmap.

---

## 📂 Project Structure

```text
deepfake-forensics-platform/
├── app/                        # Main Web Application & API Gateways
│   ├── routes/                 # FastAPI controllers (/predict, /agent, etc.)
│   ├── database/               # SQL Connection managers and SQLAlchemy schemas
│   ├── schemas/                # Request & Response data validation contracts
│   ├── utils/                  # Centralized logging, helpers, and crypto functions
│   ├── config/                 # Pydantic Settings loaders for Dev, Testing, and Production
│   └── main.py                 # ASGI Master Application bootstrap entry point
│
├── ai_engine/                  # Deep learning forensics algorithms and pipelines
│   ├── video/                  # Facial CNN feature extraction routines
│   ├── audio/                  # Mel Spectrogram analysis operations
│   ├── explainability/         # Heatmap generation (Grad-CAM)
│   ├── fusion/                 # Multimodal classifier fusion algorithms
│   ├── preprocessing/          # OpenCV frame slice & Librosa wave decoders
│   ├── training/               # Distributed model training parameters
│   ├── evaluation/             # Metrics collectors (Precision/Recall, F1)
│   └── datasets/               # Benchmark dataset registration schemas
│
├── storage/                    # Physical persistence layer
│   ├── uploads/                # Temporary user uploads buffer
│   └── reports/                # Output PDF/JSON forensic audits
│
├── monitoring/                 # Monitoring configurations
│   ├── prometheus/             # Scraping configurations for Prometheus server
│   └── grafana/                # Dashboards for telemetry
│
├── tests/                      # Pytest suites
└── Dockerfile                  # Multi-stage production container
```

---

## 🚀 Quick Start

### Option A: Local Execution (FastAPI + Streamlit Dashboard)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch backend API server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

3. **Launch interactive Streamlit dashboard:**
   ```bash
   streamlit run app_streamlit.py
   ```

### Option B: Docker Containerized Environment

To build and spin up the complete API platform integrated with Prometheus metrics scraping and Grafana dashboard:
```bash
docker-compose up --build -d
```

* **FastAPI API Endpoint:** [http://localhost:8000](http://localhost:8000)
* **Interactive Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Prometheus Console:** [http://localhost:9090](http://localhost:9090)
* **Grafana Dashboard:** [http://localhost:3000](http://localhost:3000)

---

## 🛡️ Security Roadmap & Known Gaps

This project is a localized forensic analysis workstation. The following security and operational items are intentionally deferred:

1. **Temporary File Lifecycles (Deferred P0):** Uploaded videos are saved in `storage/temp_uploads/` for frame extraction. Production installations should schedule a cron job or configure a FastAPI background task to prune uploads older than 10 minutes.
2. **In-Memory GPU Queueing (Deferred P1):** Heavy CNN models run inference in the request thread. For high-volume concurrent scans, integrate a task broker like Celery or RQ to handle job queues and avoid API event-loop blockages.
3. **MIME-Type Checks (Deferred P1):** Uploads are validated by filename extension. For network-facing endpoints, add Python-Magic checks to inspect file headers directly.
4. **Argon2id Upgrade (Deferred P2):** Password hashing is implemented via PBKDF2-HMAC-SHA256 with 100,000 iterations. In multi-user setups, migrate `app/utils/crypto.py` to `argon2-cffi` to maximize brute-force resistance.

---

## 🤝 Contributing & Feedback

Contributions, suggestions, and feedback are highly welcome!
* **Got suggestions or feature requests?** Feel free to open a new Issue or share your ideas.
* **Want to contribute?** Feel free to fork this repository, make your changes, and submit a Pull Request.

---

## ⭐ Show Your Support

**Love this tool? Help us grow:**
* 🌟 **Star the repository** on GitHub.
* 🐛 **Report bugs** via GitHub Issues.
* 💡 **Suggest features** or start discussions.
* 📢 **Share with others** on LinkedIn/Twitter.

---

## 👤 Author & Contact

<p align="center">
  <b>👤 Shlok Thorat</b><br>
  <i>Let's connect on LinkedIn, collaborate, and build amazing things together!</i>
</p>

<p align="center">
  <a href="mailto:shlokthorat29075@gmail.com">
    <img src="https://img.shields.io/badge/Email-shlokthorat29075%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white" alt="Email" />
  </a>
  <a href="https://github.com/shlok926">
    <img src="https://img.shields.io/badge/GitHub-%40shlok926-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub" />
  </a>
  <a href="https://www.linkedin.com/in/shlok-thorat-39916a405">
    <img src="https://img.shields.io/badge/LinkedIn-shlok--thorat--39916a405-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
</p>

<p align="center">
  Made with 🛡️ for Digital Forensics Excellence • <a href="#-sentinelforensicsai">Back to Top</a>
</p>

