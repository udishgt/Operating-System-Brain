# OSB — Operating System Brain

> **Your Files. Your Knowledge. Your AI Brain.**

Turn your documents into an intelligent, searchable knowledge system — privately and locally. OSB is not a chatbot. It's a knowledge operating system built for developers, researchers, and power users.

🌐 **Live Demo:** [udishgt.github.io/Operating-System-Brain](https://udishgt.github.io/Operating-System-Brain/)

---

## What is OSB?

OSB (Operating System Brain) is a local-first AI knowledge system powered by RAG (Retrieval-Augmented Generation). Upload your documents, build a vector brain, and query your entire knowledge base using natural language — with full source traceability and zero data leakage.

All processing happens **locally on your machine**. No cloud. No tracking. No subscriptions.

---

## Features

- **Semantic Search** — Find information using natural language, not just keywords
- **Source Traceability** — Every answer cites exact document names and sections
- **Multi-Format Support** — PDF, DOCX, TXT, MD, code files, entire folders
- **Fully Private** — Nothing leaves your machine
- **Terminal-Style UI** — Cyberpunk hacker aesthetic with live system logs
- **Web Knowledge Mode** — Query without files using built-in AI knowledge
- **Real-time Logs** — Live vector index stats and system monitoring

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Architecture | RAG (Retrieval-Augmented Generation) |
| Vector DB | FAISS |
| AI Model | Gemini / Claude |
| Backend | FastAPI (Python) |
| Frontend | React + Vite |
| Embeddings | all-MiniLM-L6-v2 |

---

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.10+
- Git

### Installation

```bash
# Clone the repo
git clone https://github.com/udishgt/Operating-System-Brain.git
cd Operating-System-Brain

# Open index.html directly in browser for the frontend demo
# No install needed for the static version
```

### Full Stack Setup (Coming Soon)
```bash
# Backend
pip install fastapi uvicorn faiss-cpu sentence-transformers pypdf python-docx

# Run backend
uvicorn main:app --reload

# Frontend
npm install
npm run dev
```

---

## Screenshots

### Home — Knowledge OS Landing
![OSB Home](screenshots/home.png)

### System Interface — AI Query
![OSB System](screenshots/system.png)

### Live System Logs
![OSB Logs](screenshots/logs.png)

---

## Roadmap

- [x] Frontend UI with cyberpunk aesthetic
- [x] Interactive 3D globe with particle physics
- [x] AI Query with file upload support
- [x] System logs and vector index stats
- [ ] Python FastAPI backend
- [ ] FAISS vector store integration
- [ ] Real PDF/DOCX parsing pipeline
- [ ] User authentication
- [ ] Mobile app

---

## Author

**Udish** — [@udishgt](https://github.com/udishgt)

Built with OSB · Powered by RAG · Private by default

---

## License

MIT License — free to use, modify, and distribute.
