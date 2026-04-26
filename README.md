# PharyngitisAI

A deep learning project I built to detect pharyngitis from throat images, combined with an LLM to generate human-readable explanations of the results.

Live at → **[pharyngitisai.online](https://pharyngitisai.online)**

---

## Background

This started as my final year project at Telkom University — I trained a ResNet50 model on endoscopic throat images to classify pharyngitis vs normal. After graduating, I wanted to turn it into something actually usable, so I wrapped it in a FastAPI backend, added an LLM layer for explanations, and deployed the whole thing.

---

## What it does

Upload a throat photo, and the app will:
1. Run it through a ResNet50 model to classify as **Normal** or **Pharyngitis**
2. Pass the result to a Groq LLM (LLaMA 3.3 70B) to generate a plain-language explanation
3. Return everything as a JSON response — or show it nicely on the web UI

Supports English and Indonesian explanations.

---

## Stack

- **Model** — ResNet50 trained with TensorFlow/Keras
- **Backend** — FastAPI + Python
- **LLM** — Groq API (LLaMA 3.3 70B)
- **Frontend** — Vanilla HTML/CSS/JS
- **Hosting** — Railway (API) + GitHub Pages (frontend)
- **Domain** — pharyngitisai.online

---

## API

```
POST /predict
  - file: image (JPG, PNG, WEBP)
  - language: "en" or "id"

Response:
{
  "prediction": "Pharyngitis",
  "confidence": "92.5%",
  "explanation": "..."
}
```

Full docs at [api.pharyngitisai.online/docs](https://api.pharyngitisai.online/docs)

---

## Run it locally

```bash
git clone https://github.com/rifqzzan/pharyngitis-detection-api.git
cd pharyngitis-detection-api

python -m venv env
.\env\Scripts\activate

pip install -r requirements.txt

# add your GROQ_API_KEY to a .env file
# note: model.keras is not in this repo, download separately

python -m uvicorn main:app --reload
```

---

## Heads up

The model was trained on endoscopic images, so it works best with clear throat photos taken close up. Regular phone photos might give less accurate results — something I'm planning to improve.

This is also a screening tool, not a diagnostic one. Always consult a doctor.

---

Rifqi Izza — [LinkedIn](https://www.linkedin.com/in/rifqi-izza-nuradli/) · [pharyngitisai.online](https://pharyngitisai.online)