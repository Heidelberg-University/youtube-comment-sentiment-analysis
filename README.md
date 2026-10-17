# YouTube Comment Sentiment Analysis

Sentiment classification of English YouTube comments, with a web application that fetches the comments of a video and shows the predicted sentiment of each one.

| | |
|---|---|
| Course | Data Science for Text Analysis (team project) |
| University | Heidelberg University, M.Sc. Data and Computer Science |
| Semester | Winter 2022/23 |
| Type | Team project |

## Overview

- **Data:** comments are collected with the YouTube Data API, cleaned with NLTK (stop word removal, lemmatization, regular expressions) and stored in Elasticsearch.
- **Model:** a scikit-learn pipeline of a TF-IDF vectorizer and a linear support vector machine (`LinearSVC`), saved with joblib.
- **Experiments:** the team compared preprocessing pipelines and hyperparameters, and also trained on Amazon reviews. A model trained on Amazon reviews reached only about 50% accuracy on unseen YouTube comments, so the team built its own labelled dataset of about 3,000 YouTube comments.
- **Web application:** a FastAPI backend serves the trained model and the YouTube lookups, and a Svelte frontend lets users search for a video and read its comments with their predicted sentiment.

The full planning notes, data analysis and project log are in [`docs/project-log.md`](docs/project-log.md).

## Repository structure

| Path | Contents |
|---|---|
| [`modul_preparation/`](modul_preparation/) | Elasticsearch access, dataset preparation and model training |
| [`experiments/`](experiments/) | Pipeline and hyperparameter experiments, dataset statistics, Amazon review training |
| [`data/`](data/) | Collected YouTube comments, Amazon reviews, experiment results and plots |
| [`web_application/backend/`](web_application/backend/) | FastAPI service that serves the trained model and calls the YouTube API |
| [`web_application/frontend/`](web_application/frontend/) | Svelte and TypeScript user interface |
| [`training.py`](training.py) | Entry point for training the model |

## Getting started

Training needs a running Elasticsearch instance and a `modul_preparation/credentials.py` file with its user name and password. The web application needs a YouTube Data API key in `web_application/backend/API/API_KEYS.py`:

```python
YOUTUBE_API_KEY = "your-api-key"
```

Run the backend:

```sh
cd web_application/backend
pip install -r requirements.txt
python -m uvicorn main:app
```

Run the frontend in a second terminal:

```sh
cd web_application/frontend
npm install
npm run dev
```

## Tech stack

Python, scikit-learn, NLTK, pandas, Elasticsearch, FastAPI, Svelte, TypeScript, YouTube Data API.

## Team

Kushal Gaywala, Rishabh Tiwari, Jonathan Alexander Hirsch and Jakob Forstmann.
