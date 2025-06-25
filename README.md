# sanalyse

[![PyPI version](https://img.shields.io/pypi/v/sanalyse.svg)](https://pypi.org/project/sanalyse) [![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/sanalyse/ci.yml)](https://github.com/yourusername/sanalyse/actions) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`sanalyse` is an open-source, unified Python toolkit for **Digital Humanities**. It provides a simple, consistent API to perform core text‐analysis tasks across **English**, **Hindi**, and **Urdu**. Designed to evolve incrementally, the library currently offers a suite of **basic functionalities**, with a rich roadmap of **advanced techniques** slated for upcoming releases.

---

## 🚀 Features (v0.x)

### Core Text‐Processing
- **Normalization:** Unicode normalization, lowercasing, diacritic removal.
- **Tokenization:** Language‐aware tokenizers for English, Hindi, and Urdu.
- **Stopword Removal:** Built-in stopword lists for all three languages.
- **Stemming & Lemmatization:**  
  - English: Porter & Snowball stemmers.  
  - Hindi/Urdu: Rule‐based light stemmer. 

### Exploratory Analysis
- **Frequency Analysis:** Compute word and n-gram frequencies.
- **Concordance:** KWIC (Key Word in Context) display.
- **Collocations:** Identify bigrams and trigrams with PMI scoring.
- **Basic Readability:** Flesch–Kincaid for English; placeholder metrics for Hindi/Urdu.

### Utilities
- **Language Detection:** Fast heuristic language tagger.
- **Text I/O:** Read/write plain text, UTF-8 encoded; support for CSV/TSV corpora.
- **Batch Processing:** Apply any analyzer over a directory of text files.

---

## 🔭 Upcoming Roadmap

> _Advanced features are under active development and will land incrementally in upcoming `1.x` and `2.x` releases._


- **Named Entity Recognition:** Pretrained models for people, places, organizations.
- **Network & Graph Analysis:** Build and analyze co‐occurrence and social networks.
- **Topic Modeling:** LDA, NMF, hLDA with cross‐lingual support.
- **Stylometry & Authorship Attribution:** Feature extraction + modeling tools.
- **Sentiment & Emotion Analysis:** Transformer‐based sentiment classifiers for all supported languages.
- **Stylometry & Authorship Attribution:** Feature extraction + modeling tools.
- **OCR & Image‐to‐Text:** Integrate Tesseract pipelines.
- **Geospatial Analysis:** Map place‐name occurrences; generate time‐space visualizations.
- **Deep Learning & Embeddings:** Multilingual BERT embeddings, topic‐aware embeddings.
- **Translation & Transliteration:** Bidirectional transliteration between Devanagari, Perso‐Arabic scripts and Roman.


- **Web Based Interface to Access tools** Streamlit based tool to do plug and play interface
---

## 📦 Installation

```bash
pip install sanalyse
