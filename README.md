# Semantic Similarity-Based Detection of AI-Paraphrased Academic Text

A small Python seminar demo comparing an original passage with candidate passages using lexical overlap and sentence-level semantic similarity.

## What it does

For each candidate passage, the script reports:

- **Lexical overlap:** Jaccard overlap of the unique words in the two passages.
- **Semantic similarity:** cosine similarity between Sentence-BERT sentence embeddings.
- **Flag:** prints `FLAGGED` when semantic similarity is at least the configured threshold (`0.80` by default).

This helps demonstrate how a paraphrase can use different wording while retaining similar meaning.

## Requirements

- Python 3.10+ recommended
- Internet access on the first run to download the `all-MiniLM-L6-v2` model

## Setup

Run these commands from the project directory.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python plagiarism_demo.py
```

If PowerShell blocks activation, try Command Prompt:

```bat
venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python plagiarism_demo.py
```

The first run downloads the Sentence-BERT model and may take longer. Later runs use the downloaded model.

## Try your own passages

Open `plagiarism_demo.py` and edit the `ORIGINAL` and `SUSPECTS` values near the top of the file. Run the script again to compare the passages.

## Important limitations

- This is a **proof-of-concept comparison demo**, not a complete plagiarism detection system.
- A high semantic similarity score is a signal for review, not proof of plagiarism or misconduct.
- The `0.80` threshold is an illustrative setting, not a validated universal cutoff.
- The script does not generate AI paraphrases. Supply the candidate text yourself (for example, a paraphrase made with an AI tool).
- Results depend on the text and model; paraphrases may be missed, and unrelated text may sometimes score highly.

## Project files

- `plagiarism_demo.py` — comparison script
- `requirements.txt` — Python dependencies
