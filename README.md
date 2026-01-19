# Meeting Extraction RAG System

A simple RAG (Retrieval-Augmented Generation) system for extracting public comments and public hearings from meeting transcripts.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file and add your OpenAI API Key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

```bash
python main.py
```

## Project Structure

- `data_loader.py`: Data loading and preprocessing
- `vector_store.py`: Vector database creation and retrieval
- `rag.py`: RAG query implementation
- `main.py`: Main program entry point
