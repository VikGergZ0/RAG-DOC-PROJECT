#RAG Document Q&A

I built this project to understand how retrieval-augmented generation (RAG)
works in practice, rather than only using tools that "chat with PDFs". It is a
small system that answers questions about a set of PDF documents and cites the
text passages it used.

The full pipeline runs locally, with no paid APIs, so I can explain and
reproduce every step on my own machine.

##What it does

The user places PDFs in a folder and builds an index once. The system splits
the documents into chunks, converts each chunk into a vector, retrieves the
chunks most similar to the question, and asks a local language model to answer
using only those chunks. If the answer is not present in the documents, the
system states this instead of generating unsupported content.

##Tools used

- **pypdf** — extracts text from the PDF files
- **sentence-transformers** (`all-MiniLM-L6-v2`) — computes embeddings locally
- **NumPy** — stores the vectors and performs the similarity search
- **Ollama** (`llama3.2`) — runs the language model locally

##How to run it

```bash
python -m venv .venv
.venv\Scripts\activate          #different on Mac/Linux: source .venv/bin/activate)
pip install pypdf sentence-transformers numpy ollama
ollama pull llama3.2

#put PDFs in docs/, then build the index
python -m rag_demo.main build docs/

#ask a question
python -m rag_demo.main ask "What is the purpose of the Foreign Subsidies Regulation?"
```

##Results

The system was evaluated on 4 EU legal PDF documents (247 chunks) using 5
question and keyword pairs:

|Metric|Result|
|---|---|
|Retrieval hit-rate (correct chunk retrieved)|4/5 (80%)|
|Answer recall (expected keyword present in answer)|4/5 (80%)|

The same evaluation can be reproduced with:

```bash
python -m rag_demo.evaluater
```

One question did not succeed: when asked which period the SANTE report covers,
the system did not retrieve the relevant passage, even though the answer
("2014 to 2016") appears explicitly in the text. This reflects a known
limitation of dense retrieval, which matches meaning well but performs weakly
on short factual or date-based queries. I kept this case in the evaluation
because it is the most informative result of the project, and it motivates
adding keyword-based search alongside the embeddings as future work.
