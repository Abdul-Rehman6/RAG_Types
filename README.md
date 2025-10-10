# RAG Implementation Examples

A comprehensive collection of Retrieval-Augmented Generation (RAG) implementations showcasing different approaches to enhance large language model responses with relevant context from research papers.

## Overview

This repository demonstrates three distinct RAG architectures, each with unique capabilities for retrieving and generating answers from a curated set of NLP research papers:

- **Traditional RAG** - Classic retrieve-and-generate pipeline
- **Agentic RAG** - Intelligent agent-based retrieval with multiple specialized tools
- **Corrective RAG** - Self-correcting system with web search fallback

## Research Papers Included

The implementations work with six foundational NLP research papers:

- Attention Is All You Need (Transformer architecture)
- BERT: Pre-training of Deep Bidirectional Transformers
- RoBERTa: A Robustly Optimized BERT Pretraining Approach
- ALBERT: A Lite BERT for Self-supervised Learning
- DistilBERT: A Distilled Version of BERT
- RAG: Retrieval-Augmented Generation

## Implementation Details

### 1. Traditional RAG (`1.Traditional_RAG_LG.ipynb`)

A straightforward implementation of the RAG pattern using LangChain.

**Features:**
- Document loading from PDF research papers
- Text chunking with RecursiveCharacterTextSplitter (1000 chunk size, 200 overlap)
- FAISS vector store with OpenAI embeddings
- Similarity-based retrieval (k=4)
- GPT-4o-mini for generation
- Prompt-based context injection

**Architecture:**
```
Query → Retriever → Format Docs → Prompt → LLM → Answer
```

### 2. Agentic RAG (`2.Agentic_RAG_LG.ipynb`)

An intelligent agent-based system using LangGraph with multiple specialized retrieval tools and query rewriting capabilities.

**Features:**
- Two specialized vector stores with domain-specific retrievers:
  - **Foundational Models Tool**: Transformer, BERT, RoBERTa papers
  - **Efficient Models Tool**: ALBERT, DistilBERT, RAG papers
- LangGraph workflow with state management
- Document relevance grading
- Automatic query rewriting for better retrieval
- Maximum rewrite limit to prevent infinite loops
- Agent decides which retrieval tool to use

**Architecture:**
```
Query → Agent → Tool Selection → Retrieve → Grade Relevance
                                              ↓
                                         [Relevant?]
                                         ↙        ↘
                                   Generate    Rewrite Query → Agent
```

### 3. Corrective RAG (`3.Corrective_RAG_LG.ipynb`)

A self-correcting RAG system that validates retrieved documents and falls back to web search when necessary.

**Features:**
- Document relevance validation
- Query transformation for improved retrieval
- Web search integration (Tavily) as fallback
- Minimum relevant document threshold (2 documents)
- LangGraph state machine for workflow management
- Automatic decision-making for generation vs. search

**Architecture:**
```
Query → Retrieve → Grade Documents → [≥2 Relevant?]
                        ↓                ↙        ↘
                   Filter Relevant    Generate   Transform Query
                                                       ↓
                                                  Web Search → Generate
```

## Getting Started

### Prerequisites

```bash
pip install langchain langchain-community langchain-openai langgraph
pip install faiss-cpu pymupdf python-dotenv pydantic
pip install youtube-transcript-api tavily-python
```

### Environment Setup

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### Usage

Each notebook is self-contained and can be run independently:

**Traditional RAG:**
```python
user_query = "What are different types of BERT?"
answer = main_chain.invoke(user_query)
```

**Agentic RAG:**
```python
response = graph.invoke({
    "messages": "What are different types of BERT?",
    "rewrites": 0
})
```

**Corrective RAG:**
```python
result = app.invoke({
    "question": "What is BERT and its types?"
})
```

## Key Concepts

### Vector Stores
All implementations use FAISS for efficient similarity search with OpenAI embeddings (text-embedding-3-small).

### Document Processing
- PDFs loaded using PyMuPDFLoader
- Text split into 1000-character chunks with 200-character overlap
- Maintains document metadata and source information

### Retrieval Strategy
- Similarity search with k=4 most relevant documents
- Context-aware prompting for accurate responses
- Fallback mechanisms when retrieval quality is low

### Language Models
- Primary: GPT-4o-mini for generation
- Structured output with Pydantic models for validation
- Temperature settings optimized for factual accuracy

## Architecture Comparison

| Feature | Traditional | Agentic | Corrective |
|---------|------------|---------|-----------|
| Multiple Vector Stores | ❌ | ✅ | ❌ |
| Query Rewriting | ❌ | ✅ | ✅ |
| Relevance Grading | ❌ | ✅ | ✅ |
| Web Search Fallback | ❌ | ❌ | ✅ |
| Tool Selection | ❌ | ✅ | ❌ |
| Complexity | Low | High | Medium |

## Use Cases

**Traditional RAG** - Best for:
- Simple question-answering tasks
- Known-good document collections
- Fast prototyping

**Agentic RAG** - Best for:
- Complex queries requiring specialized knowledge
- Multi-domain document collections
- When query interpretation matters

**Corrective RAG** - Best for:
- Production systems requiring high accuracy
- Cases where documents may not contain answers
- Hybrid local + web knowledge retrieval

## Project Structure

```
.
├── 1.Traditional_RAG_LG.ipynb    # Basic RAG implementation
├── 2.Agentic_RAG_LG.ipynb        # Agent-based RAG with tools
├── 3.Corrective_RAG_LG.ipynb     # Self-correcting RAG
├── Research_Papers/               # Source documents
│   ├── Attentoion_is_all_you_need.pdf
│   ├── BERT.pdf
│   ├── RoBerta.pdf
│   ├── ALBERT.pdf
│   ├── DistilBERT.pdf
│   └── RAG.pdf
├── .env                          # API keys (not in repo)
└── .gitignore
```

## Technical Stack

- **LangChain**: Framework for LLM applications
- **LangGraph**: State machine for complex workflows
- **FAISS**: Vector similarity search
- **OpenAI**: Embeddings and language models
- **Tavily**: Web search API
- **PyMuPDF**: PDF processing

## License

This project is for educational purposes demonstrating different RAG architectures.

## Acknowledgments

Research papers used in this project are the property of their respective authors and organizations.
