# SHL Conversational Assessment Recommender

A stateless, agentic FastAPI microservice that acts as an intelligent assistant for recommending SHL product catalog assessments. The system uses a semantic retrieval engine to ground its recommendations in actual catalog data and is powered by Google's Gemini LLM with strict behavioral guardrails.

## 🏗 System Architecture

The project utilizes a Retrieval-Augmented Generation (RAG) architecture. When a user sends a chat message, the system extracts a semantic query, retrieves the most relevant SHL assessments using sentence-transformers, and injects them into the Gemini LLM prompt to ensure 100% catalog compliance.

```mermaid
graph TD
    User([User Client]) -->|HTTP POST /chat| FastAPI[FastAPI Server]
    
    subgraph "Agentic Recommender System"
        FastAPI --> Agent[Agent Module]
        
        Agent -->|1. Search Query| Retriever[Semantic Retriever]
        
        subgraph "Knowledge Retrieval Base"
            Retriever -->|Encode Query| Model[all-MiniLM-L6-v2]
            Model -->|Cosine Sim| VectorStore[(Numpy Embedding Index)]
            VectorStore -->|Top-K Match| CatalogData[SHL Catalog Data]
        end
        
        Retriever -->|2. Assessment Context| Prompt[Prompt Builder]
        Agent -->|Conversation History| Prompt
        
        Prompt -->|3. System + History| Gemini[Gemini 1.5 Flash]
        Gemini -->|Structured JSON| Validator[Response Validator]
        
        Validator -->|4. Verify URLs| CatalogData
        Validator -->|5. Validated Recs| FastAPI
    end
```

## 🚀 Key Features

- **Stateless Design**: Adheres to RESTful principles. State is entirely driven by the `messages` array in the request body.
- **Semantic Grounding**: Embeds the catalog locally using `sentence-transformers` (`all-MiniLM-L6-v2`) and matches queries using vector cosine similarity.
- **Strict Schema Compliance**: Enforces exact JSON schemas (Reply, Recommendations, End-of-conversation flag) through `Pydantic` and Gemini Structured Output limits.
- **Behavioral Guardrails**: Agent explicitly refuses off-topic legal/salary inquiries, clarifies vague requests, and updates shortlists smoothly across turns.

## 🛠 Prerequisites

- Python 3.11+
- [Google Gemini API Key](https://aistudio.google.com/)

## ⚙️ Setup Instructions

1. **Environment Setup**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration**
   Open the `.env` file and insert your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_actual_key_here
   GEMINI_MODEL=gemini-1.5-flash
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

3. **Running the Server**
   ```bash
   uvicorn main:app --port 8000 --reload
   ```

## 🧪 Testing and Evaluation

The repository includes a highly robust evaluation harness `evaluate.py` which replays the 10 provided sample conversations against the API to measure correctness.

Run the evaluation:
```bash
python evaluate.py
```
*(Note: `evaluate.py` includes a 13-second rate limit throttle to respect the free-tier Gemini API limitations).*

## 🔌 API Endpoints

### `GET /health`
Verifies that the server and catalog embeddings are loaded and ready.
**Response:** `{"status": "ok"}`

### `POST /chat`
Accepts a conversation history and returns the agent's response and recommendations.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "I need an assessment for an entry-level Java developer."
    }
  ]
}
```

**Response Body:**
```json
{
  "reply": "I can help with that. Are there any specific frameworks or language skills you need to assess?",
  "recommendations": [],
  "end_of_conversation": false
}
```
