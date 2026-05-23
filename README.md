# Autonomous Agentic RAG: An Intelligent Knowledge Retrieval Agent

**Autonomous Agentic RAG** is a sophisticated implementation of Retrieval-Augmented Generation that utilizes an "agentic" workflow to decide between local knowledge retrieval and external web searching. By leveraging frameworks like **Agno**, **LangChain**, and **Google Gemini**, the system creates a reasoning loop to provide accurate, context-aware answers grounded in verifiable data.

## 🚀 Key Features
*   **Autonomous Source Selection**: Dynamically decides between local document memory and live web searching based on contextual logic.
*   **Query Reformulation**: Features a specialized **Query Rewriter agent** that refines vague user questions into precise, search-friendly queries for better retrieval.
*   **Hybrid Retrieval**: Seamlessly blends vector database retrieval with real-time web data fetch via **ExaTools** when local knowledge is insufficient.
*   **High-Precision Search**: Uses a similarity threshold (default 0.7) to ensure only mathematically relevant context is used for generating answers.
*   **Transparent Citations**: Automatically generates citations linking specific facts back to original document chunks or web sources for maximum transparency.
*   **Multi-Format Ingestion**: Supports uploading and processing of **PDF, TXT, and DOCX** files, as well as direct **Web URLs**.
*   **Deep Reasoning**: Utilizes the **Gemini 2.0 Flash-Thinking** model to perform deeper reasoning over retrieved context before producing a final response.

## 🛠️ Tech Stack
*   **Core Logic**: Python 3.10+
*   **Orchestration**: Agno (formerly Phidata) Agent Framework
*   **Models**: Google Gemini 2.0 Flash, Flash-Thinking, and Experimental 1206
*   **Embeddings**: Gemini `text-embedding-004`
*   **Vector Databases**: **ChromaDB** for persistent storage and **FAISS** for high-performance similarity search
*   **Web Search**: ExaTools API
*   **UI Framework**: Streamlit

## 🧠 System Architecture
The system is divided into three specialized agents, each with a distinct role:
1.  **Query Rewriter Agent**: Acts as a pre-processor to make user questions more search-friendly.
2.  **Web Search Agent**: Summarizes real-time findings from the internet when local documents lack the necessary details.
3.  **Gemini RAG Agent**: Synthesizes the final grounded response, prioritizing local memory but clearly identifying when information is derived from the web.

## 🚦 Workflow
1.  **Document Ingestion**: Files are uploaded, split into chunks (2000 characters for PDF, 1000 for web), and converted into embeddings.
2.  **Similarity Search**: When a query is received, the system searches the vector database and calculates a similarity score.
3.  **Decision Logic**: 
    *   **IF** relevant chunks are found (similarity > 0.7), it uses **Internal Memory**.
    *   **ELSE**, it triggers the **Web Search Agent** to fill the knowledge gap.
4.  **Response Generation**: The RAG agent combines the retrieved context and conversation history to generate a cited response.

## ⚙️ Configuration
The application requires valid API keys to function:
*   **Google Gemini API**: Required for reasoning, rewriting, and embeddings.
*   **Exa Search API**: Required for real-time internet search capabilities.
