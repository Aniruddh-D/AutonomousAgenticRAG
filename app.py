import hashlib
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import faiss
import numpy as np
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader


DATA_DIR = Path("app_data")
CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
HISTORY_PATH = DATA_DIR / "history.json"
SUPPORTED_TYPES = ["pdf", "txt", "docx"]
EMBEDDING_MODELS = [
    "models/embedding-001",
    "models/gemini-embedding-001",
    "models/text-embedding-004",
]
GENERATION_MODEL = "gemini-2.5-flash"
MAX_CHUNK_CHARS = 1200
TOP_K = 4


def ensure_storage():
    DATA_DIR.mkdir(exist_ok=True)


def load_json(path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return default


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)


def load_embeddings():
    if EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.astype("float32")
    return np.empty((0, 768), dtype="float32")


def save_embeddings(embeddings):
    np.save(EMBEDDINGS_PATH, embeddings.astype("float32"))


def bootstrap_state():
    ensure_storage()
    if "bootstrapped" in st.session_state:
        return

    st.session_state.bootstrapped = True
    st.session_state.chunks = load_json(CHUNKS_PATH, [])
    st.session_state.chat_history = load_json(HISTORY_PATH, [])
    st.session_state.embeddings = load_embeddings()
    st.session_state.last_retrieval = []
    st.session_state.embedding_model = None

    if "knowledge_index" not in st.session_state:
        st.session_state.knowledge_index = None

    rebuild_index()


def persist_state():
    save_json(CHUNKS_PATH, st.session_state.chunks)
    save_json(HISTORY_PATH, st.session_state.chat_history)
    save_embeddings(st.session_state.embeddings)


def rebuild_index():
    embeddings = st.session_state.embeddings
    if len(embeddings) == 0:
        st.session_state.knowledge_index = None
        return

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    st.session_state.knowledge_index = index


def file_digest(uploaded_file):
    return hashlib.sha1(uploaded_file.getvalue()).hexdigest()


def docx_to_text(file_bytes):
    with zipfile.ZipFile(file_bytes) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        joined = "".join(texts).strip()
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def read_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text.strip())
        return "\n\n".join(pages)
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".docx":
        return docx_to_text(uploaded_file)
    raise ValueError(f"Unsupported file type: {suffix}")


def normalize_paragraphs(text):
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    paragraphs = []
    current = []

    for line in lines:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))

    return [paragraph for paragraph in paragraphs if paragraph]


def split_long_paragraph(paragraph, max_chars):
    if len(paragraph) <= max_chars:
        return [paragraph]

    words = paragraph.split()
    pieces = []
    current = []

    for word in words:
        tentative = " ".join(current + [word]).strip()
        if len(tentative) <= max_chars:
            current.append(word)
        else:
            pieces.append(" ".join(current))
            current = [word]

    if current:
        pieces.append(" ".join(current))

    return [piece for piece in pieces if piece]


def build_chunks(text, source_name, source_type, document_id):
    paragraphs = normalize_paragraphs(text)
    expanded = []
    for paragraph in paragraphs:
        expanded.extend(split_long_paragraph(paragraph, MAX_CHUNK_CHARS))

    chunks = []
    buffer = []
    buffer_len = 0
    chunk_number = 1

    for paragraph in expanded:
        extra = len(paragraph) + (2 if buffer else 0)
        if buffer and buffer_len + extra > MAX_CHUNK_CHARS:
            chunk_text = "\n\n".join(buffer)
            chunks.append(
                {
                    "chunk_id": f"{document_id}-chunk-{chunk_number}",
                    "document_id": document_id,
                    "source_name": source_name,
                    "source_type": source_type,
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                    "preview": chunk_text[:220].replace("\n", " "),
                }
            )
            chunk_number += 1
            buffer = [paragraph]
            buffer_len = len(paragraph)
        else:
            buffer.append(paragraph)
            buffer_len += extra

    if buffer:
        chunk_text = "\n\n".join(buffer)
        chunks.append(
            {
                "chunk_id": f"{document_id}-chunk-{chunk_number}",
                "document_id": document_id,
                "source_name": source_name,
                "source_type": source_type,
                "chunk_number": chunk_number,
                "text": chunk_text,
                "preview": chunk_text[:220].replace("\n", " "),
            }
        )

    return chunks


def get_embedding(text):
    last_error = None

    preferred = []
    if st.session_state.embedding_model:
        preferred.append(st.session_state.embedding_model)
    preferred.extend([model_name for model_name in EMBEDDING_MODELS if model_name not in preferred])

    for model_name in preferred:
        try:
            response = genai.embed_content(model=model_name, content=text)
            st.session_state.embedding_model = model_name
            return np.array(response["embedding"], dtype="float32")
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Embedding request failed for all configured models: {last_error}")


def embed_chunks(chunks):
    vectors = [get_embedding(chunk["text"]) for chunk in chunks]
    return np.array(vectors, dtype="float32")


def get_history_text(limit=6):
    history = st.session_state.chat_history[-limit:]
    return "\n".join([f"User: {item['question']}\nAssistant: {item['answer']}" for item in history])


def search_documents(query, k=TOP_K):
    index = st.session_state.knowledge_index
    chunks = st.session_state.chunks
    if index is None or not chunks:
        return []

    query_vector = get_embedding(query)
    distances, indices = index.search(np.array([query_vector], dtype="float32"), min(k, len(chunks)))
    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(
            {
                "distance": float(distance),
                "score": round(1 / (1 + float(distance)), 4),
                "chunk": chunk,
            }
        )

    return results


def build_citations(results):
    citations = []
    seen = set()
    for result in results:
        chunk = result["chunk"]
        label = f"{chunk['source_name']} chunk {chunk['chunk_number']}"
        if label not in seen:
            citations.append(label)
            seen.add(label)
    return citations


def answer_query(query, mode):
    retrieval = search_documents(query) if mode != "General chat" else []
    st.session_state.last_retrieval = retrieval
    history_text = get_history_text()

    if mode == "Document only":
        if not retrieval:
            return {
                "answer": "I do not have any stored document context yet. Upload a PDF, TXT, or DOCX file first.",
                "citations": [],
            }

        context = "\n\n".join(
            [
                f"[{item['chunk']['source_name']} chunk {item['chunk']['chunk_number']}]\n{item['chunk']['text']}"
                for item in retrieval
            ]
        )
        prompt = f"""
        You are answering strictly from the provided document excerpts.
        If the answer is not supported by the excerpts, say so clearly.
        Cite the supporting excerpts inline using their labels.

        Conversation history:
        {history_text}

        Retrieved excerpts:
        {context}

        User question:
        {query}
        """
    elif mode == "Hybrid":
        context = "\n\n".join(
            [
                f"[{item['chunk']['source_name']} chunk {item['chunk']['chunk_number']}]\n{item['chunk']['text']}"
                for item in retrieval
            ]
        )
        prompt = f"""
        Answer the user's question clearly.
        Prefer the retrieved document excerpts when they are relevant and cite them inline.
        If the documents are not enough, you may use general knowledge, but say that part is not from the uploaded files.

        Conversation history:
        {history_text}

        Retrieved excerpts:
        {context if context else "No document excerpts retrieved."}

        User question:
        {query}
        """
    else:
        prompt = f"""
        Answer the user's question naturally using conversation history.

        Conversation history:
        {history_text}

        User question:
        {query}
        """

    response = model.generate_content(prompt)
    answer_text = getattr(response, "text", "").strip() or "No response generated."
    citations = build_citations(retrieval)

    st.session_state.chat_history.append(
        {
            "question": query,
            "answer": answer_text,
            "mode": mode,
            "citations": citations,
        }
    )
    save_json(HISTORY_PATH, st.session_state.chat_history)
    return {"answer": answer_text, "citations": citations}


def add_documents(uploaded_files):
    existing_ids = {chunk["document_id"] for chunk in st.session_state.chunks}
    new_chunks = []
    added_sources = []
    skipped_sources = []

    for uploaded_file in uploaded_files:
        document_id = file_digest(uploaded_file)
        if document_id in existing_ids:
            skipped_sources.append(uploaded_file.name)
            continue

        text = read_uploaded_file(uploaded_file)
        if not text.strip():
            skipped_sources.append(f"{uploaded_file.name} (empty)")
            continue

        source_type = Path(uploaded_file.name).suffix.lower().lstrip(".")
        chunks = build_chunks(text, uploaded_file.name, source_type, document_id)
        if not chunks:
            skipped_sources.append(f"{uploaded_file.name} (no chunks)")
            continue

        new_chunks.extend(chunks)
        added_sources.append(uploaded_file.name)

    if not new_chunks:
        return added_sources, skipped_sources

    vectors = embed_chunks(new_chunks)
    if len(st.session_state.embeddings) == 0:
        st.session_state.embeddings = vectors
    else:
        st.session_state.embeddings = np.vstack([st.session_state.embeddings, vectors]).astype("float32")

    st.session_state.chunks.extend(new_chunks)
    rebuild_index()
    persist_state()
    return added_sources, skipped_sources


def delete_knowledge_base():
    st.session_state.chunks = []
    st.session_state.embeddings = np.empty((0, 768), dtype="float32")
    st.session_state.last_retrieval = []
    rebuild_index()
    persist_state()


def clear_chat_history():
    st.session_state.chat_history = []
    save_json(HISTORY_PATH, st.session_state.chat_history)


st.set_page_config(page_title="Clean Agentic RAG", layout="wide")
bootstrap_state()

st.sidebar.header("API Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel(GENERATION_MODEL)

st.title("Autonomous Agentic RAG: An Intelligent Knowledge Retrieval Agent")
st.caption("Upload files, build a reusable local knowledge base, and switch how the assistant answers.")

with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, DOCX",
    )

    if st.button("Add to knowledge base", use_container_width=True):
        if uploaded_files:
            with st.spinner("Embedding and storing documents..."):
                try:
                    added_sources, skipped_sources = add_documents(uploaded_files)
                except Exception as exc:
                    st.error(f"Failed to process uploads: {exc}")
                else:
                    if added_sources:
                        st.success(f"Added {len(added_sources)} source(s): {', '.join(added_sources)}")
                    if skipped_sources:
                        st.info(f"Skipped: {', '.join(skipped_sources)}")
        else:
            st.warning("Choose at least one file first.")

    mode = st.radio(
        "Answer mode",
        ["Document only", "Hybrid", "General chat"],
        help="Document only stays grounded in uploaded files. Hybrid mixes files with model knowledge. General chat ignores the knowledge base.",
    )

    st.metric("Stored chunks", len(st.session_state.chunks))
    unique_sources = sorted({chunk["source_name"] for chunk in st.session_state.chunks})
    st.metric("Stored sources", len(unique_sources))

    if unique_sources:
        st.write("Current sources")
        for source in unique_sources:
            st.caption(source)

    if st.button("Clear chat history", use_container_width=True):
        clear_chat_history()
        st.success("Chat history cleared.")

    if st.button("Reset knowledge base", use_container_width=True):
        delete_knowledge_base()
        st.success("Knowledge base cleared.")

main_col, side_col = st.columns([1.7, 1])

with main_col:
    query = st.text_input("Ask something about your files or start a general chat")

    if st.button("Run agent", type="primary"):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    result = answer_query(query.strip(), mode)
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                else:
                    st.subheader("Answer")
                    st.write(result["answer"])
                    if result["citations"]:
                        st.caption("Citations: " + ", ".join(result["citations"]))

    st.subheader("Chat History")
    if not st.session_state.chat_history:
        st.info("No conversation yet.")
    else:
        for item in reversed(st.session_state.chat_history):
            st.write(f"**Mode:** {item['mode']}")
            st.write(f"**User:** {item['question']}")
            st.write(f"**Assistant:** {item['answer']}")
            if item["citations"]:
                st.caption("Sources: " + ", ".join(item["citations"]))
            st.write("---")

with side_col:
    st.subheader("Retrieved Context")
    if st.session_state.last_retrieval:
        for item in st.session_state.last_retrieval:
            chunk = item["chunk"]
            with st.expander(
                f"{chunk['source_name']} | chunk {chunk['chunk_number']} | score {item['score']}",
                expanded=False,
            ):
                st.write(chunk["text"])
    else:
        st.info("Run a document-aware query to inspect retrieved chunks and scores.")
