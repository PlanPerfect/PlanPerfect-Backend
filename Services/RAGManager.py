from typing import List, Dict, Optional
from pinecone import Pinecone
from google import genai
from google.genai import types
from datetime import datetime
import re
import os
import time
from Services import Logger
from dotenv import load_dotenv

load_dotenv()

"""
    RAGManager is a service that encapsulates Retrieval-Augmented Generation (RAG) capabilities into a unified pipeline.
    It manages document ingestion, text chunking, embedding generation, vector storage, retrieval and prompt construction.
    RAGManager provides cobversation history management to enable context-aware interactions with LLMs, and is heavily integrated with LLMManager.
"""

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-001"

class _DocumentProcessor: # handles document loading, cleaning, and chunking
    @staticmethod
    def _load_document(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text) # normalize whitespace
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text) # remove unwanted characters
        return text.strip() # remove leading/trailing whitespace

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        words = text.split()
        total_words = len(words)
        chunks = []

        avg_word_length = sum(len(w) for w in words[:100]) / min(100, len(words))
        words_per_chunk = int(chunk_size / (avg_word_length + 1))
        overlap_words = int(overlap / (avg_word_length + 1))

        overlap_words = min(overlap_words, words_per_chunk // 2)
        overlap_words = max(1, overlap_words)

        i = 0
        chunk_id = 0

        while i < total_words:
            end_idx = min(i + words_per_chunk, total_words) # loop and slice text into chunks

            chunk_text = ' '.join(words[i:end_idx])
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': chunk_text,
                'metadata': {
                    'chunk_id': chunk_id,
                    'start_word': i,
                    'end_word': end_idx
                }
            }) # store chunks

            i = end_idx - overlap_words # create overlap for context awareness

            if end_idx >= total_words:
                break

            chunk_id += 1

        return chunks


class _EmbeddingManager: # handles text embedding
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = EMBEDDING_MODEL
        self.max_retries = 3
        self.base_delay = 1

    def _retry_with_backoff(self, func, *args, **kwargs): # backoff for rate limiting
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()

                if '429' in error_msg or 'rate limit' in error_msg or 'quota' in error_msg:
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt)
                        time.sleep(delay)
                    else:
                        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
                            log_file.write(f"--------------RATE LIMIT HIT--------------\n")
                            log_file.write(f"Model: {self.model}\n")
                            log_file.write(f"Max retries: {self.max_retries}\n")
                            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            log_file.write(f"------------------------------------------\n\n")
                        return None
                else:
                    return None
        return None

    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]: # embed list of texts
        try:
            embeddings = []
            batch_size = 100

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                def _embed_batch():
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(output_dimensionality=1024)
                    ) # call Gemini embedding API
                    return [e.values for e in result.embeddings]

                batch_embeddings = self._retry_with_backoff(_embed_batch)

                if batch_embeddings is None:
                    return None

                embeddings.extend(batch_embeddings)

                if i + batch_size < len(texts):
                    time.sleep(0.2)

            return embeddings

        except Exception as e:
            Logger.log(f"[EMBEDDING MANAGER] - FATAL ERROR: {e}.")
            return None

    def embed_query(self, query: str) -> Optional[List[float]]:
        try:
            def _embed_single():
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=query,
                    config=types.EmbedContentConfig(output_dimensionality=1024)
                ) # call Gemini embedding API
                return result.embeddings[0].values

            result = self._retry_with_backoff(_embed_single)

            return result

        except Exception as e:
            Logger.log(f"[EMBEDDING MANAGER] - FATAL ERROR: {e}.")
            return None

class _VectorManager: # handles vector storage and retrieval with Pinecone
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        try:
            self.index = self.pc.Index(index_name)
            print(f"SUCCESSFULLY CONNECTED TO PINECONE INDEX: \033[94m{index_name}\033[0m\n")
        except Exception as e:
            Logger.log(f"[RAG MANAGER] - ERROR: Failed to connect to Pinecone. Error: {str(e)}")
            raise

    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]): # gather document chunks
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                'id': chunk['id'],
                'values': embedding,
                'metadata': {
                    'text': chunk['text'],
                    'chunk_id': chunk['metadata']['chunk_id'],
                    'start_word': chunk['metadata']['start_word'],
                    'end_word': chunk['metadata']['end_word']
                }
            })

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch) # upsert batch to Pinecone

    def search(self, query_embedding: Optional[List[float]], top_k: int = 3) -> Optional[Dict]:
        if query_embedding is None:
            return None

        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            ) # search based on cosine similarity

            documents = []
            scores = []
            metadatas = []

            for match in results['matches']: # extract results
                documents.append(match['metadata']['text'])
                scores.append(match['score'])
                metadatas.append({
                    'chunk_id': match['metadata']['chunk_id'],
                    'start_word': match['metadata']['start_word'],
                    'end_word': match['metadata']['end_word']
                })

            return {
                'documents': [documents],
                'scores': [scores],
                'metadatas': [metadatas]
            }

        except Exception as e:
            return None

    def get_stats(self):
        return self.index.describe_index_stats()


class RAGManagerClass: # singleton RAGManager class for RAG operations
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._embeddings = None
            cls._instance._vector_store = None
            cls._instance._initialized = False
            cls._instance._conversation_history = []
        return cls._instance

    def initialize(self, document_path: Optional[str] = None, force_reingest: bool = False): # ingest txt document and upsert to Pinecone
        if self._initialized and not force_reingest:
            return

        self._embeddings = _EmbeddingManager(GEMINI_API_KEY)
        self._vector_store = _VectorManager(PINECONE_API_KEY, PINECONE_INDEX_NAME)

        stats = self._vector_store.get_stats()
        vector_count = stats.total_vector_count if hasattr(stats, 'total_vector_count') else stats.get('total_vector_count', 0)

        if document_path and (force_reingest or vector_count == 0):
            if os.path.exists(document_path):
                self._ingest_document(document_path)
            else:
                raise FileNotFoundError(f"Document not found: {document_path}")

        self._initialized = True
        print("RAG MANAGER INTIALISED. DOCUMENTS READY.\n")

    def _ingest_document(self, file_path: str):
        if not self._embeddings or not self._vector_store:
            raise RuntimeError("RAGManager not initialized. Call initialize() first.")

        processor = _DocumentProcessor()

        raw_text = processor._load_document(file_path)
        clean_text = processor._clean_text(raw_text)

        chunks = processor._chunk_text(clean_text, chunk_size=500, overlap=50)

        texts = [chunk['text'] for chunk in chunks]
        embeddings = self._embeddings.embed_texts(texts)

        self._vector_store.add_documents(chunks, embeddings)

    def retrieve_query(self, query: str, top_k: int = 5, include_history: bool = True) -> Optional[str]: # retrieve relevant documents and construct prompt template
        if not self._initialized:
            raise RuntimeError("RAGManager not initialized. Call initialize() first.")

        query_embedding = self._embeddings.embed_query(query)

        if query_embedding is None:
            return None

        results = self._vector_store.search(query_embedding, top_k=top_k)

        if results is None:
            return None

        context_parts = []
        scores = results['scores'][0] if results['scores'] else []

        for i, doc in enumerate(results['documents'][0]):
            score = scores[i] if i < len(scores) else 0
            context_parts.append(f"[Context {i+1}] (similarity: {score:.3f})\n{doc}")

        context = "\n\n".join(context_parts)

        history_str = ""
        if include_history and self._conversation_history: # include last 10 messages in conversation history
            history_str = "\n\nPrevious Conversation:\n"
            for msg in self._conversation_history[-10:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

        prompt = f"""I want you to act as a document that I am having a conversation with. If I tell you my name, I want you to greet me with my name in your reply. If i don't provide my name, just answer normally. Using the provided context, answer the user's question to the best of your ability using the resources provided. Answer in 2-3 sentences, being concise and to the point.

    Your responses should be natural and conversational. IMPORTANT: You can use the conversation history to answer contextual questions. Do NOT say phrases like "according to the context" or "based on the provided information" and DO NOT reference any context such as "Context 1" or "Context 2". DO NOT answer in markdown, only answer in plain text. Simply answer the question directly as if the information is your own knowledge.

    If there is nothing in the context relevant to the question at hand, just say "Hey there! Unfortunately, I only have knowledge on Interior Design Principles. If you have any questions on Interior Design, I'd be happy to help!" and stop after that. Refuse to answer any question not about the info. Never break character.
    ------------
    {context}
    ------------
    {history_str}
    ------------
    REMEMBER:
    - Answer naturally and directly without meta-commentary about the context
    - Greet the user by name together with the reply only if they provide their name in the question
    - Do NOT say "You didn't mention your name, so I'll just have to provide a general answer." or similar. You are only required to greet by name if they provide it.
    - Do NOT mention "Context 1", "Context 2", or similar references about the context
    - Do NOT say "according to the context", or "based on the provided information" or similar phrases
    - Do NOT answer in markdown, only in plain text.
    - You CAN answer questions about the conversation using the conversation history provided
    - You CAN use the conversation history to provide context-aware responses
    - Answer in 2-3 concise sentences
    - If there is no relevant information, just say "Hey there! Unfortunately, I only have knowledge on Interior Design Principles. If you have any questions on Interior Design, I'd be happy to help!"
    - Never break character

    User Question: {query}

    Answer:"""

        return prompt

    def add_to_history(self, role: str, content: str): # add message to conversation history
        self._conversation_history.append({"role": role, "content": content})

    def clear_history(self): # clear conversation history
        self._conversation_history = []

    def get_history(self) -> List[Dict]: # get conversation history
        return self._conversation_history.copy()


RAGManager = RAGManagerClass()