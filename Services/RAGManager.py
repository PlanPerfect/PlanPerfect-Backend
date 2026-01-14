from typing import List, Dict
from pinecone import Pinecone
from mistralai import Mistral
from groq import Groq
import re
import os
import time
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "planperfect-rag-manager-v1"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "mistral-embed"
# CHAT_MODEL = "llama-3.3-70b-versatile"
# CHAT_MODEL = "llama-3.1-70b-versatile"
# CHAT_MODEL = "llama-3.1-8b-instant"
# CHAT_MODEL = "llama3-70b-8192"
# CHAT_MODEL = "llama3-8b-8192"

CHAT_MODEL = "llama-3.3-70b-versatile"

from typing import List, Dict
import re

class DocumentProcessor:
    @staticmethod
    def load_document(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
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
        last_progress = -1

        while i < total_words:
            end_idx = min(i + words_per_chunk, total_words)

            chunk_text = ' '.join(words[i:end_idx])
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': chunk_text,
                'metadata': {
                    'chunk_id': chunk_id,
                    'start_word': i,
                    'end_word': end_idx
                }
            })

            i = end_idx - overlap_words

            if end_idx >= total_words:
                break

            chunk_id += 1

        return chunks


class MistralEmbeddings:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.model = EMBEDDING_MODEL

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                inputs=batch
            )
            embeddings.extend([item.embedding for item in response.data])

            if i + batch_size < len(texts):
                time.sleep(0.5)

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            inputs=[query]
        )
        return response.data[0].embedding


class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        try:
            self.index = self.pc.Index(index_name)
            print(f"SUCCESSFULLY CONNECTED TO PINECONE: {index_name}")
        except Exception as e:
            raise Exception(f"Could not connect to index '{index_name}'. Make sure it exists in your Pinecone dashboard. Error: {str(e)}")

    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]):
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
            self.index.upsert(vectors=batch)

    def search(self, query_embedding: List[float], top_k: int = 3) -> Dict:
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        documents = []
        scores = []
        metadatas = []

        for match in results['matches']:
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

    def get_stats(self):
        return self.index.describe_index_stats()


class RAGManager:
    def __init__(self, mistral_api_key: str, pinecone_api_key: str, index_name: str):
        self.embeddings = MistralEmbeddings(mistral_api_key)
        self.vector_store = PineconeVectorStore(pinecone_api_key, index_name)
        self.llm = Groq(api_key=GROQ_API_KEY)
        self.chat_model = CHAT_MODEL
        self.conversation_history = []

    def ingest_document(self, file_path: str):
        processor = DocumentProcessor()

        raw_text = processor.load_document(file_path)
        clean_text = processor.clean_text(raw_text)

        chunks = processor.chunk_text(clean_text, chunk_size=500, overlap=50)

        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embeddings.embed_texts(texts)

        self.vector_store.add_documents(chunks, embeddings)

    def retrieve_context(self, query: str, top_k: int = 3) -> tuple[str, List[float]]:
        query_embedding = self.embeddings.embed_query(query)

        results = self.vector_store.search(query_embedding, top_k=top_k)

        context_parts = []
        scores = results['scores'][0] if results['scores'] else []

        for i, doc in enumerate(results['documents'][0]):
            score = scores[i] if i < len(scores) else 0
            context_parts.append(f"[Context {i+1}] (similarity: {score:.3f})\n{doc}")

        return "\n\n".join(context_parts), scores

    def generate_response(self, query: str, context: str) -> str:
        history_str = ""
        if self.conversation_history:
            history_str = "\n\nPrevious Conversation:\n"
            for msg in self.conversation_history[-10:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

        prompt = f"""I want you to act as a document that I am having a conversation with. Using the provided context, answer the user's question to the best of your ability using the resources provided.

Your responses should be natural and conversational. Do NOT say phrases like "according to the context" or "based on the provided information" or reference "Context 1" or "Context 2". Simply answer the question directly as if the information is your own knowledge.

If there is nothing in the context relevant to the question at hand, just say "Hey there! Unfortunately, I only have knowledge on generic Interior Design Principles. If you have any questions on Interior Design, I am here to help!" and stop after that. Refuse to answer any question not about the info. Never break character.
------------
{context}
------------
{history_str}
------------
REMEMBER:
- Answer naturally and directly without meta-commentary about the context
- Do NOT mention "Context 1", "Context 2", or similar references
- Do NOT say "according to the context" or similar phrases
- Use the conversation history to provide context-aware responses
- If there is no relevant information, just say "Hey there! Unfortunately, I only have knowledge on generic Interior Design Principles. If you have any questions on Interior Design, I am here to help!"
- Never break character

User Question: {query}

Answer:"""

        response = self.llm.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def chat(self, query: str, top_k: int = 3) -> Dict:
        context, scores = self.retrieve_context(query, top_k=top_k)

        print(f"💭 Generating response with {self.chat_model}...")

        response = self.generate_response(query, context)

        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})

        with open("rag.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"---\n")
            log_file.write(f"Query: {query}\n")
            log_file.write(f"Chat Model: {self.chat_model}\n")
            log_file.write(f"Retrieved Context:\n{context}")
            log_file.write(f"\nResponse:\n{response}\n")
            log_file.write(f"---\n\n")

        return {
            'query': query,
            'context': context,
            'response': response,
            'scores': scores
        }

    def clear_history(self):
        self.conversation_history = []
        print("🗑️  Conversation history cleared")


def main():
    try:
        chatbot = RAGManager(
            mistral_api_key=MISTRAL_API_KEY,
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME
        )
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return

    document_path = "rag_context.txt"

    if os.path.exists(document_path):
        chatbot.ingest_document(document_path)
    else:
        print(f"\n⚠️  Document not found: {document_path}")
        print("   Continuing with existing vectors in Pinecone...")

    print("\n" + "="*60)
    print("Type 'quit' to exit, or 'clear' to clear conversation history.")
    print("="*60)

    while True:
        query = input("\n🗣️  You: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if query.lower() == 'clear':
            chatbot.clear_history()
            continue

        if not query:
            continue

        try:
            result = chatbot.chat(query, top_k=5)
            print(f"\n🤖 Assistant: {result['response']}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()