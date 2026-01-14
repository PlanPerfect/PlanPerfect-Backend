# for reference only - RAGClient.py
from groq import Groq
import os
from Services import RAGManager as RAG

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHAT_MODEL = "llama-3.3-70b-versatile"
groq_client = Groq(api_key=GROQ_API_KEY)


def main():
    if not RAG._initialized:
        RAG.initialize(
            document_path="rag_context.txt",
            force_reingest=True
        )

    print("="*60)
    print("Type 'quit' to exit, 'clear' to clear conversation history")
    print("="*60)

    while True:
        user_query = input("\n🗣️  You: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if user_query.lower() == 'clear':
            RAG.clear_history()
            print("🗑️  Conversation history cleared")
            continue

        if not user_query:
            continue

        try:
            llm_prompt = RAG.retrieve_query(
                query=user_query,
                top_k=5,
                include_history=True
            )

            response = groq_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "user", "content": llm_prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )

            assistant_response = response.choices[0].message.content

            RAG.add_to_history("user", user_query)
            RAG.add_to_history("assistant", assistant_response)

            with open("rag.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"---\n")
                log_file.write(f"Query: {user_query}\n")
                log_file.write(f"Chat Model: {CHAT_MODEL}\n")
                log_file.write(f"RAG Output:\n{llm_prompt}\n")
                log_file.write(f"\nResponse:\n{assistant_response}\n")
                log_file.write(f"---\n\n")

            print(f"\n🤖 Assistant: {assistant_response}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()