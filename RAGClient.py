# RAGClient.py - For reference only
from Services import RAGManager as RAG
from Services import LLMManager as LLM

def main():
    if not RAG._initialized:
        RAG.initialize(
            document_path="rag_context.txt",
            force_reingest=False
        )

    if not LLM._initialized:
        LLM.initialize()

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
            llm_prompt = RAG.retrieve_query(query=user_query)

            assistant_response = LLM.chat(llm_prompt)

            RAG.add_to_history("user", user_query)
            RAG.add_to_history("assistant", assistant_response)

            print(f"\n🤖 Assistant: {assistant_response}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()