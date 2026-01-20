# RAGClient.py - For reference only. This cannot be used for production.
from Services import RAGManager as RAG
from Services import LLMManager as LLM

"""
    RAGClient is a simple command-line interface to interact with the RAG and LLM services.
    It serves as a simple simulation of a chatbot interface, allowing for testing retrieval and generation capabilities.
"""

def main():
    if not RAG._initialized:
        RAG.initialize(
            document_path="rag_context.txt", # ingest the rag_context.txt document
            force_reingest=False # dont ingest if already ingested before
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
            llm_prompt = RAG.retrieve_query(query=user_query) # retrieve prompt with context from RAG

            assistant_response = LLM.chat(llm_prompt) # get response from LLM

            RAG.add_to_history("user", user_query) # add user query to conversation history
            RAG.add_to_history("assistant", assistant_response) # add assistant response to conversation history

            print(f"\n🤖 Assistant: {assistant_response}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()