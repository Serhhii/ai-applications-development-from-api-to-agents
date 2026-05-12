from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_ENDPOINT, OPENAI_CHAT_COMPLETIONS_ENDPOINT
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t5_rag_advanced.chat.chat_completion_client import ChatCompletionClient
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import TextProcessor, SearchMode

SYSTEM_PROMPT = """
You are a RAG-powered assistant specializing in microwave manual assistance.
Your task is to answer user questions based on the provided RAG Context.

User messages will be structured with a "RAG Context" section followed by a "User Question".
Please adhere to the following rules:
1. Use ONLY the provided RAG Context to answer the User Question.
2. If the answer cannot be found in the context, politely state that you can only answer questions related to the microwave based on the manual.
3. Do not answer questions that are not related to microwave usage or are outside the scope of the provided context.
"""

USER_PROMPT = """
RAG Context:
{context}

User Question:
{question}
"""

def main():
    embeddings_client = EmbeddingsClient(
        endpoint=OPENAI_EMBEDDINGS_ENDPOINT,
        model_name='text-embedding-3-small',
        api_key=OPENAI_API_KEY
    )

    chat_client = ChatCompletionClient(
        endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT,
        model_name='gpt-5.2',
        api_key=OPENAI_API_KEY
    )

    db_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }

    text_processor = TextProcessor(embeddings_client, db_config)

    load_context = input("Do you want to load context from file? (y/n): ").lower() == 'y'
    if load_context:
        text_processor.process_text_file(
            file_path='t5_rag_advanced/embeddings/microwave_manual.txt',
            chunk_size=300,
            overlap=40,
            dimensions=384,
            truncate=True
        )
        print("Context loaded successfully.")

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    print("Microwave Assistant is ready. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # Retrieval
        relevant_chunks = text_processor.search(
            mode=SearchMode.COSINE_DISTANCE,
            query=user_input,
            top_k=5,
            min_score=0.5,
            dimensions=384
        )

        # Augmentation
        context = "\n---\n".join(relevant_chunks)
        augmented_prompt = USER_PROMPT.format(context=context, question=user_input)

        # Generation
        conversation.add_message(Message(Role.USER, augmented_prompt))
        response_message = chat_client.get_completion(conversation.messages)

        print(f"\nAssistant: {response_message.content}")
        conversation.add_message(response_message)

if __name__ == "__main__":
    main()



# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml
