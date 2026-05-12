import anthropic

# from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_ENDPOINT, OPENAI_CHAT_COMPLETIONS_ENDPOINT
from commons.constants import ANTHROPIC_API_KEY
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
# from t5_rag_advanced.chat.chat_completion_client import ChatCompletionClient
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import TextProcessor, SearchMode

SYSTEM_PROMPT = """
You are a Tiguan manual assistant. Answer ONLY using the provided RAG Context.
If the answer is not in the context, reply: "I don't have information about that in the manual."
Be concise.
"""

USER_PROMPT = """
RAG Context:
{context}

User Question:
{question}
"""

def main():
    embeddings_client = EmbeddingsClient(
        endpoint='',
        model_name='',
        api_key=''
    )

    # chat_client = ChatCompletionClient(
    #     endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    #     model_name='gpt-5.2',
    #     api_key=OPENAI_API_KEY
    # )
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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
            file_path='t5_rag_advanced/embeddings/tiguan-2026-owners-manual.pdf',
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
            search_mode=SearchMode.COSINE_DISTANCE,
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

        response = anthropic_client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {'role': msg.role, 'content': msg.content}
                for msg in conversation.messages
                if msg.role != Role.SYSTEM
            ]
        )
        reply = response.content[0].text
        print(f"\nAssistant: {reply}")
        conversation.add_message(Message(Role.ASSISTANT, reply))


if __name__ == "__main__":
    main()



# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml
