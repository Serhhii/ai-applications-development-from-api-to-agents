from commons.constants import ANTHROPIC_API_KEY
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from commons.user_service.client import UserServiceClient

from t8_agent.task.agents.anthropic import AnthropicBasedAgent
from t8_agent.task.prompts import SYSTEM_PROMPT
from t8_agent.task.tools.users.create_user_tool import CreateUserTool
from t8_agent.task.tools.users.delete_user_tool import DeleteUserTool
from t8_agent.task.tools.users.get_user_by_id_tool import GetUserByIdTool
from t8_agent.task.tools.users.search_users_tool import SearchUsersTool
from t8_agent.task.tools.users.update_user_tool import UpdateUserTool


def main():
    user_client = UserServiceClient()
    tools = [
        GetUserByIdTool(user_client),
        SearchUsersTool(user_client),
        CreateUserTool(user_client),
        UpdateUserTool(user_client),
        DeleteUserTool(user_client),
    ]

    agent = AnthropicBasedAgent(
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    conversation = Conversation()

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() == "exit":
            break
        conversation.add_message(Message(role=Role.USER, content=user_input))
        response = agent.get_response(conversation.get_messages(), print_request=False)
        conversation.add_message(response)
        print(f"\n🤖: {response.content}")


main()
