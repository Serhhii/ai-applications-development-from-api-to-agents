from typing import Any

from t8_agent.task.tools.users.base import BaseUserServiceTool


class GetUserByIdTool(BaseUserServiceTool):

    @property
    def name(self) -> str:
        return "get_user_by_id"

    @property
    def description(self) -> str:
        return "Fetch full information about a user by their numeric ID."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "number", "description": "The user's unique numeric ID."},
            },
            "required": ["id"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            return self.user_client.get_user(int(arguments["id"]))
        except Exception as e:
            return f"Error while retrieving user by id: {str(e)}"
