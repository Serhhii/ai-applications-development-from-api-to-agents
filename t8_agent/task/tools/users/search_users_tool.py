from typing import Any

from t8_agent.task.tools.users.base import BaseUserServiceTool


class SearchUsersTool(BaseUserServiceTool):

    @property
    def name(self) -> str:
        return "search_users"

    @property
    def description(self) -> str:
        return "Search users by name, surname, email, or gender. All parameters are optional and support partial matching."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "First name (partial match)."},
                "surname": {"type": "string", "description": "Last name (partial match)."},
                "email": {"type": "string", "description": "Email address (partial match)."},
                "gender": {"type": "string", "description": "Gender (male, female, other, prefer_not_to_say)."},
            },
            "required": [],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            return self.user_client.search_users(**arguments)
        except Exception as e:
            return f"Error while searching users: {str(e)}"
