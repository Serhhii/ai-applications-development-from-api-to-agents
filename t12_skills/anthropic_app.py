import json
import anthropic
from pathlib import Path

from commons.constants import ANTHROPIC_API_KEY


SKILLS_VERSION = "skills-2025-10-02"

def get_or_create_skill(skill_title: str, skill_dir: Path, client: anthropic.Anthropic) -> str:
    skills = client.beta.skills.list(source="custom", betas=[SKILLS_VERSION])
    for skill in skills:
        if skill.display_title == skill_title:
            print(f"Skill already exists: id={skill.id}, title={skill.display_title}")
            return skill.id

    files = anthropic.lib.files_from_dir(skill_dir)
    new_skill = client.beta.skills.create(
        display_title=skill_title,
        files=files,
        betas=[SKILLS_VERSION],
    )
    print(f"Created skill: id={new_skill.id}, title={new_skill.display_title}")
    return new_skill.id


def delete_skills(client: anthropic.Anthropic):
    skills = client.beta.skills.list(source="custom", betas=[SKILLS_VERSION])
    for skill in skills:
        versions = client.beta.skills.versions.list(skill.id, betas=[SKILLS_VERSION])
        for version in versions:
            client.beta.skills.versions.delete(version.version, skill_id=skill.id, betas=[SKILLS_VERSION])
            print(f"Deleted version {version.version} of skill {skill.id}")
        client.beta.skills.delete(skill.id, betas=[SKILLS_VERSION])
        print(f"Deleted skill {skill.id} ({skill.display_title})")


def chat(client: anthropic.Anthropic, skill_id: str, log_request: bool = True, log_response: bool = True):
    """Multi-turn chat loop that reuses the container across turns."""
    messages = []
    container_id = None
    print("\nAgent is ready. Type your request or 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        container = {
            "skills": [{"type": "custom", "skill_id": skill_id, "version": "latest"}]
        }
        if container_id:
            container["id"] = container_id

        request_payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 8096,
            "messages": messages,
            "container": container,
            "betas": ["code-execution-2025-08-25", SKILLS_VERSION],
            "tools": [{"type": "code_execution_20250825", "name": "code_execution"}],
        }

        if log_request:
            print(f"\n→ REQUEST:\n{json.dumps(request_payload, indent=2, default=str)}\n")

        response = client.beta.messages.create(**request_payload)

        if log_response:
            print(f"\n← RESPONSE:\n{json.dumps(response.model_dump(), indent=2, default=str)}\n")
        else:
            text = "".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            print(f"\nClaude: {text}\n")

        if hasattr(response, "container") and response.container:
            container_id = response.container.id

        messages.append({"role": "assistant", "content": response.content})


STYLE_SKILL_TITLE = "style-guide"
STYLE_SKILL_DIR = Path(__file__).parent / "_skills" / STYLE_SKILL_TITLE

CALCULATOR_SKILL_TITLE = "calculator"
CALCULATOR_SKILL_DIR = Path(__file__).parent / "_skills" / CALCULATOR_SKILL_TITLE


def main():
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    skill_id = get_or_create_skill(CALCULATOR_SKILL_TITLE, CALCULATOR_SKILL_DIR, client)
    chat(client, skill_id, log_request=False, log_response=False)
    delete_skills(client)


if __name__ == "__main__":
    main()
