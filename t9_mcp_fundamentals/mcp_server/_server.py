from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from commons.user_service.client import UserServiceClient
from commons.user_service.user_info import UserSearchRequest, UserCreate, UserUpdate

mcp = FastMCP(
    name="users-management-mcp-server",
    host="0.0.0.0",
    port=8005,
)
user_service = UserServiceClient()


# ==================== TOOLS ====================

@mcp.tool()
async def get_user_by_id(user_id: int) -> str:
    """Get a user by their unique ID."""
    return user_service.get_user(user_id)


@mcp.tool()
async def delete_user(user_id: int) -> str:
    """Delete a user by their unique ID."""
    return user_service.delete_user(user_id)


@mcp.tool()
async def search_user(
    name: Optional[str] = None,
    surname: Optional[str] = None,
    email: Optional[str] = None,
    gender: Optional[str] = None,
) -> str:
    """Search users by name, surname, email, or gender (all are optional, partial matching supported)."""
    return user_service.search_users(name=name, surname=surname, email=email, gender=gender)


@mcp.tool()
async def add_user(
    name: str,
    surname: str,
    email: str,
    about_me: str,
    phone: Optional[str] = None,
    date_of_birth: Optional[str] = None,
    gender: Optional[str] = None,
    company: Optional[str] = None,
    salary: Optional[float] = None,
) -> str:
    """Add a new user to the system. name, surname, email and about_me are required."""
    return user_service.add_user(UserCreate(
        name=name,
        surname=surname,
        email=email,
        about_me=about_me,
        phone=phone,
        date_of_birth=date_of_birth,
        gender=gender,
        company=company,
        salary=salary,
    ))


@mcp.tool()
async def update_user(
    user_id: int,
    name: Optional[str] = None,
    surname: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    date_of_birth: Optional[str] = None,
    gender: Optional[str] = None,
    company: Optional[str] = None,
    salary: Optional[float] = None,
) -> str:
    """Update an existing user's fields. Only provided fields will be updated."""
    return user_service.update_user(user_id, UserUpdate(
        name=name,
        surname=surname,
        email=email,
        phone=phone,
        date_of_birth=date_of_birth,
        gender=gender,
        company=company,
        salary=salary,
    ))


# ==================== MCP RESOURCES ====================

@mcp.resource(
    uri="users-management://flow-diagram",
    mime_type="image/png",
    description="Flow diagram of the Users Management Service showing available API endpoints",
)
async def get_flow_diagram() -> bytes:
    """Returns the Users Management Service flow diagram as PNG bytes."""
    flow_path = Path(__file__).parent.parent / "flow.png"
    return flow_path.read_bytes()


# ==================== MCP PROMPTS ====================

_SEARCH_PROMPT = """
You are helping users search through a dynamic user database. The database contains
realistic synthetic user profiles with the following searchable fields:

## Available Search Parameters
- **name**: First name (partial matching, case-insensitive)
- **surname**: Last name (partial matching, case-insensitive)
- **email**: Email address (partial matching, case-insensitive)
- **gender**: Exact match (male, female, other, prefer_not_to_say)

## Search Strategy Guidance

### For Name Searches
- Use partial names: "john" finds John, Johnny, Johnson, etc.
- Try common variations: "mike" vs "michael", "liz" vs "elizabeth"
- Consider cultural name variations

### For Email Searches
- Search by domain: "gmail" for all Gmail users
- Search by name patterns: "john" for emails containing john
- Use company names to find business emails

### For Demographic Analysis
- Combine gender with other criteria for targeted searches
- Use broad searches first, then narrow down

### Effective Search Combinations
- Name + Gender: Find specific demographic segments
- Email domain + Surname: Find business contacts
- Partial names: Cast wider nets for common names

## Example Search Patterns
```
"Find all Johns" → name="john"
"Gmail users named Smith" → email="gmail" + surname="smith"
"Female users with company emails" → gender="female" + email="company"
"Users with Johnson surname" → surname="johnson"
```

## Tips for Better Results
1. Start broad, then narrow down
2. Try variations of names (John vs Johnny)
3. Use partial matches creatively
4. Combine multiple criteria for precision
5. Remember searches are case-insensitive

When helping users search, suggest multiple search strategies and explain
why certain approaches might be more effective for their goals.
"""

_CREATE_PROMPT = """
You are helping create realistic user profiles for the system. Follow these guidelines
to ensure data consistency and realism.

## Required Fields
- **name**: 2-50 characters, letters only, culturally appropriate
- **surname**: 2-50 characters, letters only
- **email**: Valid format, must be unique in system
- **about_me**: Rich, realistic biography (see guidelines below)

## Optional Fields Best Practices
- **phone**: Use E.164 format (+1234567890) when possible
- **date_of_birth**: YYYY-MM-DD format, realistic ages (18-80)
- **gender**: Use standard values (male, female, other, prefer_not_to_say)
- **company**: Real-sounding company names
- **salary**: $30,000-$200,000 range for employed individuals

## Biography Creation ("about_me")
Create engaging, realistic biographies that include:

### Personality Elements
- 1-3 personality traits (curious, adventurous, analytical, etc.)
- Authentic voice and writing style

### Interests & Hobbies
- 2-4 specific hobbies or activities
- 1-3 broader interests or passion areas

### Biography Templates
- "I'm a [trait] person who loves [hobbies]..."
- "When I'm not working, you can find me [activity]..."

## Data Validation Reminders
- Email uniqueness is enforced (check existing users)
- Date formats must be exact (YYYY-MM-DD)
- Salary values should be realistic for the demographic

When creating profiles, aim for diversity in geographic representation, age, and interests.
"""


@mcp.prompt(description="Guide for searching users effectively using available search parameters")
async def search_users_guide() -> str:
    """Provides strategies for searching through the user database."""
    return _SEARCH_PROMPT


@mcp.prompt(description="Guide for creating realistic and consistent user profiles")
async def create_user_guide() -> str:
    """Provides guidelines for creating realistic user profiles."""
    return _CREATE_PROMPT
