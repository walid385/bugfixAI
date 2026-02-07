import base64
import os
from dataclasses import dataclass
from typing import Any, Dict

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GeminiConfig:
    base_url: str
    username: str
    api_key: str

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        base_url = os.getenv("GEMINI_BASE_URL", "").rstrip("/")
        username = os.getenv("GEMINI_USERNAME")
        api_key = os.getenv("GEMINI_API_KEY")
        if not base_url or not username or not api_key:
            raise RuntimeError("GEMINI_BASE_URL / GEMINI_USERNAME / GEMINI_API_KEY fehlen in .env")
        return cls(base_url=base_url, username=username, api_key=api_key)


class GeminiClient:
    def __init__(self, config: GeminiConfig):
        token = f"{config.username}:{config.api_key}".encode("utf-8")
        auth_header = base64.b64encode(token).decode("utf-8")

        self.base_url = config.base_url
        self.headers = {
            "Authorization": f"Basic {auth_header}",  # Username:API-Key, Base64 encodet
            "Accept": "application/json",
        }

    def get_item(self, item_id: int) -> Dict[str, Any]:
        """
        Holt ein einzelnes Ticket (Item) aus Gemini.
        Entspricht: GET /api/items/{itemid}  → IssueDto
        """
        url = f"{self.base_url}/api/items/{item_id}"
        resp = requests.get(url, headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json()


def extract_ticket_text(item: Dict[str, Any]) -> str:
    """
    Versucht, aus der IssueDto die wichtigen Texte zu holen.
    Laut Doku heißt das Ding IssueDto mit einer Entity, die Title/Description enthält.:contentReference[oaicite:2]{index=2}
    """
    entity = item.get("Entity") or item  # Fallback, falls JSON flach ist
    title = str(entity.get("Title", ""))
    description = str(entity.get("Description", ""))
    # Optional: weitere Felder dazunehmen, wenn ihr Custom Fields habt
    return f"{title}\n\n{description}".strip()
