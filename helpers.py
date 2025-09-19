import json
import os
import webbrowser
from datetime import datetime
from logging import getLogger
from math import log

import networkx as nx
from mastodon import Mastodon

logger = getLogger(__name__)


def json_serializer(obj) -> str:
    """
    Custom JSON serializer to handle datetime objects and other non-serializable types.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()

    # For any other non-serializable object, convert to string
    return str(obj)


def save_json(data: list[dict], filename: str) -> None:
    """
    Saves the data to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, default=json_serializer)


# Private function to authenticate with Mastodon
def _auth() -> None:
    """
    Handles authentication with Mastodon.
    """
    try:
        # Check for existing credentials
        if os.path.exists("pytooter_usercred.secret"):
            return

        client = Mastodon(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            access_token=os.getenv("ACCESS_TOKEN"),
            api_base_url="https://mastodon.social",
        )

        # Take the user to the auth url
        auth_url = client.auth_request_url()
        webbrowser.open(auth_url)

        client.log_in(
            code=input("Enter the code: "), to_file="pytooter_usercred.secret"
        )
    except Exception as e:
        logger.exception(f"Error in _auth: {e}")
        raise


def get_client() -> Mastodon | None:
    """
    Gets the Mastodon client or authenticates if the credentials file does not exist.
    """

    try:
        # If the credentials file exists, return the client
        if os.path.exists("pytooter_usercred.secret"):
            return Mastodon(
                client_id=os.getenv("CLIENT_ID"),
                client_secret=os.getenv("CLIENT_SECRET"),
                access_token=os.getenv("ACCESS_TOKEN"),
                api_base_url="https://mastodon.social",
            )
        else:
            _auth()
            return get_client()
    except Exception as e:
        logger.exception(f"Error in get_client: {e}")
        raise


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert a hex color like "#aabbcc" to an (r, g, b) tuple.
    """
    hex_color = hex_color.lstrip("#")
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert an (r, g, b) tuple to a hex string like "#aabbcc".
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def _adjust_brightness(hex_color: str, factor: float) -> str:
    """
    Adjust color brightness by multiplying each channel by factor.
    factor > 1 brightens, 0 < factor < 1 darkens.
    """
    r, g, b = _hex_to_rgb(hex_color)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return _rgb_to_hex(r, g, b)


def generate_distinct_colors(base_colors: list[str], required_count: int) -> list[str]:
    """
    Return a list of at least required_count visually distinct colors.

    Starts with provided base_colors and, if more are needed, appends simple
    lighter/darker variants of those base colors.
    """
    if required_count <= 0:
        return []

    colors: list[str] = []

    # Seed with base palette (dedup while preserving order)
    seen: set[str] = set()
    for c in base_colors:
        if c not in seen:
            colors.append(c)
            seen.add(c)
            if len(colors) >= required_count:
                return colors[:required_count]

    # If more needed, generate brightness variants in simple cycles
    # Use a few gentle steps to avoid extremes
    steps = [0.85, 1.15, 0.7, 1.3, 0.6, 1.4]
    step_index = 0
    while len(colors) < required_count:
        factor = steps[step_index % len(steps)]
        step_index += 1
        for base in base_colors:
            variant = _adjust_brightness(base, factor)
            if variant not in seen:
                colors.append(variant)
                seen.add(variant)
                if len(colors) >= required_count:
                    break

    return colors[:required_count]


def build_community_color_map(
    communities: list[str],
    base_colors: list[str] | None = None,
    default_color: str = "#e5e4df",
) -> dict[str, str]:
    """
    Build a color map for given communities. Ensures we have enough colors by
    generating variants when communities exceed base palette size.
    """
    if base_colors is None:
        base_colors = [
            "#cc785c",
            "#d4a27f",
            "#ebdbbc",
            "#ff6f61",
            "#da7756",
            "#e8a87c",
            "#c9a96e",
            "#f4e4c1",
            "#ff8a80",
            "#e09c7a",
            "#d1956b",
            "#b8926f",
            "#f0d5a8",
            "#ff7043",
            "#cd8471",
            "#a67c52",
            "#deb887",
            "#f5deb3",
            "#ff9800",
            "#d2691e",
        ]

    # Unique communities keeping input order
    unique = []
    seen = set()
    for c in communities:
        if c not in seen:
            unique.append(c)
            seen.add(c)

    # Reserve default for "untagged"
    labeled = [c for c in unique if c != "untagged"]
    needed = len(labeled)
    palette = generate_distinct_colors(base_colors, needed)

    color_map: dict[str, str] = {"untagged": default_color}
    for idx, community in enumerate(labeled):
        color_map[community] = palette[idx]

    return color_map


def save_graph_to_gephi(graph: nx.Graph | nx.DiGraph, filename: str) -> None:
    """
    Save a NetworkX graph to a Gephi-compatible GEXF file.
    """
    try:
        nx.write_gexf(graph, filename)
    except Exception:
        logger.exception("Error in save_graph_to_gephi: ")
        raise
