import argparse
import json
import logging
import os
import sys
from datetime import datetime

import networkx as nx
from mastodon import Mastodon

from helpers import _adjust_brightness, _hex_to_rgb, get_client, save_graph_to_gephi

logger = logging.getLogger(__name__)

COMPUTE_SIZES_BY_DEGREE: bool = True


def _verify_users(path: str = "users.json") -> int:
    """
    Verify the number of users saved in the given JSON file.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be an array of users")

        return len(data)
    except Exception:
        logger.exception("Error in _verify_users: ")
        raise


def find_users(
    followers_limit: int = 10,
    following_limit: int = 10,
    total_users_limit: int = 200,
    show_progress: bool = True,
) -> list[dict]:
    """
    Traverse users starting from seeds, then their followers and followees (BFS),
    until total_users_limit top-level users are collected.

    total_users_limit counts only the number of top-level "user" entries added to
    the output list, not the size of the followers/following arrays included.
    """
    try:
        client: Mastodon | None = get_client()

        if client is None:
            raise Exception("Some error occurred when finding the client")

        # Detect whether we are authenticated; remote resolution requires auth
        is_authenticated = True
        try:
            client.account_verify_credentials()
        except Exception:
            is_authenticated = False

        with open("data.json", "r") as f:
            data = json.load(f)
        seed_users = data["users"]

        all_users = []

        # Maintain visited account IDs (processed as top-level users)
        visited_ids = set()
        # Maintain queued account IDs to avoid enqueuing duplicates
        queued_ids = set()
        # Queue holds account objects (or minimal dicts with id) to process
        queue = []

        # Helper: enqueue an account object if not visited/queued
        def enqueue_account(account_obj):
            try:
                account_id = account_obj.id
            except Exception:
                account_id = (
                    account_obj.get("id") if isinstance(account_obj, dict) else None
                )
            if account_id is None:
                return
            if account_id in visited_ids or account_id in queued_ids:
                return
            # Do not grow the queue if we already have enough to meet the limit
            if total_users_limit and (len(all_users) + len(queue)) >= total_users_limit:
                return
            queue.append(account_obj)
            queued_ids.add(account_id)

        # Seed the queue from provided handles
        for user_handle in seed_users:
            clean_handle = user_handle.strip("@")
            try:
                # Avoid 401 Unauthorized by disabling resolve when unauthenticated
                search_results = client.search_v2(
                    q=clean_handle, resolve=is_authenticated
                )
                if search_results.get("accounts"):
                    seed_account = search_results["accounts"][0]
                    enqueue_account(seed_account)
                    if (
                        total_users_limit
                        and (len(all_users) + len(queue)) >= total_users_limit
                    ):
                        break
            except Exception:
                logger.exception(f"Error resolving seed user {user_handle}")

        # BFS traversal
        while queue and len(all_users) < total_users_limit:
            current = queue.pop(0)
            # Determine current account id
            try:
                current_id = current.id
            except Exception:
                current_id = current.get("id") if isinstance(current, dict) else None
            if current_id is None:
                continue
            if current_id in visited_ids:
                continue

            # Mark as visited when we decide to process as a top-level user
            visited_ids.add(current_id)

            one_user = {}
            try:
                # Fetch a fresh account object to normalize shape
                user_account = client.account(current_id)
            except Exception:
                # Fall back to whatever we already have
                user_account = current
            one_user["user"] = user_account

            # Fetch followers and following (may be empty for private/remote accounts)
            try:
                followers = client.account_followers(current_id, limit=followers_limit)
            except Exception:
                followers = []
            try:
                following = client.account_following(current_id, limit=following_limit)
            except Exception:
                following = []

            if followers:
                one_user["followers"] = followers
                # Enqueue followers for future processing
                for acct in followers:
                    if (
                        total_users_limit
                        and (len(all_users) + len(queue)) >= total_users_limit
                    ):
                        break
                    enqueue_account(acct)

            if following:
                one_user["following"] = following
                # Enqueue followees for future processing
                for acct in following:
                    if (
                        total_users_limit
                        and (len(all_users) + len(queue)) >= total_users_limit
                    ):
                        break
                    enqueue_account(acct)

            # Add the processed user block to output (counts toward total_users_limit)
            all_users.append(one_user)

            # Progress indicator
            if show_progress:
                processed = len(all_users)
                percent = (
                    int((processed / total_users_limit) * 100)
                    if total_users_limit
                    else 0
                )
                sys.stdout.write(
                    f"\rUsers processed: {processed}/{total_users_limit} ({percent}%) | queue: {len(queue)}"
                )
                sys.stdout.flush()

        # Finish progress line
        if show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return all_users
    except Exception:
        logger.exception("Error in find_users: ")
        raise


def create_user_network_graph(
    filename: str = "users.json", compute_sizes: bool = True
) -> nx.Graph:
    """
    Create an undirected user friendship network from users data.

    Each node represents a user (by account id). An undirected edge between A and B
    indicates the presence of a follower/followee relationship in either direction.

    Args:
        filename: Path to the JSON file containing users data.

    Returns:
        nx.Graph: An undirected graph where nodes are user IDs and edges represent
                  follower/following relationships.
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Users file not found: {filename}")

        with open(filename, "r") as f:
            users_data = json.load(f)

        if not isinstance(users_data, list):
            raise ValueError("Users data must be a list of user objects.")

        graph = nx.Graph()

        def get_account_id(account_obj):
            try:
                return account_obj.id
            except Exception:
                return account_obj.get("id") if isinstance(account_obj, dict) else None

        def add_node_from_account(account_obj):
            account_id = get_account_id(account_obj)
            if account_id is None:
                return None
            # Extract a few helpful attributes when available
            try:
                username = getattr(account_obj, "username", None)
                acct = getattr(account_obj, "acct", None)
                display_name = getattr(account_obj, "display_name", None)
                url = getattr(account_obj, "url", None)
            except Exception:
                username = acct = display_name = url = None

            if isinstance(account_obj, dict):
                username = account_obj.get("username", username)
                acct = account_obj.get("acct", acct)
                display_name = account_obj.get("display_name", display_name)
                url = account_obj.get("url", url)

            default_color = "#e5e4df"
            r = int(default_color[1:3], 16)
            g = int(default_color[3:5], 16)
            b = int(default_color[5:7], 16)

            graph.add_node(
                account_id,
                **{
                    "username": username or "",
                    "acct": acct or "",
                    "display_name": display_name or "",
                    "url": url or "",
                    "color": default_color,
                    "viz": {
                        "color": {
                            "r": r,
                            "g": g,
                            "b": b,
                        }
                    },
                },
            )
            return account_id

        # Build nodes and undirected edges
        for entry in users_data:
            if not isinstance(entry, dict):
                continue

            main_user = entry.get("user")
            if not main_user:
                continue

            main_id = add_node_from_account(main_user)
            if main_id is None:
                continue

            followers = entry.get("followers", []) or []
            for follower in followers:
                follower_id = add_node_from_account(follower)
                if follower_id is None or follower_id == main_id:
                    continue
                # Undirected edge represents follower/followee relationship
                graph.add_edge(main_id, follower_id)

            following = entry.get("following", []) or []
            for followee in following:
                followee_id = add_node_from_account(followee)
                if followee_id is None or followee_id == main_id:
                    continue
                graph.add_edge(main_id, followee_id)

        if compute_sizes:
            _add_node_sizes_by_degree(graph)

        return graph
    except Exception:
        logger.exception("Error in create_user_network_graph: ")
        raise


def _add_node_sizes_by_degree(
    graph: nx.Graph, min_size: int = 5, max_size: int = 50
) -> None:
    """
    Calculate degree for each node, normalize it, and add size attributes.

    For undirected graphs, degree is used as the total degree.
    """
    try:
        if not graph.nodes():
            return

        node_degrees = {}
        for node in graph.nodes():
            degree = graph.degree[node]
            node_degrees[node] = {
                "total_degree": degree,
            }

        total_degrees = [data["total_degree"] for data in node_degrees.values()]
        max_degree = max(total_degrees)
        min_degree = min(total_degrees)
        if max_degree == 0:
            max_degree = 1

        base_color = "#e5e4df"
        light_factor = 1.3
        dark_factor = 0.7
        degree_to_color: dict[int, str] = {}
        degree_to_rgb: dict[int, tuple[int, int, int]] = {}

        unique_degrees = sorted(set(total_degrees))
        for d in unique_degrees:
            if max_degree == min_degree:
                t = 0.0
            else:
                t = (d - min_degree) / (max_degree - min_degree)
            factor = (1 - t) * light_factor + t * dark_factor
            hex_color = _adjust_brightness(base_color, factor)
            r, g, b = _hex_to_rgb(hex_color)
            degree_to_color[d] = hex_color
            degree_to_rgb[d] = (r, g, b)

        for node, degrees in node_degrees.items():
            normalized_degree = degrees["total_degree"] / max_degree
            size = min_size + normalized_degree * (max_size - min_size)

            graph.nodes[node].update(
                {
                    "total_degree": degrees["total_degree"],
                    "size": size,
                    "color": degree_to_color[degrees["total_degree"]],
                }
            )

            if "viz" not in graph.nodes[node]:
                graph.nodes[node]["viz"] = {}
            graph.nodes[node]["viz"]["size"] = size
            r, g, b = degree_to_rgb[degrees["total_degree"]]
            graph.nodes[node]["viz"]["color"] = {"r": r, "g": g, "b": b}
    except Exception:
        logger.exception("Error in _add_node_sizes_by_degree: ")
        raise


def main():
    parser = argparse.ArgumentParser(description="Users network analysis")
    parser.add_argument(
        "mode",
        choices=["find", "network", "both"],
        default="network",
        nargs="?",
        help="Mode: 'find' to collect users, 'network' to create graph, 'both' to do both (default: network)",
    )
    args = parser.parse_args()

    try:
        if args.mode in ["find", "both"]:
            print(
                f"[COLLECTING USERS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            )
            find_users()

        if args.mode in ["network", "both"]:
            print(
                f"[CREATING NETWORK GRAPH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            )
            graph = create_user_network_graph(compute_sizes=COMPUTE_SIZES_BY_DEGREE)
            save_graph_to_gephi(graph, "users_network.gexf")
            print(
                f"[GRAPH SAVED TO users_network.gexf - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            )
    except Exception:
        logger.exception("Main execution failed: ")
        raise


if __name__ == "__main__":
    main()
