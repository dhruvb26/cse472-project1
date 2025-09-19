import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import networkx as nx
from mastodon import Mastodon

from helpers import (
    build_community_color_map,
    get_client,
    save_graph_to_gephi,
    save_json,
)

logger = logging.getLogger(__name__)

COMPUTE_SIZES_BY_DEGREE: bool = False


def _verify_posts(path: str = "posts.json") -> int:
    """
    Verify the number of posts saved in the given JSON file.

    - Ensures the file exists, otherwise raises FileNotFoundError.
    - Ensures the top-level JSON value is an array and returns its length.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be an array of posts.")

        return len(data)
    except Exception:
        logger.exception("Error in _verify_posts: ")
        raise


def _load_existing_posts(filename: str = "posts.json") -> tuple[list[dict], set[str]]:
    """
    Load existing posts from JSON file and return posts list and set of existing IDs.
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        with open(filename, "r") as f:
            posts = json.load(f)
        if not isinstance(posts, list):
            raise ValueError("Top-level JSON must be an array of posts.")

        existing_ids = {post.get("id") for post in posts if post.get("id")}
        return posts, existing_ids
    except Exception:
        logger.exception("Error in _load_existing_posts: ")
        raise


def _save_post_incrementally(
    post: dict,
    replies: list[dict],
    filename: str = "posts.json",
    existing_ids: set[str] | None = None,
) -> set[str]:
    """
    Save a post and its replies to JSON file incrementally, avoiding duplicates.
    Returns updated set of existing IDs.
    """
    try:
        if existing_ids is None:
            existing_ids = set()

        # Load current posts
        current_posts, _ = _load_existing_posts(filename)

        # Add main post if not duplicate
        post_id = post.get("id")
        if post_id and post_id not in existing_ids:
            current_posts.append(post)
            existing_ids.add(post_id)

        # Add replies if not duplicates
        for reply in replies:
            reply_id = reply.get("id")
            if reply_id and reply_id not in existing_ids:
                current_posts.append(reply)
                existing_ids.add(reply_id)

        # Save updated posts
        save_json(current_posts, filename)
        return existing_ids
    except Exception:
        logger.exception("Error in _save_post_incrementally: ")
        raise


def find_posts(
    hashtag_limit: int = 10, post_limit: int = 100, show_progress: bool = True
) -> list[dict]:
    """
    Finds posts with the given hashtags and includes their replies (descendants).
    Filters out any posts where replies_count < 3 (or missing).
    Processes replies immediately and saves incrementally to avoid duplicates.
    """
    try:
        client: Mastodon | None = get_client()

        if client is None:
            raise Exception("Some error occurred when finding the client.")

        with open("data.json", "r") as f:
            data = json.load(f)
        hashtags = data["keywords"]

        # Load existing posts and IDs to avoid duplicates
        all_posts, existing_ids = _load_existing_posts("posts.json")

        # Determine how many hashtags we'll actually process
        num_hashtags_to_process = min(hashtag_limit, len(hashtags))

        # Distribute post_limit across hashtags
        posts_per_hashtag = post_limit // num_hashtags_to_process
        remaining_posts = post_limit % num_hashtags_to_process

        processed_original_posts = 0
        qualifying_posts_count = len(all_posts)

        for i, hashtag in enumerate(hashtags):
            if i >= hashtag_limit:
                break

            # Give some hashtags one extra post if there's a remainder
            target_posts = posts_per_hashtag + (1 if i < remaining_posts else 0)

            # Track posts found for this hashtag
            hashtag_posts_found = 0
            max_id = None

            # Continue fetching until we have enough qualifying posts for this hashtag
            while hashtag_posts_found < target_posts:
                # API typically caps at 40 posts per call
                # Fetch more than needed since we'll filter some out
                batch_size = min(40, max(40, (target_posts - hashtag_posts_found) * 2))

                batch = client.timeline_hashtag(
                    hashtag, limit=batch_size, max_id=max_id
                )

                if not batch:  # No more posts available
                    break

                processed_original_posts += len(batch)

                # Process each post individually
                for post in batch:
                    if hashtag_posts_found >= target_posts:
                        break

                    replies_count = post.get("replies_count", 0)
                    post_id = post.get("id")

                    # Skip if doesn't meet criteria or is duplicate
                    if replies_count < 3 or not post_id or post_id in existing_ids:
                        continue

                    # Fetch replies immediately for this post
                    qualifying_replies = []
                    try:
                        context = client.status_context(post_id)
                        descendants = context.get("descendants", [])
                        if descendants:
                            # Filter for first-level replies only (replies directly to this post)
                            # and avoid duplicates - no replies_count filter for replies
                            qualifying_replies = [
                                desc
                                for desc in descendants
                                if desc.get("in_reply_to_id") == post_id
                                and desc.get("id") not in existing_ids
                            ]
                    except Exception:
                        logger.exception(f"Error fetching context for post {post_id}")
                        raise

                    # Save post and its replies incrementally
                    existing_ids = _save_post_incrementally(
                        post, qualifying_replies, "posts.json", existing_ids
                    )
                    hashtag_posts_found += 1
                    qualifying_posts_count += 1 + len(qualifying_replies)

                    if show_progress:
                        sys.stdout.write(
                            f"\rPosts processed: {processed_original_posts} | saved: {qualifying_posts_count} | hashtag: {hashtag} ({hashtag_posts_found}/{target_posts})"
                        )
                        sys.stdout.flush()

                # Prepare for next page (get oldest ID from this batch)
                if batch:
                    max_id = min(post.get("id", "") for post in batch if post.get("id"))

        if show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Load final posts to return
        final_posts, _ = _load_existing_posts("posts.json")
        return final_posts
    except Exception:
        logger.exception("Error in find_posts: ")
        raise


def create_network_graph(
    filename: str = "posts.json", compute_sizes: bool = True
) -> nx.DiGraph:
    """
    Create a directed network graph from posts data with community detection based on tags.

    Each node represents a toot (post or reply). A directed edge between nodes
    indicates the direction of information propagation, where an edge from A to B
    means A is replying to B.

    Args:
        filename: Path to the JSON file containing posts data
        compute_sizes: If True, compute node sizes based on degree; otherwise skip

    Returns:
        nx.DiGraph: A directed graph where nodes are post IDs and edges represent reply relationships

    Raises:
        FileNotFoundError: If the posts file doesn't exist
        ValueError: If the JSON structure is invalid
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Posts file not found: {filename}")

        try:
            with open(filename, "r") as f:
                posts = json.load(f)
        except json.JSONDecodeError as e:
            logger.exception(f"JSON decode error in {filename}: ")
            # Promote JSON decode errors to ValueError with context
            raise ValueError(f"Invalid JSON in {filename}: {e}")

        if not isinstance(posts, list):
            raise ValueError("Posts data must be a list of post objects.")

        G = nx.DiGraph()

        community_posts = defaultdict(list)
        post_to_community = {}

        for post in posts:
            post_id = post.get("id")
            if post_id:
                tags = post.get("tags", [])
                first_tag = (
                    tags[0].get("name") if tags and len(tags) > 0 else "untagged"
                )

                community_posts[first_tag].append(post_id)
                post_to_community[post_id] = first_tag

        communities = list(community_posts.keys())
        community_colors = build_community_color_map(communities)

        for post in posts:
            post_id = post.get("id")
            if post_id:
                community = post_to_community.get(post_id, "untagged")
                color = community_colors.get(community, "#e5e4df")

                G.add_node(
                    post_id,
                    **{
                        "url": post.get("url", ""),
                        "content": post.get("content", ""),
                        "created_at": post.get("created_at", ""),
                        "replies_count": post.get("replies_count", 0),
                        "username": post.get("account", {}).get("username", ""),
                        "community": community,
                        "color": color,
                        "viz": {
                            "color": {
                                "r": int(color[1:3], 16),
                                "g": int(color[3:5], 16),
                                "b": int(color[5:7], 16),
                            }
                        },
                    },
                )
        for post in posts:
            post_id = post.get("id")
            in_reply_to_id = post.get("in_reply_to_id")

            if post_id and in_reply_to_id and in_reply_to_id in G.nodes:
                G.add_edge(post_id, in_reply_to_id)

        if compute_sizes:
            _add_node_sizes_by_degree(G)

        return G
    except Exception:
        logger.exception("Error in create_network_graph: ")
        raise


def _add_node_sizes_by_degree(
    graph: nx.DiGraph, min_size: int = 5, max_size: int = 50
) -> None:
    """
    Calculate degree (in + out) for each node, normalize it, and add size attributes.

    Args:
        graph: NetworkX DiGraph to modify
        min_size: Minimum node size
        max_size: Maximum node size
    """
    try:
        if not graph.nodes():
            return

        node_degrees = {}
        for node in graph.nodes():
            in_degree = graph.in_degree[node]
            out_degree = graph.out_degree[node]
            total_degree = in_degree + out_degree
            node_degrees[node] = {
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": total_degree,
            }

        total_degrees = [data["total_degree"] for data in node_degrees.values()]
        max_degree = max(total_degrees)

        if max_degree == 0:
            max_degree = 1

        for node, degrees in node_degrees.items():
            # Normalize by maximum degree: C_d^max(v_i) = d_i / max_j d_j
            normalized_degree = degrees["total_degree"] / max_degree
            size = min_size + normalized_degree * (max_size - min_size)

            graph.nodes[node].update(
                {
                    "in_degree": degrees["in_degree"],
                    "out_degree": degrees["out_degree"],
                    "total_degree": degrees["total_degree"],
                    "size": size,
                }
            )

            if "viz" not in graph.nodes[node]:
                graph.nodes[node]["viz"] = {}
            graph.nodes[node]["viz"]["size"] = size
    except Exception:
        logger.exception("Error in _add_node_sizes_by_degree: ")
        raise


def main():
    parser = argparse.ArgumentParser(description="Posts network analysis")
    parser.add_argument(
        "mode",
        choices=["find", "network", "both"],
        default="network",
        nargs="?",
        help="Mode: 'find' to collect posts, 'network' to create graph, 'both' to do both (default: network)",
    )
    args = parser.parse_args()

    try:
        if args.mode in ["find", "both"]:
            print(
                f"[COLLECTING POSTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            )
            find_posts()

        if args.mode in ["network", "both"]:
            print(
                f"[CREATING NETWORK GRAPH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            )
            graph = create_network_graph(compute_sizes=COMPUTE_SIZES_BY_DEGREE)
            save_graph_to_gephi(graph, "posts_network.gexf")
            print(
                f"[GRAPH SAVED TO posts_network.gexf - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            )
    except Exception:
        logger.exception("Main execution failed: ")
        raise


if __name__ == "__main__":
    main()
