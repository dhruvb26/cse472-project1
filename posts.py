import json
import os
from helpers import get_client, save_json
from mastodon import Mastodon
import sys


def _verify_posts(path: str = "posts.json") -> int:
    """
    Verify the number of posts saved in the given JSON file.

    - Ensures the file exists, otherwise raises FileNotFoundError.
    - Ensures the top-level JSON value is an array and returns its length.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be an array of posts")

    return len(data)


def _load_existing_posts(filename: str = "posts.json") -> tuple[list[dict], set[str]]:
    """
    Load existing posts from JSON file and return posts list and set of existing IDs.
    """
    if not os.path.exists(filename):
        return [], set()

    try:
        with open(filename, "r") as f:
            posts = json.load(f)
        if not isinstance(posts, list):
            return [], set()

        # Extract existing post IDs
        existing_ids = {post.get("id") for post in posts if post.get("id")}
        return posts, existing_ids
    except Exception:
        return [], set()


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
            raise Exception("Some error occurred when finding the client")

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
                    except Exception as ctx_err:
                        print(f"Error fetching context for post {post_id}: {ctx_err}")

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
    except Exception as e:
        print(f"Error in find_posts: {e}")
        return []


def main():
    posts = find_posts()
    print(f"Total posts collected: {len(posts)}")


if __name__ == "__main__":
    main()
