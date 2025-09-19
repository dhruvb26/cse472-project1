import json
from helpers import get_client, save_json
from mastodon import Mastodon
import sys
import os


def _verify_users(path: str = "users.json") -> int:
    """
    Verify the number of users saved in the given JSON file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be an array of users")

    return len(data)


def find_users(
    followers_limit: int = 10,
    following_limit: int = 10,
    total_users_limit: int = 200,
    show_progress: bool = True,
):
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
            except Exception as e:
                print(f"Error resolving seed user {user_handle}: {e}")

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
    except Exception as e:
        print(f"Error in find_users: {e}")
        return []


def main():
    users = find_users()
    save_json(users, "users.json")


if __name__ == "__main__":
    main()
