import os
import json
import webbrowser
from mastodon import Mastodon
from datetime import datetime
# Helper functions to help with the project


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
        print(f"Error in _auth: {e}")


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
        print(f"Error in get_client: {e}")
        return None
