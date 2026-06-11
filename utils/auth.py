import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
]

_HERE = os.path.dirname(os.path.abspath(__file__))

TOKEN_PATH = os.path.join(_HERE, "..", "token.json")
CREDENTIALS_PATH = os.path.join(
    _HERE, "..", "client_secret_1044033218170-re4j4nea8s73toldtprjjn3o785gqmin.apps.googleusercontent.com.json"
)


def get_credentials(
    token_path: str = TOKEN_PATH,
    credentials_path: str = CREDENTIALS_PATH,
    scopes: list[str] = SCOPES,
) -> Credentials:
    """Load or refresh OAuth2 credentials, triggering browser login if needed.

    Args:
        token_path: Path to the cached token file.
        credentials_path: Path to the OAuth2 client secrets file.
        scopes: List of OAuth2 scopes to request.

    Returns:
        Valid Google OAuth2 Credentials object.
    """
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return creds