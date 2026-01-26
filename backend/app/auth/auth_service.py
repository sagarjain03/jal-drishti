from app.auth.users_seed import SEEDED_USERS

def authenticate_user(username: str, password: str):
    """
    Checks if the username exists and the password matches.
    Returns username if valid, False otherwise.
    """
    if username in SEEDED_USERS:
        if SEEDED_USERS[username] == password:
            return username
    return False
