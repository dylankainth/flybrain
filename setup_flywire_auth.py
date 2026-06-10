"""
Setup FlyWire authentication for CAVEclient.
This allows downloading the full connectome.
"""

from caveclient import CAVEclient
import os

def setup_auth():
    """
    Initialize CAVEclient with FlyWire authentication.
    Requires manual token setup via web browser.
    """

    print("="*60)
    print("FlyWire Authentication Setup")
    print("="*60)

    try:
        # Try to get a global client to prompt for auth
        print("\n[*] Initializing CAVEclient...")
        client = CAVEclient(server_address="https://global.daf-apis.com")

        print("\n[*] Checking for existing credentials...")
        try:
            # This will prompt for auth if needed
            client.auth.get_new_token()
            print("[OK] Token obtained successfully!")
            return True
        except Exception as e:
            print(f"[!] Auth error: {e}")
            print("\n[*] Manual setup required:")
            print("    1. Go to: https://flywire.ai/")
            print("    2. Sign in with your account")
            print("    3. Copy your API token from settings")
            print("    4. Follow the CAVEclient documentation at:")
            print("       https://caveconnectome.github.io/CAVEclient/tutorials/authentication/")
            return False

    except Exception as e:
        print(f"[!] Error: {e}")
        return False


if __name__ == "__main__":
    success = setup_auth()
    if success:
        print("\n[OK] Ready to download full connectome!")
        print("     Run: python download_real_connectome.py --full")
    else:
        print("\n[!] Authentication failed. Set up credentials manually first.")
