#!/usr/bin/env python3
"""
VideoAnnotator Token Management CLI

User-friendly command-line tool for managing API tokens, authentication,
and user access to the VideoAnnotator API server.

Usage:
    python scripts/manage_tokens.py create --user john@example.com
    python scripts/manage_tokens.py list --user john@example.com
    python scripts/manage_tokens.py revoke --token va_api_xyz123
    python scripts/manage_tokens.py cleanup
    python scripts/manage_tokens.py stats
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth import get_token_manager, initialize_token_manager
from src.utils.logging_config import get_logger, setup_videoannotator_logging


def setup_logging():
    """Setup logging for token management."""
    setup_videoannotator_logging(log_level="INFO")
    return get_logger("api")


def create_api_key(args):
    """Create a new API key."""
    logger = get_logger("api")
    token_manager = get_token_manager()

    # Get user information
    email = args.user
    if not email:
        email = input("Enter user email: ")

    username = (
        args.username
        or input(f"Enter username (default: {email.split('@')[0]}): ")
        or email.split("@")[0]
    )

    # Get scopes
    available_scopes = ["read", "write", "admin", "debug"]
    if args.scopes:
        scopes = args.scopes.split(",")
    else:
        print("\nAvailable scopes:", ", ".join(available_scopes))
        scope_input = (
            input("Enter scopes (comma-separated, default: read,write): ")
            or "read,write"
        )
        scopes = [s.strip() for s in scope_input.split(",")]

    # Validate scopes
    invalid_scopes = set(scopes) - set(available_scopes)
    if invalid_scopes:
        print(f"[ERROR] Invalid scopes: {invalid_scopes}")
        return False

    # Get expiration
    expires_in_days = args.expires_days
    if expires_in_days is None and not args.no_expiry:
        try:
            days_input = (
                input("Expires in days (default: 365, 0 for no expiry): ") or "365"
            )
            expires_in_days = int(days_input) if days_input != "0" else None
        except ValueError:
            print("[ERROR] Invalid number of days")
            return False

    # Generate token
    try:
        user_id = f"user_{hash(email) % 100000}"
        metadata = {
            "created_via": "cli",
            "description": args.description or f"API key for {username}",
        }

        token, token_info = token_manager.generate_api_key(
            user_id=user_id,
            username=username,
            email=email,
            scopes=scopes,
            expires_in_days=expires_in_days,
            metadata=metadata,
        )

        # Display results
        print("\n" + "=" * 60)
        print("[SUCCESS] API Key Created Successfully!")
        print("=" * 60)
        print(f"Token:      {token}")
        print(f"User:       {username} ({email})")
        print(f"Scopes:     {', '.join(scopes)}")
        print(f"Created:    {token_info.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if token_info.expires_at:
            print(f"Expires:    {token_info.expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("Expires:    Never")

        print("\n[IMPORTANT] Save this token securely - it cannot be recovered!")
        print("=" * 60)

        # Save to file if requested
        if args.output_file:
            token_data = {
                "token": token,
                "username": username,
                "email": email,
                "scopes": scopes,
                "created_at": token_info.created_at.isoformat(),
                "expires_at": token_info.expires_at.isoformat()
                if token_info.expires_at
                else None,
            }

            with open(args.output_file, "w") as f:
                json.dump(token_data, f, indent=2)
            print(f"[INFO] Token details saved to {args.output_file}")

        logger.info(f"Created API key for {username} via CLI")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to create token: {e}")
        logger.error(f"Token creation failed: {e}")
        return False


def list_tokens(args):
    """List tokens for a user or all tokens."""
    token_manager = get_token_manager()

    try:
        if args.user:
            # Find user by email
            user_tokens = []
            for token_info in token_manager._token_cache.values():
                if token_info.email == args.user and token_info.is_active:
                    user_tokens.append(token_info)
            tokens = user_tokens
        else:
            # List all active tokens
            tokens = [t for t in token_manager._token_cache.values() if t.is_active]

        if not tokens:
            user_msg = f" for {args.user}" if args.user else ""
            print(f"[INFO] No active tokens found{user_msg}")
            return True

        # Display tokens
        print(f"\n{'=' * 80}")
        print(f"Active API Tokens ({len(tokens)} found)")
        print(f"{'=' * 80}")

        for i, token in enumerate(tokens, 1):
            print(f"\n{i}. Token ID: {token.token_id[:20]}...")
            print(f"   User:      {token.username} ({token.email})")
            print(f"   Type:      {token.token_type.value}")
            print(f"   Scopes:    {', '.join(token.scopes)}")
            print(f"   Created:   {token.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

            if token.expires_at:
                days_left = (token.expires_at - datetime.now()).days
                status = "EXPIRED" if days_left < 0 else f"{days_left} days left"
                print(
                    f"   Expires:   {token.expires_at.strftime('%Y-%m-%d %H:%M:%S')} ({status})"
                )
            else:
                print("   Expires:   Never")

            if token.last_used_at:
                print(
                    f"   Last Used: {token.last_used_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                print("   Last Used: Never")

            if args.show_metadata and token.metadata:
                print(f"   Metadata:  {json.dumps(token.metadata, indent=14)}")

        print(f"\n{'=' * 80}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to list tokens: {e}")
        return False


def revoke_token(args):
    """Revoke a token."""
    token_manager = get_token_manager()

    try:
        # Confirm revocation
        if not args.force:
            confirm = input(
                f"Are you sure you want to revoke token {args.token[:20]}...? [y/N]: "
            )
            if confirm.lower() != "y":
                print("[INFO] Token revocation cancelled")
                return True

        success = token_manager.revoke_token(args.token)

        if success:
            print(f"[SUCCESS] Token {args.token[:20]}... has been revoked")
            get_logger("api").info(f"Token revoked via CLI: {args.token[:20]}...")
            return True
        else:
            print(f"[ERROR] Token not found or already revoked: {args.token[:20]}...")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to revoke token: {e}")
        return False


def cleanup_expired(args):
    """Clean up expired tokens."""
    token_manager = get_token_manager()

    try:
        count = token_manager.cleanup_expired_tokens()

        if count > 0:
            print(f"[SUCCESS] Cleaned up {count} expired tokens")
            get_logger("api").info(f"Cleaned up {count} expired tokens via CLI")
        else:
            print("[INFO] No expired tokens found")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to cleanup tokens: {e}")
        return False


def show_stats(args):
    """Show token statistics."""
    token_manager = get_token_manager()

    try:
        stats = token_manager.get_token_stats()

        print("\n" + "=" * 50)
        print("VideoAnnotator Token Statistics")
        print("=" * 50)
        print(f"Total Tokens:       {stats['total_tokens']}")
        print(f"Active Tokens:      {stats['active_tokens']}")
        print(f"Expired Tokens:     {stats['expired_tokens']}")
        print(f"Recently Used:      {stats['recently_used']} (last 24h)")

        print("\nTokens by Type:")
        for token_type, count in stats["by_type"].items():
            print(f"  {token_type:12} {count}")

        print("=" * 50)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to get statistics: {e}")
        return False


def validate_token_cmd(args):
    """Validate a token."""
    token_manager = get_token_manager()

    try:
        token_info = token_manager.validate_token(args.token)

        if token_info:
            print("\n" + "=" * 50)
            print("[SUCCESS] Token is VALID")
            print("=" * 50)
            print(f"User:      {token_info.username} ({token_info.email})")
            print(f"Type:      {token_info.token_type.value}")
            print(f"Scopes:    {', '.join(token_info.scopes)}")
            print(f"Created:   {token_info.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

            if token_info.expires_at:
                days_left = (token_info.expires_at - datetime.now()).days
                status = "EXPIRED" if days_left < 0 else f"{days_left} days left"
                print(
                    f"Expires:   {token_info.expires_at.strftime('%Y-%m-%d %H:%M:%S')} ({status})"
                )
            else:
                print("Expires:   Never")

            print("=" * 50)
            return True
        else:
            print("[ERROR] Token is INVALID or EXPIRED")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to validate token: {e}")
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="VideoAnnotator Token Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create API key for a user
  python scripts/manage_tokens.py create --user john@example.com --expires-days 365

  # Create long-lived admin token
  python scripts/manage_tokens.py create --user admin@company.com --scopes admin,read,write --no-expiry

  # List all tokens
  python scripts/manage_tokens.py list

  # List tokens for specific user
  python scripts/manage_tokens.py list --user john@example.com

  # Revoke a token
  python scripts/manage_tokens.py revoke --token va_api_xyz123... --force

  # Validate a token
  python scripts/manage_tokens.py validate --token va_api_xyz123...

  # Clean up expired tokens and show stats
  python scripts/manage_tokens.py cleanup
  python scripts/manage_tokens.py stats
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create token command
    create_parser = subparsers.add_parser("create", help="Create a new API token")
    create_parser.add_argument("--user", "-u", help="User email address")
    create_parser.add_argument("--username", help="Username (defaults to email prefix)")
    create_parser.add_argument(
        "--scopes", help="Comma-separated scopes (default: read,write)"
    )
    create_parser.add_argument(
        "--expires-days", type=int, help="Token expires in N days"
    )
    create_parser.add_argument(
        "--no-expiry", action="store_true", help="Create token without expiration"
    )
    create_parser.add_argument("--description", help="Token description")
    create_parser.add_argument("--output-file", help="Save token details to JSON file")

    # List tokens command
    list_parser = subparsers.add_parser("list", help="List API tokens")
    list_parser.add_argument("--user", "-u", help="Filter by user email")
    list_parser.add_argument(
        "--show-metadata", action="store_true", help="Show token metadata"
    )

    # Revoke token command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API token")
    revoke_parser.add_argument("--token", "-t", required=True, help="Token to revoke")
    revoke_parser.add_argument(
        "--force", "-f", action="store_true", help="Skip confirmation"
    )

    # Validate token command
    validate_parser = subparsers.add_parser("validate", help="Validate an API token")
    validate_parser.add_argument(
        "--token", "-t", required=True, help="Token to validate"
    )

    # Cleanup command
    subparsers.add_parser("cleanup", help="Clean up expired tokens")

    # Stats command
    subparsers.add_parser("stats", help="Show token statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize logging and token manager
    setup_logging()
    initialize_token_manager()

    # Execute command
    commands = {
        "create": create_api_key,
        "list": list_tokens,
        "revoke": revoke_token,
        "validate": validate_token_cmd,
        "cleanup": cleanup_expired,
        "stats": show_stats,
    }

    try:
        success = commands[args.command](args)
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n[INFO] Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
