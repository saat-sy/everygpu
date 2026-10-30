"""Module entrypoint: python -m client."""

import sys

from client.server import serve


def main() -> int:
    """Run the client server."""
    return serve.run()


if __name__ == "__main__":
    sys.exit(main())
