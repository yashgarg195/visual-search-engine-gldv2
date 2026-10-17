#!/usr/bin/env python3
"""Run the GLDv2 Visual Search Engine FastAPI Web Server."""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="GLDv2 Visual Search FastAPI Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (use 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable live code reload for development")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(" 🏛️  GLDv2 VISUAL LANDMARK SEARCH ENGINE - FASTAPI WEB APP")
    print("=" * 75)
    print(f"  Local Address     : http://{args.host}:{args.port}")
    print(f"  Interactive Docs  : http://{args.host}:{args.port}/docs")
    print(f"  Indexed Database  : 5,052,992 vectors (train, index, test splits)")
    print("=" * 75 + "\n")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
