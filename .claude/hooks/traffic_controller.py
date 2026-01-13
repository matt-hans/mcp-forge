#!/usr/bin/env python3
"""Traffic controller hook for Claude session management."""

import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    """Handle session stop event."""
    try:
        # Log that hook executed
        hook_dir = Path(__file__).parent
        log_file = hook_dir / "traffic.log"
        
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} - Session stop hook executed\n")
        
        return 0
    except Exception as e:
        print(f"Traffic controller error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
