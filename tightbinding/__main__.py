"""Allow running as: python -m tightbinding input.yaml"""
import sys
from .main import main

if len(sys.argv) < 2:
    print("Usage: python -m tightbinding <config.yaml>")
    sys.exit(1)

main(sys.argv[1])
