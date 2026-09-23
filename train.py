#!/usr/bin/env python3
"""Simple entry point for training a poker AI."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_poker import main

if __name__ == "__main__":
    main()
