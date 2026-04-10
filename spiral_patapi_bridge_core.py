#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible wrapper for `sr_rewind_cot.py`."""

from sr_rewind_cot import *  # noqa: F401,F403
from sr_rewind_cot import main


if __name__ == "__main__":
    main()
