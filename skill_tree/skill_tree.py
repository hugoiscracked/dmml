"""
Compatibility wrapper for the DMML skill-tree renderer.

Prefer calling:
  ./env/bin/python skill_tree/render.py --week 6 --highlight-current
"""

from render import main


if __name__ == "__main__":
    main()
