# DMML Skill Tree

The skill tree is now data-driven:

- `tree.json` contains the nodes, edges, week numbers, labels, bullet points, libraries, and positions.
- `render.py` turns that data into PNG/PDF outputs.
- `skill_tree.py` is a small compatibility wrapper.

## Common Commands

From the repository root:

```bash
# Full printable/reference version
./env/bin/python skill_tree/render.py --full

# Weekly Canvas/slide image: future topics are dimmed, current week glows
./env/bin/python skill_tree/render.py --week 6 --highlight-current

# Weekly image with future topics hidden
./env/bin/python skill_tree/render.py --week 6 --highlight-current --future hide

# Regenerate every weekly view
./env/bin/python skill_tree/render.py --all-weeks --highlight-current
```

Outputs are written to `skill_tree/out/` by default.

## Editing

To add a topic, add a node to `tree.json`, give it a unique `id`, a `week`,
`branch`, `position`, and then connect it in `edges`.

Branches currently available:

- `tab`
- `ts`
- `unsup`
- `img`
- `seq`
- `shap_special`

Use `kind: "header"` for branch headers. Ordinary method cards omit `kind`.
