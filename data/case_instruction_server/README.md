# Case Instruction Server State (Runtime Only)

This directory is used as the persistent runtime state for the local case
instruction server.

It will contain generated artifacts such as:
- issued instructions
- generated case submissions
- training JSONL exports
- summary dashboards

Those files can grow very large during the 5000-case generation campaign, so
they are intentionally ignored by git (see `.gitignore` in this folder).

If you need to reset generation, you can delete the contents of this directory
while the server is stopped.
