"""PRISM-Min: minimal, attribution-capable LLM agent loop for photonic device design.

Loop per iteration:  hypothesis -> edit design.py -> DRC/sanity -> model.py prediction
registered -> simulate -> loss breakdown + closure -> keep/revert -> journal.
One rule: prediction miss > 2 sigma  =>  next turn revises model.py, not the design.
"""

__version__ = "0.2.0"
