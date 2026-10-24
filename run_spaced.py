"""Double-click / press Run: no command line needed.
Runs ml_finetune_t5_v2.py with the setting for this experiment (tag = spaced)."""
import os, sys, runpy
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["ml_finetune_t5_v2.py", "--tag", "spaced", "--spaced"]
runpy.run_path(os.path.join(here, "ml_finetune_t5_v2.py"), run_name="__main__")
