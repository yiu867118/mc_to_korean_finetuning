"""Under- vs over-fitting run: 20 epochs, same settings as the epochs20 run, plus extra measurements every epoch:
train-set accuracy, train loss and test loss measured the same way. tag = diag"""
import os, sys, runpy
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["ml_finetune_t5_v2.py", "--tag", "diag", "--epochs", "20", "--diag"]
runpy.run_path(os.path.join(here, "ml_finetune_t5_v2.py"), run_name="__main__")
