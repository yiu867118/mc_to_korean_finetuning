# Fine-tuning t5-small: Middle Chinese -> Korean

    pip install torch transformers sentencepiece pandas numpy
    python run_diagnose.py   # 20 epochs; train/test loss and accuracy per epoch
    python run_spaced.py     # letters written apart ("d e n g")

- Prompt: "Middle Chinese {mc} to Korean:". Pairs from data/mcpdict.db (2,195 pairs; 1,976 train / 219 test, split by MC syllable, 90/10, seed 0).
- Pretrained t5-small, batch 32, learning rate 3e-4, exact-match accuracy with Wilson 95% CIs.
- First run downloads t5-small from Hugging Face. One run per setting, so small differences can be noise.
- data/mcpdict.db is from github.com/osfans/MCPDict (MIT license, see data/MCPDict_LICENSE.txt).
