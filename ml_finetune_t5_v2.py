"""
Mini fine-tuning demo, version 2: the same experiment, with two switches to test improvements.

    pip install torch transformers sentencepiece pandas
    python ml_finetune_t5_v2.py --tag baseline                      # 8 epochs, letters written together (your first run)
    python ml_finetune_t5_v2.py --tag spaced   --spaced             # change 1: write letters apart  d e n g
    python ml_finetune_t5_v2.py --tag epochs20 --epochs 20          # change 2: train longer

Rules for a fair comparison
 * Same data, same train/test split, same seed in every run (the split code is identical to ml_finetune_t5.py).
 * Change ONE thing per run.
 * The test set is only used to measure; nothing is tuned on it.

Each run prints everything and also saves results_<tag>.json (accuracy, loss and test accuracy after every epoch,
examples) so you can paste it back.
Options:
  --spaced        write the Middle Chinese syllable letter by letter: "d e n g"
  --spaced_out    also write the Korean answer letter by letter (spaces are removed again before scoring)
  --epochs N      number of passes over the learning set (default 8)
  --batch N       group size per adjustment (default 32)
  --lr X          learning rate (default 3e-4)
Expect about 1 minute per epoch on a laptop CPU, so --epochs 20 takes about 20 minutes.
"""
import argparse, json, os, re, sqlite3, random, time
import pandas as pd, torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="run")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--spaced", action="store_true", help='write MC letters apart: "d e n g"')
ap.add_argument("--diag", action="store_true", help="also record train-set accuracy and held-out loss every epoch (to see under/over-fitting)")
ap.add_argument("--spaced_out", action="store_true", help="also write the Korean answer letter by letter")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))     # data/ and results are looked up next to this file
random.seed(0); torch.manual_seed(0)
MODEL, N_TRAIN = "t5-small", 3000

# ---------------------------------------------------------------- 1. data (identical to ml_finetune_t5.py)
pairs = set()
for mc, kr in sqlite3.connect(os.path.join(HERE, "data", "mcpdict.db")).execute(
        "select mc, kr from mcpdict where mc is not null and kr is not null"):
    if re.fullmatch(r"[a-z]+", mc) and re.fullmatch(r"[a-z]+", kr):
        pairs.add((mc, kr))
df = pd.DataFrame(sorted(pairs), columns=["mc", "kr"])
df = df.drop_duplicates("mc", keep=False)
mcs = list(df.mc); random.shuffle(mcs)
test_mc = set(mcs[: int(0.1 * len(mcs))])
train = df[~df.mc.isin(test_mc)].sample(frac=1, random_state=0).head(N_TRAIN)
test = df[df.mc.isin(test_mc)].head(300)
print(f"[{args.tag}] train pairs {len(train)} | test pairs {len(test)} | epochs {args.epochs} | spaced={args.spaced} spaced_out={args.spaced_out}")

sp = lambda w: " ".join(w)
prompt = lambda mc: f"Middle Chinese {sp(mc) if args.spaced else mc} to Korean:"
target = lambda kr: sp(kr) if args.spaced_out else kr
clean = lambda s: s.replace(" ", "") if args.spaced_out else s.strip()

tok = AutoTokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL)
print("example input :", prompt("deng"), "| example target:", target("jeong"))
print("example tokens:", tok.tokenize(prompt("deng")))

def predict(mc_list):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(mc_list), 64):
            enc = tok([prompt(m) for m in mc_list[i:i+64]], return_tensors="pt", padding=True)
            gen = model.generate(**enc, max_new_tokens=24, num_beams=1)
            out += [clean(s) for s in tok.batch_decode(gen, skip_special_tokens=True)]
    return out

def accuracy():
    pred = predict(list(test.mc))
    return sum(p == g for p, g in zip(pred, test.kr)) / len(test), pred

def mean_loss(df_):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for i in range(0, len(df_), 64):
            part = df_.iloc[i:i + 64]
            enc = tok([prompt(m) for m in part.mc], return_tensors="pt", padding=True)
            labels = tok([target(k) for k in part.kr], return_tensors="pt", padding=True).input_ids
            labels[labels == tok.pad_token_id] = -100
            tot += model(**enc, labels=labels).loss.item() * len(part)
    return tot / len(df_)

train_probe = train.head(len(test))        # as many learning-set questions as there are test questions

def train_accuracy():
    pred = predict(list(train_probe.mc))
    return sum(p == g for p, g in zip(pred, train_probe.kr)) / len(train_probe)

res = {"tag": args.tag, "epochs": args.epochs, "spaced": args.spaced, "spaced_out": args.spaced_out,
       "batch": args.batch, "lr": args.lr, "n_train": len(train), "n_test": len(test)}

# ---------------------------------------------------------------- 2. BEFORE training
demo = list(train.head(6).itertuples(index=False))
print("\n[LEARNING-SET examples BEFORE training]")
res["learning_examples_before"] = []
for r, p_ in zip(demo, predict([r.mc for r in demo])):
    print(f"   {r.mc:8s} gold={r.kr:8s} model={p_}"); res["learning_examples_before"].append([r.mc, r.kr, p_])
acc0, pred0 = accuracy()
print(f"\n[BEFORE] exact-match accuracy on the {len(test)} test syllables: {acc0:.3f}")
res["acc_before"] = acc0

# ---------------------------------------------------------------- 3. supervised fine-tuning
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
data = list(zip(train.mc, train.kr))
res["loss"], res["test_acc_per_epoch"] = [], []
res["train_acc_per_epoch"], res["train_loss_eval"], res["test_loss"] = [], [], []
for epoch in range(1, args.epochs + 1):
    model.train(); random.shuffle(data); tot, t0 = 0.0, time.time()
    for i in range(0, len(data), args.batch):
        batch = data[i:i + args.batch]
        enc = tok([prompt(m) for m, _ in batch], return_tensors="pt", padding=True)
        labels = tok([target(k) for _, k in batch], return_tensors="pt", padding=True).input_ids
        labels[labels == tok.pad_token_id] = -100
        loss = model(**enc, labels=labels).loss          # compare with the gold answer
        loss.backward(); opt.step(); opt.zero_grad()      # adjust the weights
        tot += loss.item() * len(batch)
    a, _ = accuracy()                                     # measure only; nothing is tuned on it
    res["loss"].append(round(tot / len(data), 3)); res["test_acc_per_epoch"].append(round(a, 3))
    extra = ""
    if args.diag:
        ta, trl, tel = train_accuracy(), mean_loss(train_probe), mean_loss(test)
        res["train_acc_per_epoch"].append(round(ta, 3)); res["train_loss_eval"].append(round(trl, 3)); res["test_loss"].append(round(tel, 3))
        extra = f"  | train accuracy {ta:.3f}  train loss(eval) {trl:.3f}  test loss {tel:.3f}"
    print(f"epoch {epoch:2d}/{args.epochs}  training loss {tot/len(data):.3f}  test accuracy {a:.3f}{extra}  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- 4. AFTER training
acc1, pred1 = accuracy()
res["acc_after"] = acc1
res["exact_matches"] = int(round(acc1 * len(test)))
res["test_examples"] = [[m, g, b, a_] for m, g, b, a_ in list(zip(test.mc, test.kr, pred0, pred1))[:12]]
print("\n[TEST examples]  question | gold | before | after")
for m, g, b, a_ in res["test_examples"]:
    print(f"   {m:8s} | {g:8s} | {b:16s} | {a_}")
print(f"\n[{args.tag}] accuracy {acc0:.3f} -> {acc1:.3f}  ({res['exact_matches']}/{len(test)} exact)")
json.dump(res, open(os.path.join(HERE, f"results_{args.tag}.json"), "w"), ensure_ascii=False, indent=1)
print(f"saved results_{args.tag}.json")
