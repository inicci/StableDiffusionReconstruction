import argparse, subprocess, sys, pathlib
p=argparse.ArgumentParser()
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--list", type=str, required=True)
args=p.parse_args()

ids=[int(x) for x in open(args.list) if x.strip()]
for i in ids:
    subprocess.run([sys.executable, "img2feat_sd.py",
                    "--imgidx", str(i), str(i+1),
                    "--gpu", str(args.gpu)],
                   check=True)