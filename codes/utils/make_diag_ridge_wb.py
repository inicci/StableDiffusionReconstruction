import numpy as np
import argparse, os

def load_arr(path):
    """支持 (N,77,768) 或 (N,59136)；返回 (N,59136) 的 float64"""
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[1:] == (77, 768):
        arr = arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected shape for {path}: {arr.shape}")
    return arr.astype(np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_pred", required=True, help="预测的文本嵌入 (N,77,768) 或 (N,59136)")
    ap.add_argument("--y_tgt",  required=True, help="目标文本嵌入 (N,77,768) 或 (N,59136)")
    ap.add_argument("--out",    required=True, help="输出 npz 路径")
    ap.add_argument("--lam",    type=float, default=1e-3, help="岭回归系数 λ")
    args = ap.parse_args()

    X = load_arr(args.x_pred)   # (N,D)
    Y = load_arr(args.y_tgt)    # (N,D)
    assert X.shape == Y.shape, f"Shape mismatch: {X.shape} vs {Y.shape}"
    N, D = X.shape
    print(f"[info] Fitting diagonal ridge on N={N}, D={D}, lambda={args.lam}")

    # 逐维闭式解
    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    vx = X.var(axis=0, ddof=1)
    cov = ((X - mx) * (Y - my)).mean(axis=0)

    w = cov / (vx + args.lam)
    b = my - w * mx

    # 存成 (77,768) 以便阅读，也可存成 (59136,)
    w = w.reshape(77, 768).astype(np.float32)
    b = b.reshape(77, 768).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, w=w, b=b)
    print("[ok] Saved to:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
