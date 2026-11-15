"""Estimate the app's mine density from screenshots of cleared sectors.

Usage: python scripts/screenshot_density.py shot1.png shot2.png ...

Needs Pillow and numpy. Sector boundaries are the long pink lines; each sector
is cut into 8x8 cells. A cell is a flag if its centre is about half white with
pink in it, and revealed if it is plain light gray. Sectors with any other
kind of cell (unfinished, dark, covered by an icon) are skipped, and sectors
seen in more than one screenshot are counted once.
"""
import sys, json
from PIL import Image
import numpy as np
def peaks(v,th=0.45):
    idx=np.where(v>th)[0]; out=[]
    for i in idx:
        if out and i-out[-1][-1]<=2: out[-1].append(i)
        else: out.append([i])
    return [sum(o)/len(o) for o in out]
def sectors(path):
    im=np.asarray(Image.open(path).convert("RGB")).astype(int)
    r,g,b=im[...,0],im[...,1],im[...,2]
    pink=(r>200)&(g<170)&(b<200)&(r-g>60)
    xs=peaks(pink.mean(0)); ys=peaks(pink.mean(1))
    out=[]
    for j in range(len(ys)-1):
        for i in range(len(xs)-1):
            x0,x1,y0,y1=xs[i],xs[i+1],ys[j],ys[j+1]
            cw=(x1-x0)/8; ch=(y1-y0)/8
            grid=[]; ok=True
            for cy in range(8):
                rowc=[]
                for cx in range(8):
                    a=int(x0+cx*cw+cw*0.2); bb=int(x0+(cx+1)*cw-cw*0.2)
                    c=int(y0+cy*ch+ch*0.2); d=int(y0+(cy+1)*ch-ch*0.2)
                    patch=im[c:d,a:bb].reshape(-1,3)
                    white=((patch.min(1)>235)).mean()
                    pk=((patch[:,0]>200)&(patch[:,0]-patch[:,1]>50)).mean()
                    lum=patch.mean()
                    sat=(patch.max(1)-patch.min(1)).mean()
                    rowc.append((white,pk,lum,sat))
                grid.append(rowc)
            out.append(((i,j),grid))
    return out
def cleared_patterns(paths):
    seen = {}
    for p in paths:
        for ij, grid in sectors(p):
            pat, ok = [], True
            for row in grid:
                for white, pk, lum, _ in row:
                    if 0.25 <= white <= 0.75 and pk > 0.1:
                        pat.append(1)
                    elif white < 0.05 and pk < 0.03 and 165 <= lum <= 220:
                        pat.append(0)
                    else:
                        ok = False
            if ok:
                seen.setdefault(tuple(pat), (p, ij))
    return list(seen)


if __name__ == "__main__":
    pats = cleared_patterns(sys.argv[1:])
    counts = np.array([sum(p) for p in pats])
    n = len(counts)
    p = counts.sum() / (64 * n)
    se = (p * (1 - p) / (64 * n)) ** 0.5
    print(f"cleared sectors: {n}")
    print(f"mines: {counts.sum()} of {64 * n} cells")
    print(f"density: {p:.3f}  (95% interval {p - 1.96 * se:.3f} to {p + 1.96 * se:.3f})")
    print(f"mines per sector: min {counts.min()}, max {counts.max()}, mean {counts.mean():.2f}")
    print(f"variance: {counts.var(ddof=1):.2f}  (binomial would be {64 * p * (1 - p):.2f})")