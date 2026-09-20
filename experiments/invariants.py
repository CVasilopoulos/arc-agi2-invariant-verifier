import numpy as np
from collections import Counter

FEATURES = ("h", "w", "bh", "bw", "nc", "nobj")
SYMMETRIES = ("sym_lr", "sym_ud", "sym_rot180", "sym_transpose")


def as_grid(g):
    return np.array(g, dtype=np.int16)


def background(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[np.argmax(c)])


def palette(g):
    return frozenset(int(v) for v in np.unique(g))


def bbox(g, bg):
    m = g != bg
    if not m.any():
        return 0, 0
    r = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    return int(r[-1] - r[0] + 1), int(c[-1] - c[0] + 1)


def n_objects(g, bg):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    n = 0
    for i in range(h):
        for j in range(w):
            if g[i, j] == bg or seen[i, j]:
                continue
            n += 1
            stack = [(i, j)]
            seen[i, j] = True
            while stack:
                a, b = stack.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p, q = a + da, b + db
                    if 0 <= p < h and 0 <= q < w and not seen[p, q] and g[p, q] == g[a, b]:
                        seen[p, q] = True
                        stack.append((p, q))
    return n


def features(x):
    bg = background(x)
    bh, bw = bbox(x, bg)
    return {"h": x.shape[0], "w": x.shape[1], "bh": bh, "bw": bw,
            "nc": len(palette(x)) - 1, "nobj": n_objects(x, bg)}


def is_subgrid(small, big):
    sh, sw = small.shape
    bh, bw = big.shape
    if sh > bh or sw > bw:
        return False
    for i in range(bh - sh + 1):
        for j in range(bw - sw + 1):
            if np.array_equal(big[i:i + sh, j:j + sw], small):
                return True
    return False


def symmetric(g, kind):
    if kind == "sym_lr":
        return np.array_equal(g, g[:, ::-1])
    if kind == "sym_ud":
        return np.array_equal(g, g[::-1, :])
    if kind == "sym_rot180":
        return np.array_equal(g, g[::-1, ::-1])
    if kind == "sym_transpose":
        return g.shape[0] == g.shape[1] and np.array_equal(g, g.T)
    raise ValueError(kind)


class TaskInvariants:
    def __init__(self, train_pairs):
        self.train = [(as_grid(x), as_grid(y)) for x, y in train_pairs]
        self.k = len(self.train)
        self.xf = [features(x) for x, _ in self.train]
        self._fit_shape()
        self._fit_palette()
        self._fit_counts()
        self._fit_content()
        self._fit_symmetry()

    def _fit_shape(self):
        self.shape_hyp = {0: [], 1: []}
        for d in (0, 1):
            ys = [y.shape[d] for _, y in self.train]
            if len(set(ys)) == 1:
                self.shape_hyp[d].append(("const", ys[0]))
            for f in FEATURES:
                us = [xf[f] for xf in self.xf]
                for a in range(1, 7):
                    for q in range(1, 4):
                        for b in range(-3, 4):
                            if all(a * u == q * (yv - b) for u, yv in zip(us, ys)):
                                self.shape_hyp[d].append((f, a, q, b))

    def shape_options(self, x):
        key = (x.shape, x.tobytes())
        cache = self.__dict__.setdefault("_fcache", {})
        if key not in cache:
            cache[key] = features(x)
        xf = cache[key]
        out = []
        for d in (0, 1):
            vals = set()
            for hyp in self.shape_hyp[d]:
                if hyp[0] == "const":
                    vals.add(hyp[1])
                else:
                    f, a, q, b = hyp
                    num = a * xf[f]
                    if num % q == 0:
                        vals.add(num // q + b)
            out.append(vals)
        return out

    def _fit_palette(self):
        adds = [palette(y) - palette(x) for x, y in self.train]
        rems = [palette(x) - palette(y) for x, y in self.train]
        self.add_union = frozenset().union(*adds)
        self.rem_union = frozenset().union(*rems)
        self.pal_hyp = []
        if len(set(adds)) == 1 and len(set(rems)) == 1:
            self.pal_hyp.append(("delta", adds[0], rems[0]))
        outs = [palette(y) for _, y in self.train]
        if len(set(outs)) == 1:
            self.pal_hyp.append(("const", outs[0]))

    def _fit_counts(self):
        self.counts = {}
        self.counts["hist_equal"] = all(Counter(x.ravel().tolist()) == Counter(y.ravel().tolist()) for x, y in self.train)
        nb = [(int((x != background(x)).sum()), int((y != background(x)).sum())) for x, y in self.train]
        self.counts["nonbg_equal"] = all(a == b for a, b in nb)
        self.counts["nonbg_ge"] = all(b >= a for a, b in nb)
        self.counts["nonbg_le"] = all(b <= a for a, b in nb)

    def _fit_content(self):
        same = all(x.shape == y.shape for x, y in self.train)
        self.content = {}
        if same:
            self.content["keep_nonbg"] = all(np.array_equal(y[x != background(x)], x[x != background(x)]) for x, y in self.train)
            self.content["keep_bg"] = all(np.all(y[x == background(x)] == background(x)) for x, y in self.train)
        else:
            self.content["keep_nonbg"] = False
            self.content["keep_bg"] = False
        nontrivial = any(x.shape != y.shape for x, y in self.train)
        self.content["crop_of_input"] = nontrivial and all(is_subgrid(y, x) for x, y in self.train)
        self.content["input_in_output"] = nontrivial and all(is_subgrid(x, y) for x, y in self.train)

    def _fit_symmetry(self):
        nonuniform = [len(palette(y)) > 1 for _, y in self.train]
        self.sym = {s: all(nonuniform) and all(symmetric(y, s) for _, y in self.train) for s in SYMMETRIES}

    def active(self):
        act = {}
        act["shape"] = bool(self.shape_hyp[0]) and bool(self.shape_hyp[1])
        act["palette_exact"] = bool(self.pal_hyp)
        act["palette_subset"] = True
        act["palette_keep"] = True
        for k, v in self.counts.items():
            act[k] = v
        for k, v in self.content.items():
            act[k] = v
        for k, v in self.sym.items():
            act[k] = v
        return act

    def check(self, x, y):
        x = as_grid(x)
        y = as_grid(y)
        act = self.active()
        res = {}
        if act["shape"]:
            hs, ws = self.shape_options(x)
            res["shape"] = y.shape[0] in hs and y.shape[1] in ws
        px, py = palette(x), palette(y)
        if act["palette_exact"]:
            ok = False
            for hyp in self.pal_hyp:
                if hyp[0] == "delta" and py == (px | hyp[1]) - hyp[2]:
                    ok = True
                if hyp[0] == "const" and py == hyp[1]:
                    ok = True
            res["palette_exact"] = ok
        res["palette_subset"] = py <= (px | self.add_union)
        res["palette_keep"] = (px - self.rem_union) <= py
        bgx = background(x)
        if act["hist_equal"]:
            res["hist_equal"] = Counter(x.ravel().tolist()) == Counter(y.ravel().tolist())
        a, b = int((x != bgx).sum()), int((y != bgx).sum())
        if act["nonbg_equal"]:
            res["nonbg_equal"] = a == b
        if act["nonbg_ge"]:
            res["nonbg_ge"] = b >= a
        if act["nonbg_le"]:
            res["nonbg_le"] = b <= a
        if act["keep_nonbg"]:
            res["keep_nonbg"] = x.shape == y.shape and np.array_equal(y[x != bgx], x[x != bgx])
        if act["keep_bg"]:
            res["keep_bg"] = x.shape == y.shape and bool(np.all(y[x == bgx] == bgx))
        if act["crop_of_input"]:
            res["crop_of_input"] = is_subgrid(y, x)
        if act["input_in_output"]:
            res["input_in_output"] = is_subgrid(x, y)
        for s in SYMMETRIES:
            if act[s]:
                res[s] = symmetric(y, s)
        return res


FAMILIES = ["shape", "palette_exact", "palette_subset", "palette_keep", "hist_equal", "nonbg_equal",
            "nonbg_ge", "nonbg_le", "keep_nonbg", "keep_bg", "crop_of_input", "input_in_output"] + list(SYMMETRIES)


def rerank(task_inv, x, candidates, families):
    passing, failing = [], []
    for c in candidates:
        r = task_inv.check(x, c)
        if all(r.get(f, True) for f in families):
            passing.append(c)
        else:
            failing.append(c)
    return passing + failing
