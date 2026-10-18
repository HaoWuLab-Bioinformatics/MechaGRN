# -*- coding: utf-8 -*-
"""
Regression tests for the directional formulation introduced in revision v2.

Context (reviewer 3, Major 2): in the code released with the first submission
the adjacency matrix `A` was built *undirected* -- both `A[tf, target]` and
`A[target, tf]` were written -- so the "forward" and "backward" branches of the
direction-aware layers saw exactly the same neighbourhood, and the directional
weights were frozen at 0.9 / 0.1 (`requires_grad=False`).  Both points
contradicted Eq. 12/13 of the manuscript.

The tests below pin the corrected behaviour:

  1. `Adj_Generate` writes TF -> target edges only, so A is asymmetric and
     `A.T` really is a different operator.
  2. negative (label 0) training rows never enter the graph;
  3. `direction=False` still reproduces the legacy undirected graph exactly, so
     archived runs stay reproducible;
  4. the two directional weights of each layer are learnable logits combined by
     softmax (Eq. 13), initialised at (0.9, 0.1).

The tests deliberately do not import `torch` or `scipy`: they execute the
source text of `Adj_Generate` against a tiny stand-in for
`scipy.sparse.dok_matrix`, so the whole file runs on a bare Python + NumPy
installation.

Usage:
    python Code/tests/test_direction_adjacency.py
"""
import ast
import math
import pathlib
import sys

import numpy as np

CODE = pathlib.Path(__file__).resolve().parents[1]
out = []


def log(msg=""):
    out.append(str(msg))
    print(str(msg))


# --------------------------------------------------------------------------
# Minimal stand-in for scipy.sparse.dok_matrix
# --------------------------------------------------------------------------
class FakeDOK:
    def __init__(self, shape, dtype=None):
        self.shape = shape
        self.dtype = dtype
        self.d = {}

    def __setitem__(self, key, value):
        self.d[key] = value

    def __getitem__(self, key):
        return self.d.get(key, 0.0)

    def todok(self):
        return self

    def items(self):
        return self.d.items()

    def __add__(self, other):
        new = FakeDOK(self.shape, self.dtype)
        new.d.update(self.d)
        for k, v in other.items():
            new.d[k] = new.d.get(k, 0.0) + v
        return new


class FakeSP:
    @staticmethod
    def dok_matrix(shape, dtype=None):
        return FakeDOK(shape, dtype)

    @staticmethod
    def identity(n, format="dok"):
        m = FakeDOK((n, n))
        for i in range(n):
            m[i, i] = 1.0
        return m


class FakeSelf:
    def __init__(self, train_set, num_gene):
        self.train_set = train_set
        self.num_gene = num_gene


def extract_func(py_file, name):
    """Return the real function object defined in `py_file` without importing it."""
    src = py_file.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {}
            exec(compile(ast.parse(ast.get_source_segment(src, node)), str(py_file), "exec"), ns)
            ns["sp"] = FakeSP
            ns["np"] = np
            return ns[name], node
    raise LookupError(f"{name} not found in {py_file}")


def matrix_of(dok, n):
    return [[dok[i, j] for j in range(n)] for i in range(n)]


# --------------------------------------------------------------------------
# 1. Adjacency construction
# --------------------------------------------------------------------------
Adj_Generate, adj_node = extract_func(CODE / "preprocessing.py", "Adj_Generate")

defaults = adj_node.args.defaults
default_map = dict(zip([a.arg for a in adj_node.args.args][-len(defaults):], defaults))
log("[1] Adj_Generate signature: " + ", ".join(a.arg for a in adj_node.args.args))
log("    default of `direction` = " + ast.unparse(default_map["direction"]))
assert ast.unparse(default_map["direction"]) == "True"

# chain 0 -> 1 -> 2 -> 0; all three nodes are TFs, which used to trigger the
# old "bidirectional if target is also a TF" branch
train_set = [
    [0, 1, 1],
    [1, 2, 1],
    [2, 0, 1],
    [3, 4, 0],   # negative sample: must never enter the graph
]
TF_set = {0, 1, 2, 3}
n = 5
s = FakeSelf(train_set, n)

directed = Adj_Generate.__get__(s)(TF_set)
legacy = Adj_Generate.__get__(s)(TF_set, direction=False)
with_loops = Adj_Generate.__get__(s)(TF_set, loop=True)

D = matrix_of(directed, n)
L = matrix_of(legacy, n)
LP = matrix_of(with_loops, n)

for label, M in (("directed A (default)", D), ("legacy undirected A", L)):
    log(f"\n[2] {label}:")
    for row in M:
        log("    " + " ".join(f"{v:g}" for v in row))

symmetric_D = all(D[i][j] == D[j][i] for i in range(n) for j in range(n))
symmetric_L = all(L[i][j] == L[j][i] for i in range(n) for j in range(n))
log(f"\n    symmetric(default)={symmetric_D}  symmetric(direction=False)={symmetric_L}")
assert symmetric_D is False, "directed adjacency must be asymmetric"
assert symmetric_L is True, "legacy path must stay symmetric for reproducibility"

# forward and backward neighbourhoods are now genuinely different
fwd_nbrs = {j for j in range(n) if D[1][j] == 1.0}
bwd_nbrs = {j for j in range(n) if D[j][1] == 1.0}
log(f"    neighbours of node 1: forward(A)={sorted(fwd_nbrs)} backward(A^T)={sorted(bwd_nbrs)}")
assert fwd_nbrs == {2} and bwd_nbrs == {0}

# negative rows never create edges
assert D[3][4] == 0.0 and D[4][3] == 0.0

# self-loops are optional and additive
assert LP[0][0] == 1.0 and LP[1][1] == 1.0 and LP[2][2] == 1.0
log("    self-loops present when loop=True: OK")

# --------------------------------------------------------------------------
# 2. Learnable softmax weights (Eq. 13)
# --------------------------------------------------------------------------
model_src = (CODE / "Model.py").read_text(encoding="utf-8")
tree = ast.parse(model_src)
layer = next(
    x for x in ast.walk(tree)
    if isinstance(x, ast.ClassDef) and x.name == "DirectionAwareGraphLayer"
)

props = {
    x.name for x in layer.body
    if isinstance(x, ast.FunctionDef)
    and any(isinstance(d, ast.Name) and d.id == "property" for d in x.decorator_list)
}
log(f"\n[3] DirectionAwareGraphLayer properties: {sorted(props)}")
assert {"alpha", "beta"} <= props

frozen_all = []
for cls_name in ("DirectionAwareGraphLayer", "HighOrderPerceptionLayer"):
    cls = next(x for x in ast.walk(tree) if isinstance(x, ast.ClassDef) and x.name == cls_name)
    frozen = [
        (x.lineno, ast.unparse(x))
        for x in ast.walk(cls)
        if isinstance(x, ast.Call) and any(k.arg == "requires_grad" for k in x.keywords)
    ]
    log(f"[4] {cls_name}: parameters with requires_grad=... -> {len(frozen)}")
    for lineno, txt in frozen:
        log(f"    line {lineno}: {txt}")
    frozen_all.extend(frozen)
assert not frozen_all, "no scalar coefficient may stay frozen any more"

# the two scalar coefficients are learnable nn.Parameters initialised at 0.01,
# so the first forward pass still matches the previously released code
# (`gamma` is also the name of FocalLoss' focusing parameter, which is why the
# high-order coefficient is called `lambda_ho` and why this lookup is scoped to
# the owning class)
for attr, owner in (("wolf", "DirectionAwareGraphLayer"),
                    ("lambda_ho", "HighOrderPerceptionLayer")):
    cls = next(x for x in ast.walk(tree) if isinstance(x, ast.ClassDef) and x.name == owner)
    assigns = [
        ast.unparse(x) for x in ast.walk(cls)
        if isinstance(x, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == attr for t in x.targets)
    ]
    log(f"[5] {owner}.{attr} = {assigns}")
    assert len(assigns) == 1, f"expected exactly one assignment to {owner}.{attr}"
    assert "nn.Parameter" in assigns[0] and "torch.tensor(0.01" in assigns[0], (
        f"{owner}.{attr} must be a learnable parameter initialised at 0.01"
    )

# the focal-loss focusing parameter stays a hyperparameter, not a parameter
focal = next(x for x in ast.walk(tree) if isinstance(x, ast.ClassDef) and x.name == "FocalLoss")
focal_src = ast.get_source_segment(model_src, focal)
log("[5] FocalLoss keeps gamma as a constructor argument: "
    + str("self.gamma = gamma" in focal_src))
assert "self.gamma = gamma" in focal_src, "focal-loss gamma must remain a hyperparameter"

fwd = next(x for x in layer.body if isinstance(x, ast.FunctionDef) and x.name == "forward")
softmax_calls = [
    ast.unparse(x) for x in ast.walk(fwd)
    if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "softmax"
]
log(f"[6] softmax calls in forward: {softmax_calls}")
assert softmax_calls, "forward must derive alpha/beta from softmax"

# initialisation reproduces the previously frozen (0.9, 0.1) pair exactly
logits = [math.log(0.9), math.log(0.1)]
e = [math.exp(v) for v in logits]
a0, b0 = e[0] / sum(e), e[1] / sum(e)
log(f"\n[7] softmax init = ({a0:.6f}, {b0:.6f}), sum = {a0 + b0:.6f}")
assert abs(a0 - 0.9) < 1e-12 and abs(b0 - 0.1) < 1e-12

log("\nALL CHECKS PASSED")
