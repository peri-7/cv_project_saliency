import scipy.io as sio
import numpy as np

path = r"mini_data\fixations\train\COCO_train2014_000000000009.mat"
d = sio.loadmat(path)

print("=== top-level keys ===")
print([k for k in d.keys() if not k.startswith('__')])

g = d['gaze']
print("\n=== gaze array ===")
print("shape:", g.shape)
print("dtype:", g.dtype)
print("field names:", g.dtype.names)
print("num subjects:", g.shape[1])

print("\n=== subject 0 ===")
s0 = g[0, 0]
for name in g.dtype.names:
    field = s0[name]
    print(f"  field '{name}': type={type(field).__name__}, shape={getattr(field, 'shape', None)}, dtype={getattr(field, 'dtype', None)}")

fix = s0['fixations']
print("\n=== subject 0 fixations ===")
print("shape:", fix.shape, "dtype:", fix.dtype)
print("first 5 rows:")
print(fix[:5])
print("min:", fix.min(axis=0) if fix.size else None, "max:", fix.max(axis=0) if fix.size else None)

print("\n=== all subjects: fixation counts ===")
for i in range(g.shape[1]):
    f = g[0, i]['fixations']
    if f.dtype == object and f.size == 1:
        f = f[0, 0]
    print(f"  subject {i}: shape={f.shape}, dtype={f.dtype}")
