import numpy as np
import matplotlib.pyplot as plt

loaded = np.load('endeavour.npz')

task1 = loaded["task1"]
task2 = loaded["task2"]
task3 = loaded["task3"]
task4 = loaded["task4"]

for k in loaded.keys():
    print(f"key {k} has shape: {loaded[k].shape}")

img0 = task3[0].reshape(33,33)


ref = img0

rot0 = ref
rot90 = np.rot90(ref, k=-1)
rot180 = np.rot90(ref, k=-2)
rot270 = np.rot90(ref, k=-3)

refs = [
    rot0.flatten(),
    rot90.flatten(),
    rot180.flatten(),
    rot270.flatten()
]


def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

labels = []

for pattern in task3:
    sims = [similarity(pattern, r) for r in refs]
    labels.append(np.argmax(sims))

labels = np.array(labels)

unique, counts = np.unique(labels, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Rotation {u * 90} degrees: {c} patterns")