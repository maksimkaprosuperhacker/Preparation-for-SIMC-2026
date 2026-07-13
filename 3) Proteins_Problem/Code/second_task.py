import numpy as np
import matplotlib.pyplot as plt


def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


loaded = np.load('endeavour.npz')
task2 = loaded["task2"]
ref = task2[0].reshape(36,36)

#First(A)

plt.imshow(ref, cmap = 'Greys', interpolation = 'nearest')
plt.colorbar()
plt.show()
"""

rot0 = ref
rot90 = np.rot90(ref, k=-1)
rot180 = np.rot90(ref, k=-2)
rot270 = np.rot90(ref, k=-3)

refs = [
    rot0.flatten(), rot90.flatten(),
    rot180.flatten(),
    rot270.flatten()
]


labels = []


for pattern in task2:
    sims = [similarity(pattern, r) for r in refs]
    labels.append(np.argmax(sims))


labels = np.array(labels)
unique, counts = np.unique(labels, return_counts=True)

#Second
for u, c in zip(unique, counts):
    print(f"Rotation {u * 90} degrees: {c} patterns")

"""