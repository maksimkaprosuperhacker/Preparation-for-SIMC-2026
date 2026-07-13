import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def transform(task6_a, task6_b, n_features=625):
    n_ids = max(task6_a) + 1
    result = np.zeros((n_ids, n_features), dtype=int)


    for i in range(len(task6_a)):
        result[task6_a[i], task6_b[i]] += 1

    return result

#First(A)
def task_a(data_a,data_b):
    avg_val = len(data_b)/65535
    print(f"Average: {avg_val}")

#Second(B)
def task_b(data_a,data_b):
    pattern = np.zeros(625)


    for i in range(625):
        pattern[i] = np.sum(data_b == i)
        pattern[i]/=len(data_b)
    img0 = pattern.reshape(25, 25)
    plt.imshow(img0, cmap = "Blues")
    plt.title("Average Unoriented Pattern")
    plt.colorbar()
    plt.show()
#Third(C)
"""
OLD VERSION

def task_c(data_a,data_b):
task_imgs = transform(data_a,data_b)
task_imgs = task_imgs.reshape(-1, 25, 25)

code
Code
ref_img=task_imgs[0]
refs = [np.rot90(ref_img, k=-i) for i in range(4)]
groups=[[],[],[],[]]


for img in task_imgs:
    diffs = [np.sum((img - t) ** 2) for t in refs]
    best_orientation = np.argmin(diffs)
    groups[best_orientation].append(img)


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

for i in range(4):
    if groups[i]:
        class_avg = np.mean(groups[i], axis=0)
        axs[i].imshow(class_avg, cmap="gray")
        axs[i].set_title(f"Orientation {i*90}°\n(n={len(groups[i])})")
    axs[i].axis('off')

plt.tight_layout()
plt.show()
"""

def task_c(data_a, data_b):

    n_ids = np.max(data_a) + 1
    task_imgs = transform(data_a, data_b)



    #initial image
    current_model = np.mean(task_imgs, axis = 0).reshape(25, 25)





    for q in range(20):
        refs = np.array([np.rot90(current_model, k = i).flatten() for i in range(4)])

        #Find the maximum similarity between all and 4 references
        scores = task_imgs @ refs.T
        best_orientation = np.argmax(scores, axis = 1)



        new_model = np.zeros((25, 25))


        #rotate everything to 0 degrees
        for i in range(4):
            mask = (best_orientation == i)
            if np.sum(mask) > 0:
                group_sum = np.sum(task_imgs[mask], axis = 0).reshape(25, 25)
                new_model += np.rot90(group_sum, k = -i)

        # average
        current_model = new_model / n_ids

    fig, axs = plt.subplots(2, 2, figsize = (10, 10))
    axs = axs.flatten()

    for i in range(4):
        class_avg = np.rot90(current_model, k = i)
        axs[i].imshow(class_avg, cmap = "gray")
        axs[i].set_title(f"Orientation {i * 90}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


loaded = np.load('endeavour.npz')
task1 = loaded["task7a"]
task2 = loaded["task7b"]

#task_a(task1,task2)
#task_b(task1,task2)
task_c(task1,task2)