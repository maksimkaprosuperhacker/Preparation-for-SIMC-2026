import numpy as np
import matplotlib.pyplot as plt

#First(A)
def task_a(data):
    avg_val = np.mean(np.sum(data, axis=1))
    print(f"Average: {avg_val}")

#Second(B)
def task_b(data):
    mean_pattern_flat = np.mean(data, axis=0)
    mean_pattern_img = mean_pattern_flat.reshape(50, 50)
    plt.imshow(mean_pattern_img, cmap = "Blues")
    plt.title("Average Unoriented Pattern")
    plt.colorbar()
    plt.show()

# Third(C)
def task_c(data):
    task_imgs = data.reshape(-1, 50, 50)
    ref_img=task_imgs[0]
    refs = [np.rot90(ref_img, k=i) for i in range(4)]
    groups=[[],[],[],[]]


    for img in task_imgs:
        diffs = [np.sum((img - t)**2) for t in refs]
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


loaded = np.load('endeavour.npz')
task = loaded["task4"]
#task_a(task)
#task_b(task)
task_c(task)