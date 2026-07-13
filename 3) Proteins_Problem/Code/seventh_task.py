import numpy as np
import matplotlib.pyplot as plt


def transform(task6_a, task6_b, n_features=625):
    n_ids = max(task6_a) + 1
    result = np.zeros((n_ids, n_features), dtype=int)


    for i in range(len(task6_a)):
        result[task6_a[i], task6_b[i]] += 1

    return result

def task_a(data_a, data_b):

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

        #average
        current_model = new_model / n_ids

    plt.imshow(np.rot90(current_model, k=-1), cmap = "gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


loaded = np.load('endeavour.npz')
task1 = loaded["task7a"]
task2 = loaded["task7b"]

task_a(task1,task2)
