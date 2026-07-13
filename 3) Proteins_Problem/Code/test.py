def task_c(data_a, data_b):
    from scipy.sparse import coo_matrix

    n_ids = np.max(data_a) + 1
    data = np.ones(len(data_a))
    sparse_mat = coo_matrix((data, (data_a, data_b)), shape = (n_ids, 625))
    task_imgs = sparse_mat.toarray()

    current_model = np.mean(task_imgs, axis = 0).reshape(25, 25)

    for _ in range(10):
        refs = np.array([np.rot90(current_model, k = i).flatten() for i in range(4)])
        scores = task_imgs @ refs.T
        best_orientation = np.argmax(scores, axis = 1)

        new_model = np.zeros((25, 25))
        for i in range(4):
            mask = (best_orientation == i)
            if np.sum(mask) > 0:
                group_sum = np.sum(task_imgs[mask], axis = 0).reshape(25, 25)
                new_model += np.rot90(group_sum, k = -i)

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