#@title Plot
def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    # Plot the loss & accuracy curves
    plt.figure(figsize=(10, 4))

    # Plot the training loss over iterations
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')