import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Transformer Training Loss")
    plt.grid(True)
    plt.show()
