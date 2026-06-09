"""
Executa todos os experimentos e salva os plots em results/.
Equivalente ao notebook, mas roda direto: python run_experiments.py
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # sem display
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from mlp import MLP, load_mnist
from mlp.optimizers import SGD, Adam

os.makedirs("results", exist_ok=True)

# ---------- dados ----------
print("=== Carregando MNIST ===")
(X_train_full, y_train_full), (X_test, y_test) = load_mnist("data")

X_val   = X_train_full[-5000:]
y_val   = y_train_full[-5000:]
X_train = X_train_full[:-5000]
y_train = y_train_full[:-5000]
print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}\n")

# ---------- gradient check ----------
print("=== Gradient Check ===")
net_check = MLP([784, 16, 10], hidden_activation="relu", seed=0)
diff = net_check.gradient_check(X_train[:10], y_train[:10])
print(f"Max relative difference: {diff:.2e}")
assert diff < 1e-4, "Gradient check FAILED"
print("Gradient check PASSED\n")

# ---------- experimentos ----------
EPOCHS = 30
BATCH  = 128

configs = [
    ("A — SGD 256-128",      [784, 256, 128, 10],      SGD(learning_rate=0.1),          "tab:blue"),
    ("B — SGD 512-256-128",  [784, 512, 256, 128, 10], SGD(learning_rate=0.1),          "tab:orange"),
    ("C — Adam 256-128",     [784, 256, 128, 10],      Adam(learning_rate=0.001),       "tab:green"),
    ("D — SGD+mom 256-128",  [784, 256, 128, 10],      SGD(learning_rate=0.05, momentum=0.9), "tab:red"),
]

histories = []
test_accs = []

for label, layers, opt, color in configs:
    print(f"=== {label} ===")
    net = MLP(layers, hidden_activation="relu", optimizer=opt, seed=42)
    hist = net.train(X_train, y_train, X_val=X_val, y_val=y_val,
                     epochs=EPOCHS, batch_size=BATCH)
    acc = net.accuracy(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}\n")
    histories.append((label, hist, color, net))
    test_accs.append(acc)

# ---------- curvas ----------
print("=== Salvando plots ===")
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, hist, color, _ in histories:
    axes[0].plot(epochs_range, hist["val_loss"], label=label, color=color)
    axes[1].plot(epochs_range, hist["val_acc"],  label=label, color=color)

axes[0].set_title("Validation Loss por Época")
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].set_title("Validation Accuracy por Época")
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Acurácia")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/curves_comparison.png", bbox_inches="tight")
plt.close()
print("Salvo: results/curves_comparison.png")

# ---------- tabela ----------
print("\n=== Tabela Comparativa ===")
header = f"{'Exp':<6} {'Camadas':<18} {'Otimizador':<24} {'Test Acc'}"
print(header)
print("-" * 65)
rows = [
    ("A", "256-128",      "SGD lr=0.1",           test_accs[0]),
    ("B", "512-256-128",  "SGD lr=0.1",           test_accs[1]),
    ("C", "256-128",      "Adam lr=0.001",         test_accs[2]),
    ("D", "256-128",      "SGD mom=0.9 lr=0.05",  test_accs[3]),
]
for exp, layers, opt_name, acc in rows:
    print(f"{exp:<6} {layers:<18} {opt_name:<24} {acc:.4f}")

# ---------- matriz de confusão ----------
best_idx = int(np.argmax(test_accs))
best_net = histories[best_idx][3]
y_pred = best_net.predict(X_test)

n_classes = 10
cm = np.zeros((n_classes, n_classes), dtype=int)
for true, pred in zip(y_test, y_pred):
    cm[true, pred] += 1

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xlabel("Predito"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de Confusão — {histories[best_idx][0]}")
for i in range(n_classes):
    for j in range(n_classes):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", bbox_inches="tight")
plt.close()
print("\nSalvo: results/confusion_matrix.png")

# ---------- pesos W1 ----------
W1 = best_net.params["W1"]
n_show = min(64, W1.shape[1])
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for idx, ax in enumerate(axes.flat):
    if idx < n_show:
        ax.imshow(W1[:, idx].reshape(28, 28), cmap="RdBu", vmin=-0.3, vmax=0.3)
    ax.axis("off")
plt.suptitle("Pesos aprendidos — camada oculta 1 (64 neurônios)", y=1.01)
plt.tight_layout()
plt.savefig("results/weights_visualization.png", bbox_inches="tight")
plt.close()
print("Salvo: results/weights_visualization.png")

# ---------- t-SNE ----------
try:
    from sklearn.manifold import TSNE
    print("\nRodando t-SNE (pode demorar ~30s)...")
    X_sub = X_test[:2000]
    y_sub = y_test[:2000]
    A = X_sub
    for i in range(best_net.n_layers - 1):
        import numpy as np
        Z = A @ best_net.params[f"W{i+1}"] + best_net.params[f"b{i+1}"]
        A = best_net.act_fn(Z)
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
    emb = tsne.fit_transform(A)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_sub, cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Dígito")
    ax.set_title("t-SNE — Embeddings da penúltima camada")
    plt.tight_layout()
    plt.savefig("results/tsne_embeddings.png", bbox_inches="tight")
    plt.close()
    print("Salvo: results/tsne_embeddings.png")
except Exception as e:
    print(f"t-SNE pulado: {e}")

print("\nConcluído. Veja os plots em results/")
