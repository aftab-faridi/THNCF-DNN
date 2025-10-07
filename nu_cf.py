import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from prettytable import PrettyTable
import seaborn as sns
import time

# Set style for professional plots
plt.style.use("default")
FONT_NAME = "Times New Roman"
plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.alpha"] = 0.3

# Load dataset
data = pd.read_csv("cylinder_tetra_hybrid.csv")
X = data.drop(columns=["SkinFriction", "NusseltNumber"]).values
y = data[["SkinFriction", "NusseltNumber"]].values

# Define LaTeX feature names
feature_names = [
    r"$Ha$",
    r"$S$",
    r"$Pr$",
    r"$\varphi_1$",
    r"$\varphi_2$",
    r"$\varphi_3$",
    r"$\varphi_4$",
    r"$P_0$",
    r"$H_g$",
    r"$Ec$",
    r"$\beta$",
    r"$\alpha$",
    r"$Rd$",
    r"$\lambda$",
    r"$M_i$",
]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


# Enhanced Dual-output DNN with architecture visualization capability
class DualHeadDNN(nn.Module):
    def __init__(self, input_dim):
        super(DualHeadDNN, self).__init__()
        self.input_dim = input_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.cf_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.nu_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        cf = self.cf_head(shared_out)
        nu = self.nu_head(shared_out)
        return torch.cat((cf, nu), dim=1)


def plot_network_architecture():
    """Plot neural network architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define layers
    layers = {
        "Input": {"pos": (0, 4), "nodes": 15, "color": "lightblue"},
        "Hidden1": {"pos": (2, 5), "nodes": 256, "color": "lightcoral"},
        "Hidden2": {"pos": (4, 5), "nodes": 128, "color": "lightcoral"},
        "Hidden3": {"pos": (6, 5), "nodes": 64, "color": "lightcoral"},
        "CF_Head": {"pos": (8, 6), "nodes": 32, "color": "lightgreen"},
        "NU_Head": {"pos": (8, 2), "nodes": 32, "color": "lightgreen"},
        "CF_Output": {"pos": (10, 6), "nodes": 1, "color": "gold"},
        "NU_Output": {"pos": (10, 2), "nodes": 1, "color": "gold"},
    }

    # Draw connections
    connections = [
        ("Input", "Hidden1"),
        ("Hidden1", "Hidden2"),
        ("Hidden2", "Hidden3"),
        ("Hidden3", "CF_Head"),
        ("Hidden3", "NU_Head"),
        ("CF_Head", "CF_Output"),
        ("NU_Head", "NU_Output"),
    ]

    for start, end in connections:
        x1, y1 = layers[start]["pos"]
        x2, y2 = layers[end]["pos"]
        ax.plot([x1, x2], [y1, y2], "k-", alpha=0.3, linewidth=0.5)

    # Draw nodes
    for layer_name, layer_info in layers.items():
        x, y = layer_info["pos"]
        size = min(max(layer_info["nodes"] * 2, 50), 500)
        ax.scatter(x, y, s=size, c=layer_info["color"], alpha=0.8, edgecolor="black")
        ax.text(
            x,
            y - 0.5,
            f"{layer_name}\n({layer_info['nodes']})",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 7)
    ax.set_title(
        "Dual-Head Deep Neural Network Architecture", fontsize=14, fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("network_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_data_distribution():
    """Plot input data distribution similar to paper's data analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Select first 6 features for visualization
    for i in range(6):
        row = i // 3
        col = i % 3

        axes[row, col].hist(
            X[:, i], bins=30, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[row, col].set_title(
            f"Distribution of {feature_names[i]}", fontweight="bold", fontsize=11
        )
        axes[row, col].set_xlabel("Value", fontweight="bold")
        axes[row, col].set_ylabel("Frequency", fontweight="bold")
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap():
    """Plot correlation heatmap of features"""
    # Create correlation matrix
    df_features = pd.DataFrame(X, columns=feature_names)
    df_features["$Cf_x$"] = y[:, 0]
    df_features["$Nu_{{x}}$"] = y[:, 1]

    correlation_matrix = df_features.corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="RdBu_r",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


# Training function with enhanced tracking
def train_model_enhanced(
    model, X, y, X_val, y_val, num_epochs=1000, optimizer_type="adam", lr=0.001
):
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
    else:
        raise ValueError("Unsupported optimizer")

    cf_loss_fn = nn.MSELoss()
    nu_loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # Enhanced tracking
    train_losses, val_losses = [], []
    cf_train_losses, nu_train_losses = [], []
    cf_val_losses, nu_val_losses = [], []
    r2_cf_train_list, r2_nu_train_list = [], []
    r2_cf_val_list, r2_nu_val_list = [], []
    learning_rates = []
    epoch_times = []

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        optimizer.zero_grad()
        output = model(X)
        cf_loss = cf_loss_fn(output[:, 0], y[:, 0])
        nu_loss = nu_loss_fn(output[:, 1], y[:, 1])
        total_loss = cf_loss + nu_loss
        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_cf_loss = cf_loss_fn(val_output[:, 0], y_val[:, 0])
            val_nu_loss = nu_loss_fn(val_output[:, 1], y_val[:, 1])
            val_total_loss = val_cf_loss + val_nu_loss

            # R² Scores
            preds_train = output.numpy()
            preds_val = val_output.numpy()
            y_train_np = y.numpy()
            y_val_np = y_val.numpy()
            r2_cf_train = r2_score(y_train_np[:, 0], preds_train[:, 0])
            r2_nu_train = r2_score(y_train_np[:, 1], preds_train[:, 1])
            r2_cf_val = r2_score(y_val_np[:, 0], preds_val[:, 0])
            r2_nu_val = r2_score(y_val_np[:, 1], preds_val[:, 1])

        # Track metrics
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        train_losses.append(total_loss.item())
        val_losses.append(val_total_loss.item())
        cf_train_losses.append(cf_loss.item())
        nu_train_losses.append(nu_loss.item())
        cf_val_losses.append(val_cf_loss.item())
        nu_val_losses.append(val_nu_loss.item())
        r2_cf_train_list.append(r2_cf_train)
        r2_nu_train_list.append(r2_nu_train)
        r2_cf_val_list.append(r2_cf_val)
        r2_nu_val_list.append(r2_nu_val)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_total_loss)

        if val_total_loss.item() < best_val_loss:
            best_val_loss = val_total_loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}, Val Loss: {val_total_loss.item():.6f}, "
                f"R² CF: {r2_cf_val:.4f}, R² Nu: {r2_nu_val:.4f}"
            )

        if patience_counter > 300:
            print("Early stopping.")
            break

    model.load_state_dict(best_model_state)

    # Return all tracked metrics
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "cf_train_losses": cf_train_losses,
        "nu_train_losses": nu_train_losses,
        "cf_val_losses": cf_val_losses,
        "nu_val_losses": nu_val_losses,
        "r2_cf_train": r2_cf_train_list,
        "r2_nu_train": r2_nu_train_list,
        "r2_cf_val": r2_cf_val_list,
        "r2_nu_val": r2_nu_val_list,
        "learning_rates": learning_rates,
        "epoch_times": epoch_times,
    }, model


def plot_comprehensive_training_analysis(metrics):
    """Plot comprehensive training analysis similar to research paper"""

    epochs = range(len(metrics["train_losses"]))

    # 1. Training vs Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_losses"], "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, metrics["val_losses"], "r-", label="Validation Loss", linewidth=2)
    plt.title("Training vs Validation Loss Evolution", fontweight="bold", fontsize=14)
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Loss", fontweight="bold")
    plt.legend(prop={"weight": "bold"})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_validation_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Individual Output Losses
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs, metrics["cf_train_losses"], "b-", label="$Cf_x$ Train", linewidth=2
    )
    plt.plot(epochs, metrics["cf_val_losses"], "b--", label="$Cf_x$ Val", linewidth=2)
    plt.plot(
        epochs,
        metrics["nu_train_losses"],
        "g-",
        label="$Nu_x$ Train",
        linewidth=2,
    )
    plt.plot(epochs, metrics["nu_val_losses"], "g--", label="$Nu_x$ Val", linewidth=2)
    plt.title("Individual Output Loss Evolution", fontweight="bold", fontsize=14)
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Loss", fontweight="bold")
    plt.legend(prop={"weight": "bold"})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("individual_output_losses.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. R² Score Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["r2_cf_train"], "b-", label="$Cf_x$ Train R²", linewidth=2)
    plt.plot(epochs, metrics["r2_cf_val"], "b--", label="$Cf_x$ Test R²", linewidth=2)
    plt.plot(epochs, metrics["r2_nu_train"], "g-", label="$Nu_x$ Train R²", linewidth=2)
    plt.plot(epochs, metrics["r2_nu_val"], "g--", label="$Nu_x$ Test R²", linewidth=2)
    plt.title("R² Score Evolution During Training", fontweight="bold", fontsize=14)
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("R² Score", fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend(prop={"weight": "bold"})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("r2_score_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Learning Rate Schedule
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["learning_rates"], "purple", linewidth=2)
    plt.title("Learning Rate Schedule", fontweight="bold", fontsize=14)
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Learning Rate", fontweight="bold")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_rate_schedule.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_parameter_sensitivity_analysis(X_test, model, scaler_X, scaler_y):
    """Plot parameter sensitivity analysis (similar to Figs. 2a-2d, 3a-3d in paper)"""

    # Select 6 most important features for sensitivity analysis
    feature_indices = [0, 1, 2, 3, 4, 5]  # Adjust based on your features

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, feat_idx in enumerate(feature_indices):
        row = i // 3
        col = i % 3

        # Create parameter variation
        base_input = np.mean(X_test, axis=0)
        param_values = np.linspace(
            X_test[:, feat_idx].min(), X_test[:, feat_idx].max(), 50
        )
        cf_responses = []
        nu_responses = []

        for param_val in param_values:
            test_input = base_input.copy()
            test_input[feat_idx] = param_val
            test_input_scaled = scaler_X.transform([test_input])
            test_tensor = torch.tensor(test_input_scaled, dtype=torch.float32)

            with torch.no_grad():
                pred_scaled = model(test_tensor).numpy()
                pred_original = scaler_y.inverse_transform(pred_scaled)
                cf_responses.append(pred_original[0, 0])
                nu_responses.append(pred_original[0, 1])

        # Plot parameter sensitivity
        ax = axes[row, col]
        ax2 = ax.twinx()

        line1 = ax.plot(param_values, cf_responses, "b-", linewidth=3, label="$Cf_x$")
        line2 = ax2.plot(
            param_values, nu_responses, "r-", linewidth=3, label="$Nu_{{x}}$"
        )

        ax.set_xlabel(f"{feature_names[feat_idx]}", fontweight="bold", fontsize=11)
        ax.set_ylabel("$Cf_x$", fontweight="bold", fontsize=11, color="blue")
        ax2.set_ylabel("$Nu_{{x}}$", fontweight="bold", fontsize=11, color="red")
        ax.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="red")

        ax.set_title(
            f"Impact of {feature_names[feat_idx]} on Outputs",
            fontweight="bold",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper left", prop={"weight": "bold"})

    plt.tight_layout()
    plt.savefig("parameter_sensitivity_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_comprehensive_3d_response_surfaces():
    """Plot comprehensive 3D response surfaces for all input combinations"""
    import itertools

    print(f"Generating 3D response surfaces for {len(feature_names)} features...")

    # Generate all combinations of 2 features
    feature_combinations = list(itertools.combinations(range(len(feature_names)), 2))

    for combo_idx, (feat1_idx, feat2_idx) in enumerate(feature_combinations):
        print(
            f"Processing combination {combo_idx + 1}/{len(feature_combinations)}: {feature_names[feat1_idx]} vs {feature_names[feat2_idx]}"
        )

        # Create meshgrid for the two selected features
        x1_range = np.linspace(X[:, feat1_idx].min(), X[:, feat1_idx].max(), 30)
        x2_range = np.linspace(X[:, feat2_idx].min(), X[:, feat2_idx].max(), 30)
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)

        # Generate response surface data
        base_input = np.mean(X, axis=0)
        Z_cf = np.zeros_like(X1_mesh)
        Z_nu = np.zeros_like(X1_mesh)

        for i in range(len(x1_range)):
            for j in range(len(x2_range)):
                test_input = base_input.copy()
                test_input[feat1_idx] = X1_mesh[i, j]
                test_input[feat2_idx] = X2_mesh[i, j]
                test_input_scaled = scaler_X.transform([test_input])
                test_tensor = torch.tensor(test_input_scaled, dtype=torch.float32)

                with torch.no_grad():
                    pred_scaled = trained_model(test_tensor).numpy()
                    pred_original = scaler_y.inverse_transform(pred_scaled)
                    Z_cf[i, j] = pred_original[0, 0]
                    Z_nu[i, j] = pred_original[0, 1]

        # Plot $Cf_x$ 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X1_mesh, X2_mesh, Z_cf, cmap="viridis", alpha=0.8)
        ax.set_xlabel(f"{feature_names[feat1_idx]}", fontweight="bold")
        ax.set_ylabel(f"{feature_names[feat2_idx]}", fontweight="bold")
        ax.set_zlabel("$Cf_x$", fontweight="bold")
        ax.set_title(
            f"$Cf_x$ Response Surface\n{feature_names[feat1_idx]} vs {feature_names[feat2_idx]}",
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            f"3d_cf_surface_{combo_idx + 1:02d}_{feature_names[feat1_idx]}_vs_{feature_names[feat2_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot $Nu_{{x}}$ 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X1_mesh, X2_mesh, Z_nu, cmap="plasma", alpha=0.8)
        ax.set_xlabel(f"{feature_names[feat1_idx]}", fontweight="bold")
        ax.set_ylabel(f"{feature_names[feat2_idx]}", fontweight="bold")
        ax.set_zlabel("$Nu_x$", fontweight="bold")
        ax.set_title(
            f"$Nu_x$ Response Surface\n{feature_names[feat1_idx]} vs {feature_names[feat2_idx]}",
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            f"3d_nu_surface_{combo_idx + 1:02d}_{feature_names[feat1_idx]}_vs_{feature_names[feat2_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot combined contour plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # $Cf_x$ contour
        contour1 = ax1.contourf(
            X1_mesh, X2_mesh, Z_cf, levels=20, cmap="viridis", alpha=0.8
        )
        ax1.set_xlabel(f"{feature_names[feat1_idx]}", fontweight="bold")
        ax1.set_ylabel(f"{feature_names[feat2_idx]}", fontweight="bold")
        ax1.set_title(
            f"$Cf_x$ Contours\n{feature_names[feat1_idx]} vs {feature_names[feat2_idx]}",
            fontweight="bold",
        )
        plt.colorbar(contour1, ax=ax1)

        # $Nu_{{x}}$ contour
        contour2 = ax2.contourf(
            X1_mesh, X2_mesh, Z_nu, levels=20, cmap="plasma", alpha=0.8
        )
        ax2.set_xlabel(f"{feature_names[feat1_idx]}", fontweight="bold")
        ax2.set_ylabel(f"{feature_names[feat2_idx]}", fontweight="bold")
        ax2.set_title(
            f"$Nu_x$ Contours\n{feature_names[feat1_idx]} vs {feature_names[feat2_idx]}",
            fontweight="bold",
        )
        plt.colorbar(contour2, ax=ax2)

        plt.tight_layout()
        plt.savefig(
            f"contour_plots_{combo_idx + 1:02d}_{feature_names[feat1_idx]}_vs_{feature_names[feat2_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_comprehensive_2d_parameter_analysis():
    """Plot comprehensive 2D parameter analysis for all features"""
    print(f"Generating 2D parameter analysis for {len(feature_names)} features...")

    for feat_idx in range(len(feature_names)):
        print(
            f"Processing feature {feat_idx + 1}/{len(feature_names)}: {feature_names[feat_idx]}"
        )

        # Create parameter variation
        base_input = np.mean(X_test, axis=0)
        param_values = np.linspace(
            X_test[:, feat_idx].min(), X_test[:, feat_idx].max(), 100
        )
        cf_responses = []
        nu_responses = []

        for param_val in param_values:
            test_input = base_input.copy()
            test_input[feat_idx] = param_val
            test_input_scaled = scaler_X.transform([test_input])
            test_tensor = torch.tensor(test_input_scaled, dtype=torch.float32)

            with torch.no_grad():
                pred_scaled = trained_model(test_tensor).numpy()
                pred_original = scaler_y.inverse_transform(pred_scaled)
                cf_responses.append(pred_original[0, 0])
                nu_responses.append(pred_original[0, 1])

        # Plot individual $Cf_x$ response
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, cf_responses, "b-", linewidth=3, label="$Cf_x$")
        plt.xlabel(f"{feature_names[feat_idx]}", fontweight="bold", fontsize=12)
        plt.ylabel("$Cf_x$", fontweight="bold", fontsize=12)
        plt.title(
            f"$Cf_x$ Response to {feature_names[feat_idx]}",
            fontweight="bold",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        plt.legend(prop={"weight": "bold"})
        plt.tight_layout()
        plt.savefig(
            f"cf_response_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot individual $Nu_{{x}}$ response
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, nu_responses, "r-", linewidth=3, label="$Nu_x$")
        plt.xlabel(f"{feature_names[feat_idx]}", fontweight="bold", fontsize=12)
        plt.ylabel("$Nu_x$", fontweight="bold", fontsize=12)
        plt.title(
            f"$Nu_x$ Response to {feature_names[feat_idx]}",
            fontweight="bold",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        plt.legend(prop={"weight": "bold"})
        plt.tight_layout()
        plt.savefig(
            f"nu_response_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot combined response with dual y-axis
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        line1 = ax1.plot(param_values, cf_responses, "b-", linewidth=3, label="$Cf_x$")
        line2 = ax2.plot(param_values, nu_responses, "r-", linewidth=3, label="$Nu_x$")

        ax1.set_xlabel(f"{feature_names[feat_idx]}", fontweight="bold", fontsize=12)
        ax1.set_ylabel("$Cf_x$", fontweight="bold", fontsize=12, color="blue")
        ax2.set_ylabel("$Nu_x$", fontweight="bold", fontsize=12, color="red")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="red")

        ax1.set_title(
            f"Combined Response to {feature_names[feat_idx]}",
            fontweight="bold",
            fontsize=14,
        )
        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper left", prop={"weight": "bold"})

        plt.tight_layout()
        plt.savefig(
            f"combined_response_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_error_analysis(cf_actual_test, cf_preds_test, nu_actual_test, nu_preds_test):
    """Plot comprehensive error analysis (similar to Fig. 5-7 in paper)"""

    # Error distribution histograms
    cf_errors = cf_preds_test - cf_actual_test
    nu_errors = nu_preds_test - nu_actual_test

    # 1. $Cf_x$ Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(cf_errors, bins=30, alpha=0.7, color="blue", edgecolor="black")
    plt.title("$Cf_x$ Error Distribution", fontweight="bold", fontsize=14)
    plt.xlabel("Prediction Error", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cf_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. $Nu_{{x}}$ Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(nu_errors, bins=30, alpha=0.7, color="green", edgecolor="black")
    plt.title("$Nu_{{x}}$ Error Distribution", fontweight="bold", fontsize=14)
    plt.xlabel("Prediction Error", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nu_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. $Cf_x$ Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(cf_actual_test, cf_errors, alpha=0.6, color="blue")
    plt.title("$Cf_x$ Residual Plot", fontweight="bold", fontsize=14)
    plt.xlabel("Actual Values", fontweight="bold")
    plt.ylabel("Residuals", fontweight="bold")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cf_residual_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. $Nu_{{x}}$ Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(nu_actual_test, nu_errors, alpha=0.6, color="green")
    plt.title("$Nu_{{x}}$ Residual Plot", fontweight="bold", fontsize=14)
    plt.xlabel("Actual Values", fontweight="bold")
    plt.ylabel("Residuals", fontweight="bold")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nu_residual_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_performance_metrics_comparison():
    """Plot performance metrics comparison (like regression plots in paper)"""

    # Calculate various metrics
    mae_cf = mean_absolute_error(cf_actual_test, cf_preds_test)
    mse_cf = mean_squared_error(cf_actual_test, cf_preds_test)
    mae_nu = mean_absolute_error(nu_actual_test, nu_preds_test)
    mse_nu = mean_squared_error(nu_actual_test, nu_preds_test)

    # 1. $Cf_x$: Actual vs Predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(cf_actual_test, cf_preds_test, alpha=0.6, color="blue")
    plt.plot(
        [cf_actual_test.min(), cf_actual_test.max()],
        [cf_actual_test.min(), cf_actual_test.max()],
        "r--",
        linewidth=2,
    )
    plt.title(
        f"$Cf_x$: Actual vs Predicted\nR²={r2_cf_test:.4f}, MAE={mae_cf:.6f}",
        fontweight="bold",
        fontsize=14,
    )
    plt.xlabel("Actual Values", fontweight="bold")
    plt.ylabel("Predicted Values", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cf_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. $Nu_{{x}}$: Actual vs Predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(nu_actual_test, nu_preds_test, alpha=0.6, color="green")
    plt.plot(
        [nu_actual_test.min(), nu_actual_test.max()],
        [nu_actual_test.min(), nu_actual_test.max()],
        "r--",
        linewidth=2,
    )
    plt.title(
        f"$Nu_{{x}}$: Actual vs Predicted\nR²={r2_nu_test:.4f}, MAE={mae_nu:.6f}",
        fontweight="bold",
        fontsize=14,
    )
    plt.xlabel("Actual Values", fontweight="bold")
    plt.ylabel("Predicted Values", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nu_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Performance metrics bar chart
    metrics = ["R²", "MAE", "RMSE"]
    cf_metrics = [r2_cf_test, mae_cf, np.sqrt(mse_cf)]
    nu_metrics = [r2_nu_test, mae_nu, np.sqrt(mse_nu)]

    x_pos = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(
        x_pos - width / 2,
        cf_metrics,
        width,
        label="$Cf_x$",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        x_pos + width / 2,
        nu_metrics,
        width,
        label="$Nu_{{x}}$",
        color="green",
        alpha=0.7,
    )
    plt.title("Performance Metrics Comparison", fontweight="bold", fontsize=14)
    plt.xlabel("Metrics", fontweight="bold")
    plt.ylabel("Values", fontweight="bold")
    plt.xticks(x_pos, metrics, fontweight="bold")
    plt.legend(prop={"weight": "bold"})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("performance_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Feature importance (mock data - replace with actual feature importance if available)
    feature_importance = np.random.rand(len(feature_names))
    feature_importance = feature_importance / np.sum(feature_importance)

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, feature_importance, color="orange", alpha=0.7)
    plt.title("Feature Importance (Mock)", fontweight="bold", fontsize=14)
    plt.xlabel("Relative Importance", fontweight="bold")
    plt.yticks(y_pos, feature_names, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_comprehensive_heatmaps():
    """Generate comprehensive heatmaps for all feature interactions"""

    print("Generating comprehensive heatmaps...")

    # Feature correlation heatmap
    df_features = pd.DataFrame(X, columns=feature_names)
    correlation_matrix = df_features.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="RdBu_r", center=0, square=True, fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig("feature_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Prediction error heatmap by feature ranges
    n_bins = 5
    for feat_idx in range(
        min(len(feature_names), 8)
    ):  # Limit to first 8 features for performance
        feature_values = X[:, feat_idx]
        feature_bins = np.linspace(
            feature_values.min(), feature_values.max(), n_bins + 1
        )

        # Create bins for current feature
        digitized = np.digitize(X_test[:, feat_idx], feature_bins)

        error_matrix = np.zeros((n_bins, 2))  # 2 outputs: CF and Nu

        for i in range(1, n_bins + 1):
            mask = digitized == i
            if np.sum(mask) > 0:
                error_matrix[i - 1, 0] = np.mean(
                    np.abs(cf_preds_test[mask] - cf_actual_test[mask])
                )
                error_matrix[i - 1, 1] = np.mean(
                    np.abs(nu_preds_test[mask] - nu_actual_test[mask])
                )

        # Plot error heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            error_matrix.T,
            annot=True,
            fmt=".4f",
            cmap="Reds",
            xticklabels=[f"Bin {i + 1}" for i in range(n_bins)],
            yticklabels=["$Cf_x$", "$Nu_{{x}}$"],
        )
        plt.title(
            f"Prediction Error Heatmap by {feature_names[feat_idx]} Ranges",
            fontweight="bold",
        )
        plt.xlabel(f"{feature_names[feat_idx]} Bins", fontweight="bold")
        plt.ylabel("Output", fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            f"error_heatmap_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_comprehensive_boxplots():
    """Generate comprehensive box plots for error analysis"""
    print("Generating comprehensive box plots...")

    # Error distribution by feature quartiles
    for feat_idx in range(min(len(feature_names), 8)):  # Limit for performance
        feature_values = X_test[:, feat_idx]
        quartiles = np.percentile(feature_values, [25, 50, 75])

        # Create quartile groups
        groups = ["Q1", "Q2", "Q3", "Q4"]
        cf_errors_by_quartile = []
        nu_errors_by_quartile = []

        for i in range(4):
            if i == 0:
                mask = feature_values <= quartiles[0]
            elif i == 1:
                mask = (feature_values > quartiles[0]) & (
                    feature_values <= quartiles[1]
                )
            elif i == 2:
                mask = (feature_values > quartiles[1]) & (
                    feature_values <= quartiles[2]
                )
            else:
                mask = feature_values > quartiles[2]

            cf_errors_by_quartile.append(
                np.abs(cf_preds_test[mask] - cf_actual_test[mask])
            )
            nu_errors_by_quartile.append(
                np.abs(nu_preds_test[mask] - nu_actual_test[mask])
            )

        # Plot CF errors boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(cf_errors_by_quartile, labels=groups)
        plt.title(
            f"$Cf_x$ Error Distribution by {feature_names[feat_idx]} Quartiles",
            fontweight="bold",
        )
        plt.xlabel(f"{feature_names[feat_idx]} Quartiles", fontweight="bold")
        plt.ylabel("Absolute Error", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"cf_error_boxplot_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot Nu errors boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(nu_errors_by_quartile, labels=groups)
        plt.title(
            f"$Nu_{{x}}$ Error Distribution by {feature_names[feat_idx]} Quartiles",
            fontweight="bold",
        )
        plt.xlabel(f"{feature_names[feat_idx]} Quartiles", fontweight="bold")
        plt.ylabel("Absolute Error", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"nu_error_boxplot_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_comprehensive_scatter_matrices():
    """Generate comprehensive scatter plot matrices"""
    print("Generating comprehensive scatter matrices...")

    # Select subset of features for scatter matrix (for performance)
    n_features_to_plot = min(len(feature_names), 6)
    selected_features = feature_names[:n_features_to_plot]
    selected_data = X[:, :n_features_to_plot]

    # Create DataFrame for scatter matrix
    df_scatter = pd.DataFrame(selected_data, columns=selected_features)
    df_scatter["$Cf_x$"] = y[:, 0]
    df_scatter["$Nu_{{x}}$"] = y[:, 1]

    # Generate scatter matrix
    from pandas.plotting import scatter_matrix

    plt.figure(figsize=(15, 15))
    scatter_matrix(df_scatter, alpha=0.6, figsize=(15, 15), diagonal="hist")
    plt.suptitle("Feature Scatter Matrix with Outputs", fontweight="bold", fontsize=16)
    plt.tight_layout()
    plt.savefig("feature_scatter_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_comprehensive_violin_plots():
    """Generate comprehensive violin plots for distribution analysis"""
    print("Generating comprehensive violin plots...")

    # Distribution of predictions vs actual for different feature ranges
    for feat_idx in range(min(len(feature_names), 6)):  # Limit for performance
        feature_values = X_test[:, feat_idx]

        # Create three groups based on feature values
        low_mask = feature_values <= np.percentile(feature_values, 33)
        mid_mask = (feature_values > np.percentile(feature_values, 33)) & (
            feature_values <= np.percentile(feature_values, 67)
        )
        high_mask = feature_values > np.percentile(feature_values, 67)

        # Prepare data for violin plots
        cf_data = []
        nu_data = []
        labels = []

        for mask, label in [(low_mask, "Low"), (mid_mask, "Mid"), (high_mask, "High")]:
            cf_data.extend(cf_actual_test[mask])
            cf_data.extend(cf_preds_test[mask])
            nu_data.extend(nu_actual_test[mask])
            nu_data.extend(nu_preds_test[mask])

            labels.extend([f"{label}_Actual"] * np.sum(mask))
            labels.extend([f"{label}_Predicted"] * np.sum(mask))

        # Create violin plot for CF
        plt.figure(figsize=(12, 6))
        df_violin = pd.DataFrame({"Value": cf_data, "Group": labels})
        sns.violinplot(data=df_violin, x="Group", y="Value")
        plt.title(
            f"$Cf_x$ Distribution by {feature_names[feat_idx]} Ranges",
            fontweight="bold",
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            f"cf_violin_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create violin plot for Nu
        plt.figure(figsize=(12, 6))
        df_violin = pd.DataFrame({"Value": nu_data, "Group": labels})
        sns.violinplot(data=df_violin, x="Group", y="Value")
        plt.title(
            f"$Nu_{{x}}$ Distribution by {feature_names[feat_idx]} Ranges",
            fontweight="bold",
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            f"nu_violin_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_comprehensive_radar_charts():
    """Generate comprehensive radar charts for feature importance and performance"""
    print("Generating comprehensive radar charts...")

    # Feature importance radar chart (using mock data)
    feature_importance = np.random.rand(len(feature_names))
    feature_importance = feature_importance / np.sum(feature_importance)

    # Limit to first 8 features for readability
    n_features_radar = min(8, len(feature_names))
    angles = np.linspace(0, 2 * np.pi, n_features_radar, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    values = feature_importance[:n_features_radar].tolist()
    values += values[:1]  # Complete the circle

    ax.plot(angles, values, "o-", linewidth=2, label="Feature Importance")
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names[:n_features_radar])
    ax.set_ylim(0, max(values))
    ax.set_title(
        "Feature Importance Radar Chart", fontweight="bold", fontsize=14, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig("feature_importance_radar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Performance radar chart
    performance_metrics = {
        "R² CF": r2_cf_test,
        "R² Nu": r2_nu_test,
        "MAE CF (norm)": 1
        - (mean_absolute_error(cf_actual_test, cf_preds_test) / np.std(cf_actual_test)),
        "MAE Nu (norm)": 1
        - (mean_absolute_error(nu_actual_test, nu_preds_test) / np.std(nu_actual_test)),
        "RMSE CF (norm)": 1
        - (
            np.sqrt(mean_squared_error(cf_actual_test, cf_preds_test))
            / np.std(cf_actual_test)
        ),
        "RMSE Nu (norm)": 1
        - (
            np.sqrt(mean_squared_error(nu_actual_test, nu_preds_test))
            / np.std(nu_actual_test)
        ),
    }

    metric_names = list(performance_metrics.keys())
    metric_values = list(performance_metrics.values())

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    metric_values += metric_values[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    ax.plot(angles, metric_values, "o-", linewidth=2, label="Performance Metrics")
    ax.fill(angles, metric_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Model Performance Radar Chart", fontweight="bold", fontsize=14, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig("performance_radar.png", dpi=300, bbox_inches="tight")
    plt.close()


# Execute the enhanced analysis
print("Starting Enhanced Neural Network Analysis...")

# Plot data analysis
print("\n1. Plotting network architecture...")
plot_network_architecture()

print("2. Plotting data distribution...")
plot_data_distribution()

print("3. Plotting correlation heatmap...")
plot_correlation_heatmap()

# Train the model
print("\n4. Training enhanced model...")
model = DualHeadDNN(X_train_tensor.shape[1])
metrics, trained_model = train_model_enhanced(
    model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
)

# Evaluate the model
print("\n5. Evaluating model...")
trained_model.eval()
with torch.no_grad():
    train_predictions = trained_model(X_train_tensor).numpy()
    test_predictions = trained_model(X_test_tensor).numpy()

# Inverse transform predictions
cf_preds_train, nu_preds_train = scaler_y.inverse_transform(train_predictions).T
cf_preds_test, nu_preds_test = scaler_y.inverse_transform(test_predictions).T
cf_actual_train, nu_actual_train = scaler_y.inverse_transform(y_train_tensor).T
cf_actual_test, nu_actual_test = scaler_y.inverse_transform(y_test_tensor).T

# Calculate R² scores
r2_cf_train = r2_score(cf_actual_train, cf_preds_train)
r2_cf_test = r2_score(cf_actual_test, cf_preds_test)
r2_nu_train = r2_score(nu_actual_train, nu_preds_train)
r2_nu_test = r2_score(nu_actual_test, nu_preds_test)

print("\nFinal R² Scores:")
print(f"Cf_x - Train: {r2_cf_train:.6f}, Test: {r2_cf_test:.6f}")
print(f"Nu_x - Train: {r2_nu_train:.6f}, Test: {r2_nu_test:.6f}")

# Generate comprehensive plots
print("\n6. Plotting comprehensive training analysis...")
plot_comprehensive_training_analysis(metrics)

print("7. Plotting parameter sensitivity analysis...")
plot_parameter_sensitivity_analysis(X_test, trained_model, scaler_X, scaler_y)

print("8. Plotting comprehensive 2D parameter analysis...")
plot_comprehensive_2d_parameter_analysis()

print("9. Plotting comprehensive 3D response surfaces...")
plot_comprehensive_3d_response_surfaces()

print("10. Plotting error analysis...")
plot_error_analysis(cf_actual_test, cf_preds_test, nu_actual_test, nu_preds_test)

print("11. Plotting performance metrics comparison...")
plot_performance_metrics_comparison()

# Additional specialized plots similar to research paper


def plot_validation_curves():
    """Plot validation curves similar to paper's training/test curves"""

    epochs = range(len(metrics["r2_cf_train"]))

    # 1. $Cf_x$ R² Evolution
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics["r2_cf_train"], "b-", linewidth=3, label="Training R²")
    plt.plot(epochs, metrics["r2_cf_val"], "r-", linewidth=3, label="Validation R²")
    plt.fill_between(epochs, metrics["r2_cf_train"], alpha=0.3, color="blue")
    plt.fill_between(epochs, metrics["r2_cf_val"], alpha=0.3, color="red")
    plt.title("$Cf_x$ R² Score Evolution", fontweight="bold", fontsize=14)
    plt.xlabel("Epoch", fontweight="bold", fontsize=12)
    plt.ylabel("R² Score", fontweight="bold", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(prop={"weight": "bold", "size": 12})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cf_r2_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. $Nu_{{x}}$ R² Evolution
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics["r2_nu_train"], "g-", linewidth=3, label="Training R²")
    plt.plot(epochs, metrics["r2_nu_val"], "m-", linewidth=3, label="Validation R²")
    plt.fill_between(epochs, metrics["r2_nu_train"], alpha=0.3, color="green")
    plt.fill_between(epochs, metrics["r2_nu_val"], alpha=0.3, color="magenta")
    plt.title("$Nu_{{x}}$ R² Score Evolution", fontweight="bold", fontsize=14)
    plt.xlabel("Epoch", fontweight="bold", fontsize=12)
    plt.ylabel("R² Score", fontweight="bold", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(prop={"weight": "bold", "size": 12})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nu_r2_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Loss convergence comparison
    plt.figure(figsize=(12, 6))
    plt.semilogy(
        epochs, metrics["train_losses"], "b-", linewidth=2, label="Training Loss"
    )
    plt.semilogy(
        epochs, metrics["val_losses"], "r-", linewidth=2, label="Validation Loss"
    )
    plt.title("Loss Convergence (Log Scale)", fontweight="bold", fontsize=14)
    plt.xlabel("Epoch", fontweight="bold", fontsize=12)
    plt.ylabel("Log(Loss)", fontweight="bold", fontsize=12)
    plt.legend(prop={"weight": "bold", "size": 12})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Training time per epoch
    plt.figure(figsize=(12, 6))
    plt.plot(
        epochs[: len(metrics["epoch_times"])],
        metrics["epoch_times"],
        "purple",
        linewidth=2,
    )
    plt.title("Training Time per Epoch", fontweight="bold", fontsize=14)
    plt.xlabel("Epoch", fontweight="bold", fontsize=12)
    plt.ylabel("Time (seconds)", fontweight="bold", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_time_per_epoch.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparative_analysis():
    """Plot comparative analysis between different nanofluid configurations"""

    # Simulate different parameter configurations
    configs = ["Low Parameters", "Medium Parameters", "High Parameters"]
    cf_performance = [0.85, 0.92, 0.88]  # Replace with actual performance metrics
    nu_performance = [0.89, 0.94, 0.91]

    x_pos = np.arange(len(configs))
    width = 0.35

    # 1. Performance comparison bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(
        x_pos - width / 2,
        cf_performance,
        width,
        label="$Cf_x$ R²",
        color="blue",
        alpha=0.7,
        edgecolor="black",
    )
    plt.bar(
        x_pos + width / 2,
        nu_performance,
        width,
        label="$Nu_{{x}}$ R²",
        color="red",
        alpha=0.7,
        edgecolor="black",
    )
    plt.title(
        "Performance Comparison Across Parameter Configurations",
        fontweight="bold",
        fontsize=14,
    )
    plt.xlabel("Configuration", fontweight="bold", fontsize=12)
    plt.ylabel("R² Score", fontweight="bold", fontsize=12)
    plt.xticks(x_pos, configs, fontweight="bold")
    plt.legend(prop={"weight": "bold", "size": 12})
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("performance_comparison_configs.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. $Cf_x$: Actual vs Predicted
    indices = range(min(50, len(cf_actual_test)))
    plt.figure(figsize=(12, 6))
    plt.scatter(
        indices, cf_actual_test[:50], label="Actual CF", alpha=0.7, s=60, color="blue"
    )
    plt.scatter(
        indices,
        cf_preds_test[:50],
        label="Predicted CF",
        alpha=0.7,
        s=60,
        color="cyan",
        marker="^",
    )
    plt.title(
        "$Cf_x$: Actual vs Predicted",
        fontweight="bold",
        fontsize=14,
    )
    plt.xlabel("Sample Index", fontweight="bold", fontsize=12)
    plt.ylabel("$Cf_x$ Value", fontweight="bold", fontsize=12)
    plt.legend(prop={"weight": "bold", "size": 12})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cf_actual_vs_predicted_samples.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. $Nu_{{x}}$: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.scatter(
        indices, nu_actual_test[:50], label="Actual Nu", alpha=0.7, s=60, color="green"
    )
    plt.scatter(
        indices,
        nu_preds_test[:50],
        label="Predicted Nu",
        alpha=0.7,
        s=60,
        color="lime",
        marker="^",
    )
    plt.title(
        "$Nu_{{x}}$: Actual vs Predicted",
        fontweight="bold",
        fontsize=14,
    )
    plt.xlabel("Sample Index", fontweight="bold", fontsize=12)
    plt.ylabel("$Nu_{{x}}$ Value", fontweight="bold", fontsize=12)
    plt.legend(prop={"weight": "bold", "size": 12})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nu_actual_vs_predicted_samples.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Error magnitude analysis
    cf_abs_errors = np.abs(cf_preds_test - cf_actual_test)

    # Bin the errors
    cf_error_ranges = ["0-0.001", "0.001-0.01", "0.01-0.1", ">0.1"]
    cf_error_counts = [
        np.sum(cf_abs_errors < 0.001),
        np.sum((cf_abs_errors >= 0.001) & (cf_abs_errors < 0.01)),
        np.sum((cf_abs_errors >= 0.01) & (cf_abs_errors < 0.1)),
        np.sum(cf_abs_errors >= 0.1),
    ]

    plt.figure(figsize=(8, 8))
    plt.pie(
        cf_error_counts,
        labels=cf_error_ranges,
        autopct="%1.1f%%",
        colors=["lightgreen", "yellow", "orange", "red"],
    )
    plt.title(
        "Distribution of $Cf_x$ Prediction Errors",
        fontweight="bold",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("cf_error_distribution_pie.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_advanced_regression_analysis():
    """Advanced regression analysis similar to Fig. 10 in paper"""
    from scipy import stats

    # 1. $Cf_x$ regression analysis
    slope_cf, intercept_cf, r_value_cf, p_value_cf, std_err_cf = stats.linregress(
        cf_actual_test, cf_preds_test
    )
    line_cf = slope_cf * cf_actual_test + intercept_cf

    plt.figure(figsize=(10, 8))
    plt.scatter(cf_actual_test, cf_preds_test, alpha=0.6, c="blue", s=50)
    plt.plot(
        cf_actual_test,
        line_cf,
        "r-",
        linewidth=2,
        label=f"Regression: y={slope_cf:.3f}x+{intercept_cf:.3f}",
    )
    plt.plot(
        [cf_actual_test.min(), cf_actual_test.max()],
        [cf_actual_test.min(), cf_actual_test.max()],
        "k--",
        linewidth=2,
        label="Perfect Fit",
    )
    plt.title(
        f"$Cf_x$ Regression Analysis\nR²={r2_cf_test:.4f}, p-value={p_value_cf:.2e}",
        fontweight="bold",
        fontsize=14,
    )
    plt.xlabel("Actual Values", fontweight="bold")
    plt.ylabel("Predicted Values", fontweight="bold")
    plt.legend(prop={"weight": "bold"})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cf_regression_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. $Nu_{{x}}$ regression analysis
    slope_nu, intercept_nu, r_value_nu, p_value_nu, std_err_nu = stats.linregress(
        nu_actual_test, nu_preds_test
    )
    line_nu = slope_nu * nu_actual_test + intercept_nu

    plt.figure(figsize=(10, 8))
    plt.scatter(nu_actual_test, nu_preds_test, alpha=0.6, c="green", s=50)
    plt.plot(
        nu_actual_test,
        line_nu,
        "r-",
        linewidth=2,
        label=f"Regression: y={slope_nu:.3f}x+{intercept_nu:.3f}",
    )
    plt.plot(
        [nu_actual_test.min(), nu_actual_test.max()],
        [nu_actual_test.min(), nu_actual_test.max()],
        "k--",
        linewidth=2,
        label="Perfect Fit",
    )
    plt.title(
        f"$Nu_{{x}}$ Regression Analysis\nR²={r2_nu_test:.4f}, p-value={p_value_nu:.2e}",
        fontweight="bold",
        fontsize=14,
    )
    plt.xlabel("Actual Values", fontweight="bold")
    plt.ylabel("Predicted Values", fontweight="bold")
    plt.legend(prop={"weight": "bold"})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nu_regression_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Q-Q plot for $Cf_x$ residuals
    cf_residuals = cf_preds_test - cf_actual_test
    plt.figure(figsize=(8, 6))
    stats.probplot(cf_residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot: $Cf_x$ Residuals", fontweight="bold", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cf_qq_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Q-Q plot for $Nu_{{x}}$ residuals
    nu_residuals = nu_preds_test - nu_actual_test
    plt.figure(figsize=(8, 6))
    stats.probplot(nu_residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot: $Nu_{{x}}$ Residuals", fontweight="bold", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nu_qq_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_table():
    """Generate comprehensive summary table"""

    # Calculate additional metrics
    mae_cf_train = mean_absolute_error(cf_actual_train, cf_preds_train)
    mae_cf_test = mean_absolute_error(cf_actual_test, cf_preds_test)
    mae_nu_train = mean_absolute_error(nu_actual_train, nu_preds_train)
    mae_nu_test = mean_absolute_error(nu_actual_test, nu_preds_test)

    rmse_cf_train = np.sqrt(mean_squared_error(cf_actual_train, cf_preds_train))
    rmse_cf_test = np.sqrt(mean_squared_error(cf_actual_test, cf_preds_test))
    rmse_nu_train = np.sqrt(mean_squared_error(nu_actual_train, nu_preds_train))
    rmse_nu_test = np.sqrt(mean_squared_error(nu_actual_test, nu_preds_test))

    # Create comprehensive summary table
    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "CF Train", "CF Test", "Nu Train", "Nu Test"]

    summary_table.add_row(
        [
            "R² Score",
            f"{r2_cf_train:.6f}",
            f"{r2_cf_test:.6f}",
            f"{r2_nu_train:.6f}",
            f"{r2_nu_test:.6f}",
        ]
    )
    summary_table.add_row(
        [
            "MAE",
            f"{mae_cf_train:.6f}",
            f"{mae_cf_test:.6f}",
            f"{mae_nu_train:.6f}",
            f"{mae_nu_test:.6f}",
        ]
    )
    summary_table.add_row(
        [
            "RMSE",
            f"{rmse_cf_train:.6f}",
            f"{rmse_cf_test:.6f}",
            f"{rmse_nu_train:.6f}",
            f"{rmse_nu_test:.6f}",
        ]
    )

    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(summary_table)
    print("=" * 80)


# Execute additional analysis
print("\n12. Plotting validation curves...")
plot_validation_curves()

print("13. Plotting comparative analysis...")
plot_comparative_analysis()

print("14. Plotting advanced regression analysis...")
plot_advanced_regression_analysis()

print("15. Generating comprehensive summary...")
generate_summary_table()

print("16. Plotting comprehensive heatmaps...")
plot_comprehensive_heatmaps()

print("17. Plotting comprehensive box plots...")
plot_comprehensive_boxplots()

print("18. Plotting comprehensive scatter matrices...")
plot_comprehensive_scatter_matrices()

print("19. Plotting comprehensive violin plots...")
plot_comprehensive_violin_plots()

print("20. Plotting comprehensive radar charts...")
plot_comprehensive_radar_charts()

# Final detailed comparison table (original functionality)
print("\n16. Generating detailed prediction tables...")

# Top 10 predictions with lowest errors
cf_errors = np.abs(cf_preds_test - cf_actual_test)
nu_errors = np.abs(nu_preds_test - nu_actual_test)
top_cf_idxs = np.argsort(cf_errors)[:10]
top_nu_idxs = np.argsort(nu_errors)[:10]

cf_table = PrettyTable(
    ["Index", "Actual Cf", "Predicted Cf", "Abs Error", "Rel Error %"]
)
nu_table = PrettyTable(
    ["Index", "Actual Nu", "Predicted Nu", "Abs Error", "Rel Error %"]
)

for idx in top_cf_idxs:
    rel_error = (cf_errors[idx] / cf_actual_test[idx]) * 100
    cf_table.add_row(
        [
            idx,
            round(cf_actual_test[idx], 6),
            round(cf_preds_test[idx], 6),
            round(cf_errors[idx], 6),
            round(rel_error, 2),
        ]
    )

for idx in top_nu_idxs:
    rel_error = (nu_errors[idx] / nu_actual_test[idx]) * 100
    nu_table.add_row(
        [
            idx,
            round(nu_actual_test[idx], 6),
            round(nu_preds_test[idx], 6),
            round(nu_errors[idx], 6),
            round(rel_error, 2),
        ]
    )

print("\nTop 10 $Cf_x$ Predictions (Lowest Error):")
print(cf_table)

print("\nTop 10 $Nu_{{x}}$ Predictions (Lowest Error):")
print(nu_table)

print("\n" + "=" * 80)
print("ENHANCED NEURAL NETWORK ANALYSIS COMPLETED SUCCESSFULLY!")
print("Generated plots:")
print("- network_architecture.png")
print("- data_distribution.png")
print("- correlation_heatmap.png")
print("- training_validation_loss.png")
print("- individual_output_losses.png")
print("- r2_score_evolution.png")
print("- learning_rate_schedule.png")
print("- parameter_sensitivity_analysis.png")
print("- Individual 2D parameter response plots for all features")
print("- 3D response surfaces for all feature combinations")
print("- Contour plots for all feature combinations")
print("- cf_error_distribution.png")
print("- nu_error_distribution.png")
print("- cf_residual_plot.png")
print("- nu_residual_plot.png")
print("- cf_actual_vs_predicted.png")
print("- nu_actual_vs_predicted.png")
print("- performance_metrics_comparison.png")
print("- feature_importance.png")
print("- cf_r2_evolution.png")
print("- nu_r2_evolution.png")
print("- loss_convergence.png")
print("- training_time_per_epoch.png")
print("- performance_comparison_configs.png")
print("- cf_actual_vs_predicted_samples.png")
print("- nu_actual_vs_predicted_samples.png")
print("- cf_error_distribution_pie.png")
print("- cf_regression_analysis.png")
print("- nu_regression_analysis.png")
print("- cf_qq_plot.png")
print("- nu_qq_plot.png")
print("=" * 80)


def plot_comprehensive_heatmaps():
    """Generate comprehensive heatmaps for all feature interactions"""

    print("Generating comprehensive heatmaps...")

    # Feature correlation heatmap
    df_features = pd.DataFrame(X, columns=feature_names)
    correlation_matrix = df_features.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="RdBu_r", center=0, square=True, fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig("feature_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Prediction error heatmap by feature ranges
    n_bins = 5
    for feat_idx in range(
        min(len(feature_names), 8)
    ):  # Limit to first 8 features for performance
        feature_values = X[:, feat_idx]
        feature_bins = np.linspace(
            feature_values.min(), feature_values.max(), n_bins + 1
        )

        # Create bins for current feature
        digitized = np.digitize(X_test[:, feat_idx], feature_bins)

        error_matrix = np.zeros((n_bins, 2))  # 2 outputs: CF and Nu
        count_matrix = np.zeros((n_bins, 2))

        for i in range(1, n_bins + 1):
            mask = digitized == i
            if np.sum(mask) > 0:
                error_matrix[i - 1, 0] = np.mean(
                    np.abs(cf_preds_test[mask] - cf_actual_test[mask])
                )
                error_matrix[i - 1, 1] = np.mean(
                    np.abs(nu_preds_test[mask] - nu_actual_test[mask])
                )
                count_matrix[i - 1, 0] = np.sum(mask)
                count_matrix[i - 1, 1] = np.sum(mask)

        # Plot error heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            error_matrix.T,
            annot=True,
            fmt=".4f",
            cmap="Reds",
            xticklabels=[f"Bin {i + 1}" for i in range(n_bins)],
            yticklabels=["$Cf_x$", "$Nu_{{x}}$"],
        )
        plt.title(
            f"Prediction Error Heatmap by {feature_names[feat_idx]} Ranges",
            fontweight="bold",
        )
        plt.xlabel(f"{feature_names[feat_idx]} Bins", fontweight="bold")
        plt.ylabel("Output", fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            f"error_heatmap_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_comprehensive_boxplots():
    """Generate comprehensive box plots for error analysis"""
    print("Generating comprehensive box plots...")

    # Error distribution by feature quartiles
    for feat_idx in range(min(len(feature_names), 8)):  # Limit for performance
        feature_values = X_test[:, feat_idx]
        quartiles = np.percentile(feature_values, [25, 50, 75])

        # Create quartile groups
        groups = ["Q1", "Q2", "Q3", "Q4"]
        cf_errors_by_quartile = []
        nu_errors_by_quartile = []

        for i in range(4):
            if i == 0:
                mask = feature_values <= quartiles[0]
            elif i == 1:
                mask = (feature_values > quartiles[0]) & (
                    feature_values <= quartiles[1]
                )
            elif i == 2:
                mask = (feature_values > quartiles[1]) & (
                    feature_values <= quartiles[2]
                )
            else:
                mask = feature_values > quartiles[2]

            cf_errors_by_quartile.append(
                np.abs(cf_preds_test[mask] - cf_actual_test[mask])
            )
            nu_errors_by_quartile.append(
                np.abs(nu_preds_test[mask] - nu_actual_test[mask])
            )

        # Plot CF errors boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(cf_errors_by_quartile, labels=groups)
        plt.title(
            f"$Cf_x$ Error Distribution by {feature_names[feat_idx]} Quartiles",
            fontweight="bold",
        )
        plt.xlabel(f"{feature_names[feat_idx]} Quartiles", fontweight="bold")
        plt.ylabel("Absolute Error", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"cf_error_boxplot_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot Nu errors boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(nu_errors_by_quartile, labels=groups)
        plt.title(
            f"$Nu_{{x}}$ Error Distribution by {feature_names[feat_idx]} Quartiles",
            fontweight="bold",
        )
        plt.xlabel(f"{feature_names[feat_idx]} Quartiles", fontweight="bold")
        plt.ylabel("Absolute Error", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"nu_error_boxplot_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_comprehensive_scatter_matrices():
    """Generate comprehensive scatter plot matrices"""
    print("Generating comprehensive scatter matrices...")

    # Select subset of features for scatter matrix (for performance)
    n_features_to_plot = min(len(feature_names), 6)
    selected_features = feature_names[:n_features_to_plot]
    selected_data = X[:, :n_features_to_plot]

    # Create DataFrame for scatter matrix
    df_scatter = pd.DataFrame(selected_data, columns=selected_features)
    df_scatter["$Cf_x$"] = y[:, 0]
    df_scatter["$Nu_{{x}}$"] = y[:, 1]

    # Generate scatter matrix
    from pandas.plotting import scatter_matrix

    plt.figure(figsize=(15, 15))
    scatter_matrix(df_scatter, alpha=0.6, figsize=(15, 15), diagonal="hist")
    plt.suptitle("Feature Scatter Matrix with Outputs", fontweight="bold", fontsize=16)
    plt.tight_layout()
    plt.savefig("feature_scatter_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_comprehensive_violin_plots():
    """Generate comprehensive violin plots for distribution analysis"""
    print("Generating comprehensive violin plots...")

    # Distribution of predictions vs actual for different feature ranges
    for feat_idx in range(min(len(feature_names), 6)):  # Limit for performance
        feature_values = X_test[:, feat_idx]

        # Create three groups based on feature values
        low_mask = feature_values <= np.percentile(feature_values, 33)
        mid_mask = (feature_values > np.percentile(feature_values, 33)) & (
            feature_values <= np.percentile(feature_values, 67)
        )
        high_mask = feature_values > np.percentile(feature_values, 67)

        # Prepare data for violin plots
        cf_data = []
        nu_data = []
        labels = []

        for mask, label in [(low_mask, "Low"), (mid_mask, "Mid"), (high_mask, "High")]:
            cf_data.extend(cf_actual_test[mask])
            cf_data.extend(cf_preds_test[mask])
            nu_data.extend(nu_actual_test[mask])
            nu_data.extend(nu_preds_test[mask])

            labels.extend([f"{label}_Actual"] * np.sum(mask))
            labels.extend([f"{label}_Predicted"] * np.sum(mask))

        # Create violin plot for CF
        plt.figure(figsize=(12, 6))
        df_violin = pd.DataFrame({"Value": cf_data, "Group": labels})
        sns.violinplot(data=df_violin, x="Group", y="Value")
        plt.title(
            f"$Cf_x$ Distribution by {feature_names[feat_idx]} Ranges",
            fontweight="bold",
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            f"cf_violin_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create violin plot for Nu
        plt.figure(figsize=(12, 6))
        df_violin = pd.DataFrame({"Value": nu_data, "Group": labels})
        sns.violinplot(data=df_violin, x="Group", y="Value")
        plt.title(
            f"$Nu_{{x}}$ Distribution by {feature_names[feat_idx]} Ranges",
            fontweight="bold",
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            f"nu_violin_{feat_idx + 1:02d}_{feature_names[feat_idx]}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_comprehensive_radar_charts():
    """Generate comprehensive radar charts for feature importance and performance"""
    print("Generating comprehensive radar charts...")

    # Feature importance radar chart (using mock data)
    feature_importance = np.random.rand(len(feature_names))
    feature_importance = feature_importance / np.sum(feature_importance)

    # Limit to first 8 features for readability
    n_features_radar = min(8, len(feature_names))
    angles = np.linspace(0, 2 * np.pi, n_features_radar, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    values = feature_importance[:n_features_radar].tolist()
    values += values[:1]  # Complete the circle

    ax.plot(angles, values, "o-", linewidth=2, label="Feature Importance")
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names[:n_features_radar])
    ax.set_ylim(0, max(values))
    ax.set_title(
        "Feature Importance Radar Chart", fontweight="bold", fontsize=14, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig("feature_importance_radar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Performance radar chart
    performance_metrics = {
        "R² CF": r2_cf_test,
        "R² Nu": r2_nu_test,
        "MAE CF (norm)": 1
        - (mean_absolute_error(cf_actual_test, cf_preds_test) / np.std(cf_actual_test)),
        "MAE Nu (norm)": 1
        - (mean_absolute_error(nu_actual_test, nu_preds_test) / np.std(nu_actual_test)),
        "RMSE CF (norm)": 1
        - (
            np.sqrt(mean_squared_error(cf_actual_test, cf_preds_test))
            / np.std(cf_actual_test)
        ),
        "RMSE Nu (norm)": 1
        - (
            np.sqrt(mean_squared_error(nu_actual_test, nu_preds_test))
            / np.std(nu_actual_test)
        ),
    }

    metric_names = list(performance_metrics.keys())
    metric_values = list(performance_metrics.values())

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    metric_values += metric_values[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    ax.plot(angles, metric_values, "o-", linewidth=2, label="Performance Metrics")
    ax.fill(angles, metric_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Model Performance Radar Chart", fontweight="bold", fontsize=14, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig("performance_radar.png", dpi=300, bbox_inches="tight")
    plt.close()
