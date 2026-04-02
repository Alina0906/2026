import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cirq
except Exception:  # pragma: no cover - optional dependency
    cirq = None


REAL_DTYPE = torch.float32
COMPLEX_DTYPE = torch.complex64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ancilla_fidelity_from_probs(probs: Iterable[float]) -> float:
    probs = np.asarray(list(probs), dtype=float)
    return float(np.clip(2.0 * probs[0] - 1.0, 0.0, 1.0))


def save_json(payload: dict, destination: Path) -> None:
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, Path):
        return str(value)
    return np.asarray(value).tolist()


def save_pennylane_plot(qnode, destination: Path, title: str) -> None:
    ensure_dir(destination.parent)
    fig, _ = qml.draw_mpl(qnode)()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_task1_first_circuit():
    dev = qml.device("default.qubit", wires=5)

    @qml.qnode(dev)
    def circuit():
        for wire in range(5):
            qml.Hadamard(wires=wire)
        for control, target in ((0, 1), (1, 2), (2, 3), (3, 4)):
            qml.CNOT(wires=[control, target])
        qml.SWAP(wires=[0, 4])
        qml.RX(math.pi / 2.0, wires=2)
        return qml.state()

    return circuit


def build_task1_second_circuit():
    dev = qml.device("default.qubit", wires=5)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=1)
        qml.RX(math.pi / 3.0, wires=2)
        qml.Hadamard(wires=3)
        qml.Hadamard(wires=4)
        qml.Hadamard(wires=0)
        qml.CSWAP(wires=[0, 1, 3])
        qml.CSWAP(wires=[0, 2, 4])
        qml.Hadamard(wires=0)
        return qml.probs(wires=0)

    return circuit


def build_task1_second_cirq_circuit():
    if cirq is None:
        raise ImportError("cirq is not installed.")
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[1]))
    circuit.append(cirq.rx(math.pi / 3.0)(qubits[2]))
    circuit.append(cirq.H(qubits[3]))
    circuit.append(cirq.H(qubits[4]))
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CSWAP(qubits[0], qubits[1], qubits[3]))
    circuit.append(cirq.CSWAP(qubits[0], qubits[2], qubits[4]))
    circuit.append(cirq.H(qubits[0]))
    return circuit


def simulate_task1_second_cirq():
    circuit = build_task1_second_cirq_circuit()
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    state = result.final_state_vector
    probs = np.zeros(2, dtype=float)
    for index, amplitude in enumerate(state):
        ancilla = (index >> 4) & 1
        probs[ancilla] += float(np.abs(amplitude) ** 2)
    return circuit, probs


def z2_x_z2_dataset(n_samples: int = 512, seed: int = 7):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1.0, 1.0, size=(n_samples // 4, 2))
    class0 = []
    class1 = []
    for x1, x2 in base:
        mirror = np.array([x2, x1])
        flip = np.array([-x1, -x2])
        flip_mirror = np.array([-x2, -x1])
        target = class0 if x1 * x2 >= 0 else class1
        target.extend([np.array([x1, x2]), mirror, flip, flip_mirror])
    x = np.asarray(class0 + class1, dtype=np.float32)
    y = np.asarray([0] * len(class0) + [1] * len(class1), dtype=np.int64)
    return x, y


class TinyQuantumClassifier(nn.Module):
    def __init__(self, num_qubits: int = 2, equivariant: bool = False):
        super().__init__()
        self.num_qubits = num_qubits
        self.equivariant = equivariant
        if equivariant:
            self.theta_shared = nn.Parameter(torch.randn(2, dtype=REAL_DTYPE) * 0.1)
        else:
            self.theta = nn.Parameter(torch.randn(2, num_qubits, dtype=REAL_DTYPE) * 0.1)

    def _angles(self, features: torch.Tensor) -> torch.Tensor:
        x1 = features[:, 0]
        x2 = features[:, 1]
        return torch.stack([x1, x2], dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        angles = self._angles(features)
        probs = []
        for row in angles:
            dev = qml.device("default.qubit", wires=self.num_qubits)

            @qml.qnode(dev, interface="torch")
            def circuit():
                for wire in range(self.num_qubits):
                    qml.RY(row[wire], wires=wire)
                if self.equivariant:
                    qml.RZ(self.theta_shared[0], wires=0)
                    qml.RZ(self.theta_shared[0], wires=1)
                    qml.CNOT(wires=[0, 1])
                    qml.RY(self.theta_shared[1], wires=0)
                    qml.RY(self.theta_shared[1], wires=1)
                else:
                    for layer in range(2):
                        for wire in range(self.num_qubits):
                            qml.RY(self.theta[layer, wire], wires=wire)
                        qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0))

            probs.append(circuit())
        logits = torch.stack(probs)
        return logits.unsqueeze(1)


class PQCRegressor(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_qubits: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits * 2),
        )

    def predict_parameters(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parameters = self.predict_parameters(x)
        batch_states = []
        for param_row in parameters:
            dev = qml.device("default.qubit", wires=self.num_qubits)

            @qml.qnode(dev, interface="torch")
            def circuit():
                for wire in range(self.num_qubits):
                    qml.RY(param_row[wire], wires=wire)
                    qml.RZ(param_row[wire + self.num_qubits], wires=wire)
                for wire in range(self.num_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                return qml.state()

            state = circuit()
            # PennyLane may return float64-backed states even when the model parameters are float32.
            batch_states.append(torch.view_as_real(state).reshape(-1).to(dtype=REAL_DTYPE))
        return torch.stack(batch_states)


def target_state_from_input(x: torch.Tensor, num_qubits: int) -> torch.Tensor:
    weights = torch.linspace(0.5, 1.5, x.shape[1], dtype=x.dtype, device=x.device)
    projected = torch.tanh(x * weights)
    repeated = projected.repeat_interleave(math.ceil((2**num_qubits) / x.shape[1]), dim=1)
    raw = repeated[:, : 2**num_qubits]
    raw = raw / torch.linalg.norm(raw, dim=1, keepdim=True).clamp_min(1e-8)
    imag = torch.zeros_like(raw)
    state = torch.complex(raw, imag)
    return torch.view_as_real(state).reshape(x.shape[0], -1)


def qgnn_circuit_from_adjacency(node_features: np.ndarray, adjacency: np.ndarray):
    num_nodes = node_features.shape[0]
    dev = qml.device("default.qubit", wires=num_nodes)

    @qml.qnode(dev)
    def circuit():
        for wire in range(num_nodes):
            qml.RY(float(node_features[wire, 0]), wires=wire)
            qml.RZ(float(node_features[wire, 1]), wires=wire)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adjacency[i, j] > 0:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(float(adjacency[i, j]), wires=j)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(num_nodes)]

    return circuit


def quick_quantum_summary(config, metrics: Optional[dict] = None) -> dict:
    summary = {"config": asdict(config) if hasattr(config, "__dataclass_fields__") else config}
    if metrics is not None:
        summary["metrics"] = metrics
    return summary
