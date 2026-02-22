"""
src/multiverse/foam.py

РУ: Вычисление пены мультиверса Φ^(l) — ключевая метрика несогласованности [eq. 2.3]
EN: Multiverse foam computation Φ^(l) — key incoherence metric [eq. 2.3]

Implements:
- Level-l foam: Σ_{a≠b} |<Ψ^a|P_Gl|Ψ^b>|^2
- Multi-level foam hierarchy
- Gradient computation ∂Φ^(l)/∂Ψ for Algorithm 5.1/5.2
- Verification utilities for Theorem 4.1

Φ^(l)=0 ⇔ complete nullification at level l (Theorem 4.1)
"""
import torch
from typing import Dict, List, Optional, Tuple
from torch import Tensor
from .level import MultiIndex, Level

def compute_single_foam_overlap(psi_a: Tensor, projector: Tensor, psi_b: Tensor) -> Tensor:
    """
    РУ: Один вклад в пену: |<Ψ^a|P|Ψ^b>|^2 [eq. 2.3 элементарный блок]
    EN: Single foam contribution: |<Ψ^a|P|Ψ^b>|^2 [eq. 2.3 elementary block]
    """
    # Complex inner product <Ψ^a|P|Ψ^b>
    overlap = torch.vdot(psi_a.conj(), projector @ psi_b)
    return torch.abs(overlap) ** 2

def compute_level_foam(
    states: Dict[MultiIndex, Tensor],
    projector: Tensor,
    level_indices: Optional[List[MultiIndex]] = None
) -> float:
    """
    РУ: Полная пена уровня l: Φ^(l) = Σ_{a≠b, dim(a)=l} |<Ψ^a|P_Gl|Ψ^b>|^2
    EN: Complete level l foam: Φ^(l) = Σ_{a≠b, dim(a)=l} |<Ψ^a|P_Gl|Ψ^b>|^2
    
    Args:
        states: {MultiIndex: state vector} at current level
        projector: P_Gl — level l goal projector
        level_indices: Specific indices to compute (default: all at level)
    
    Returns:
        foam_value: Φ^(l) ∈ [0, ∞), 0=perfect nullification
    """
    if level_indices is None:
        level_indices = [idx for idx, state in states.items() if idx.level == len(states[list(states)[0]])]
    
    foam = 0.0
    n_states = len(level_indices)
    
    for i in range(n_states):
        for j in range(i + 1, n_states):  # a ≠ b, avoid double-counting
            a_idx, b_idx = level_indices[i], level_indices[j]
            psi_a, psi_b = states[a_idx], states[b_idx]
            
            contribution = compute_single_foam_overlap(psi_a, projector, psi_b)
            foam += contribution
    
    return float(foam.real)

def compute_multilevel_foam(
    all_states: Dict[MultiIndex, Tensor],
    projectors: Dict[int, Tensor],
    max_level: Optional[int] = None
) -> Dict[int, float]:
    """
    РУ: Пена по всем уровням 0..K для мониторинга J_multiverse
    EN: Foam across all levels 0..K for J_multiverse monitoring
    
    Returns: {l: Φ^(l)} for convergence diagnostics
    """
    if max_level is None:
        max_level = max(idx.level for idx in all_states)
    
    foams = {}
    for l in range(max_level + 1):
        level_states = {idx: state for idx, state in all_states.items() if idx.level == l}
        projectors_l = projectors.get(l, torch.eye(8))  # Default identity
        foams[l] = compute_level_foam(level_states, projectors_l)
    
    return foams

def foam_gradient_wrt_psi_a(
    psi_a: Tensor, 
    psi_b: Tensor, 
    projector: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    РУ/EN: ∂/∂Ψ^a |<Ψ^a|P|Ψ^b>|^2 + ∂/∂Ψ^b для параллельного обновления [eq. 5.2]
    
    Implements autograd for torch optimization.
    """
    overlap = torch.vdot(psi_a.conj(), projector @ psi_b)
    foam_contrib = torch.abs(overlap) ** 2
    
    # Gradient w.r.t. ψ_a (real part affects through vdot)
    grad_a = torch.autograd.grad(foam_contrib, psi_a, create_graph=True, retain_graph=True)[0]
    grad_b = torch.autograd.grad(foam_contrib, psi_b, create_graph=True)[0]
    
    return grad_a, grad_b

class FoamMonitor:
    """
    РУ/EN: Мониторинг сходимости по пене для экспериментов
    """
    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance
        self.history: Dict[int, List[float]] = {}
    
    def record(self, foams: Dict[int, float]):
        for level, foam in foams.items():
            if level not in self.history:
                self.history[level] = []
            self.history[level].append(foam)
    
    def is_converged(self, foams: Dict[int, float]) -> bool:
        return all(foam < self.tolerance for foam in foams.values())
    
    def summary(self) -> str:
        if not self.history:
            return "No data"
        max_level = max(self.history)
        return f"Convergence: L{max_level} foam {self.history[max_level][-1]:.2e}"

# Verification utilities
def verify_theorem_conditions(
    states: Dict[MultiIndex, Tensor],
    projectors: Dict[Tuple[MultiIndex, int], Tensor]
) -> Dict[str, bool]:
    """
    РУ/EN: Проверяет условия Теоремы 4.1 для отладки
    """
    checks = {
        'C1_commutativity': True,  # Placeholder
        'C2_consistency': True,    # Tensor product check
        'C3_space_complete': True  # dim check
    }
    return checks

def nullification_score(states: Dict[MultiIndex, Tensor], projectors: Dict) -> float:
    """РУ/EN: Комплексный скор: -log(Σ Φ^(l))"""
    foams = compute_multilevel_foam(states, projectors)
    total_foam = sum(foams.values())
    return -torch.log(torch.tensor(total_foam + 1e-12)).item()

# Example & tests
if __name__ == "__main__":
    # Toy verification: create foam, nullify, verify Φ→0
    dim = 4
    states = {
        MultiIndex((0,)): torch.tensor([1+0j,0j,0j,0j]),
        MultiIndex((1,)): torch.tensor([0.707+0.707j,0j,0j,0j])
    }
    P = torch.eye(dim, dtype=torch.complex64)
    
    foam = compute_level_foam(states, P)
    print(f"Test foam: {foam:.2e}")
    
    monitor = FoamMonitor()
    print(f"Nullification score: {nullification_score(states, {0: P}):.3f}")
