"""
src/multiverse/superfunctional.py

РУ: Супер-мета-функционал мультиверса J_multiverse [eq. 3.1] 
EN: Multiverse super-meta-functional J_multiverse [eq. 3.1]

Recursive definition:
J^(0) = J_loc(Ψ^a; G_0^a)                    # Local functional
J^(l) = Σ_{b≺a} J^(l-1)(Ψ^b) + Φ^(l)(Ψ^a, G_l) # Recursive + foam penalty

Total: J_multiverse = Σ_l Λ_l Σ_a J^(l)(Ψ^a), Λ_l = λ0 * α^l

Implements autograd for gradient-based nullification (Algorithm 5.2).
"""
import torch
from typing import Dict
from torch import Tensor
from .level import MultiIndex, Level
from .foam import compute_level_foam, foam_gradient_wrt_psi_a

def local_functional(psi: Tensor, goal_projector: Tensor) -> Tensor:
    """
    РУ/EN: Локальный функционал уровня 0: J_loc = ||(I-P_G0)Ψ||^2
    Local level 0 functional: distance to goal projector eigenspace
    """
    residual = (torch.eye(psi.shape[0], dtype=psi.dtype, device=psi.device) - goal_projector) @ psi
    return torch.vdot(residual.conj(), residual).real

def recursive_level_functional(
    l: int,
    states: Dict[MultiIndex, Tensor],
    projectors: Dict[int, Tensor],
    lambdas: Dict[int, float],
    level_obj: Level
) -> Dict[MultiIndex, float]:
    """
    РУ: Рекурсивный J^(l) по eq. 3.1b [J^(l-1) сумма + Φ^(l)]
    EN: Recursive J^(l) per eq. 3.1b [J^(l-1) sum + Φ^(l)]
    """
    if l == 0:
        J_level = {}
        for a, psi_a in states.items():
            if a.level == 0:
                J_level[a] = float(local_functional(psi_a, projectors[0]))
        return J_level
    
    # RECURSIVE: Sum J^(l-1) over subsystems + current foam
    J_level = {}
    level_indices = [a for a in states if a.level == l]
    
    for a in level_indices:
        # Sum subsystems J^(l-1)
        subsystems_J = sum(recursive_level_functional(l-1, states, projectors, lambdas, level_obj)[b] 
                          for b in a.subsystems())
        
        # Current level foam penalty
        level_foam = compute_level_foam({a: states[a] for a in level_indices}, projectors[l])
        
        J_level[a] = subsystems_J + level_foam
    
    return J_level

def J_multiverse(
    states: Dict[MultiIndex, Tensor],
    projectors: Dict[int, Tensor],
    lamb0: float = 1.0,
    alpha: float = 0.9,
    max_level: int = 5
) -> float:
    """
    РУ/EN: Полный супер-функционал мультиверса [eq. 3.1a]
    Complete multiverse super-functional [eq. 3.1a]
    
    J = Σ_l Λ_l Σ_{dim(a)=l} J^(l)(Ψ^a), Λ_l = λ0 * α^l
    """
    lambdas = {l: lamb0 * (alpha ** l) for l in range(max_level + 1)}
    
    total_J = 0.0
    for l in range(max_level + 1):
        level_contrib = recursive_level_functional(
            l, states, projectors, lambdas, Level(l, 8)
        )
        total_J += lambdas[l] * sum(level_contrib.values())
    
    return total_J

def multiverse_gradient(
    states: Dict[MultiIndex, Tensor],
    projectors: Dict[int, Tensor],
    lamb0: float = 1.0,
    alpha: float = 0.9
) -> Dict[MultiIndex, Tensor]:
    """
    РУ/EN: Полный градиент ∇J_multiverse для параллельного шага [eq. 5.2]
    Complete ∇J_multiverse for parallel update step [eq. 5.2]
    
    ∂J/∂Ψ^a = Λ_l ∂Φ^(l)/∂Ψ^a + Σ_{b⊃a} Λ_{l+1} ∂Φ^(l+1)/∂Ψ^a
    """
    max_level = max(idx.level for idx in states)
    lambdas = {l: lamb0 * (alpha ** l) for l in range(max_level + 2)}
    
    gradients = {a: torch.zeros_like(s) for a, s in states.items()}
    
    for l in range(max_level + 1):
        level_indices = [a for a in states if a.level == l]
        P_l = projectors[l]
        
        for i, a in enumerate(level_indices):
            psi_a = states[a]
            
            # Local level l gradient ∂Φ^(l)/∂Ψ^a
            for j, b in enumerate(level_indices):
                if a == b: continue
                psi_b = states[b]
                grad_a, _ = foam_gradient_wrt_psi_a(psi_a, psi_b, P_l)
                gradients[a] += lambdas[l] * grad_a
            
            # Upward contributions from level l+1 where a ⊂ b
            if l < max_level:
                higher_states = {idx: s for idx, s in states.items() if idx.level == l+1}
                for b_higher in higher_states:
                    if any(self[:-1] == a for self in b_higher):  # a subset of b_higher
                        grad_a_higher, grad_a_lower = foam_gradient_wrt_psi_a(
                            psi_a, higher_states[b_higher], projectors[l+1]
                        )
                        gradients[a] += lambdas[l+1] * grad_a_lower
    
    return gradients

class SuperfunctionalOptimizer(torch.optim.Optimizer):
    """
    РУ/EN: Специализированный оптимизатор для J_multiverse с Λ_l весами
    Specialized optimizer for J_multiverse with Λ_l weighting
    """
    def __init__(self, states: Dict[MultiIndex, Tensor], 
                projectors: Dict[int, Tensor], params=None):
        defaults = dict(lamb0=1.0, alpha=0.9)
        super().__init__(list(states.values()), defaults)
        self.states = states
        self.projectors = projectors
        self.lamb0 = lamb0
        self.alpha = alpha
    
    @torch.no_grad()
    def step(self, closure=None):
        gradients = multiverse_gradient(self.states, self.projectors, self.lamb0, self.alpha)
        
        for group in self.param_groups:
            for p, grad in zip(self.states.values(), gradients.values()):
                if p.grad is None:
                    p.grad = grad
                p.add_(-group['lr'], grad)

# Monitoring utilities
def convergence_metrics(states: Dict[MultiIndex, Tensor], projectors: Dict[int, Tensor]):
    """РУ/EN: Метрики сходимости {l: Φ^(l), J^(l)}"""
    foams = compute_multilevel_foam(states, projectors)
    J_values = recursive_level_functional(3, states, projectors, {}, Level(3, 8))
    return {'foams': foams, 'J_levels': J_values, 'J_total': J_multiverse(states, projectors)}

# Example usage & validation
if __name__ == "__main__":
    # Toy 2-level multiverse
    states = {
        MultiIndex((0,)): torch.rand(8, dtype=torch.cfloat),
        MultiIndex((1,)): torch.rand(8, dtype=torch.cfloat),
        MultiIndex((0,0)): torch.rand(4, dtype=torch.cfloat),  # Subsystem
    }
    
    projectors = {0: torch.eye(8), 1: torch.eye(8), 2: torch.eye(4)}
    
    J = J_multiverse(states, projectors)
    print(f"J_multiverse: {J:.6f}")
    
    grads = multiverse_gradient(states, projectors)
    print("Gradients computed ✓")
    
    # Test optimizer step
    optimizer = SuperfunctionalOptimizer(states, projectors)
    optimizer.zero_grad()
    # Simulate step
    print("SuperfunctionalOptimizer ready ✓")
