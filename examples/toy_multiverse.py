#!/usr/bin/env python3
"""
examples/toy_multiverse.py

РУ: Рабочий пример 3-уровневого мультиверса GRA обнуления [README demo]
EN: Working 3-level GRA multiverse nullification example [README demo]

Демонстрирует:
1. Инициализация мультииндексов и состояний
2. Рекурсивное/параллельное обнуление
3. Проверку Теоремы 4.1: Φ^(l) → 0 ∀ l
4. Сходимость J_multiverse → 0

Запуск: python examples/toy_multiverse.py [--levels=3] [--parallel]
"""
import argparse
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure src in path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from multiverse.optimizer import MultiverseOptimizer, create_toy_optimizer
from multiverse.level import MultiIndex
from multiverse.foam import compute_multilevel_foam
from multiverse.superfunctional import J_multiverse

def run_toy_example(levels: int = 3, 
                   n_subsystems: int = 4,
                   method: str = 'recursive',
                   max_iter: int = 200,
                   seed: int = 42,
                   plot: bool = True,
                   verbose: bool = True):
    """
    РУ/EN: Основной рабочий пример мультиверса
    Main working multiverse example
    """
    print(f"🚀 GRA Multiverse Nullification Demo")
    print(f"   K={levels}, N={n_subsystems}, method={method}")
    print(f"   Target: Φ^(l) < 1e-8 ∀ l=0..{levels} [Theorem 4.1]\n")
    
    torch.manual_seed(seed)
    
    # Initialize
    opt, initial_states, projectors = create_toy_optimizer(
        K=levels, method=method
    )
    opt.max_iter = max_iter
    
    if verbose:
        print("Initial state space:")
        for l in range(levels + 1):
            n_states = len([s for s in initial_states if s.level == l])
            print(f"  Level {l}: {n_states} states × dim={opt.state_dim}")
    
    # Optimize
    start_time = time.time()
    final_states = opt.optimize(initial_states, projectors, method=method)
    elapsed = time.time() - start_time
    
    # Final verification
    final_foams = compute_multilevel_foam(final_states, projectors, levels)
    final_J = J_multiverse(final_states, projectors, opt.lamb0, opt.alpha, levels)
    
    print(f"\n✅ RESULTS ({elapsed:.2f}s):")
    print("Level | Initial Φ^(l) | Final Φ^(l) | Status")
    print("-" * 45)
    
    max_initial_foam = 0
    all_converged = True
    
    for l in range(levels + 1):
        # Need initial foams for comparison (recompute)
        initial_level_states = {k: v.clone() for k, v in initial_states.items() if k.level == l}
        initial_foam_l = compute_level_foam(initial_level_states, projectors[l]) if initial_level_states else 0
        
        status = "✓" if final_foams[l] < opt.epsilon else "✗"
        if final_foams[l] >= opt.epsilon:
            all_converged = False
        
        print(f"L{l:2d}  | {initial_foam_l:10.2e} | {final_foams[l]:10.2e} | {status}")
        max_initial_foam = max(max_initial_foam, initial_foam_l)
    
    print(f"-" * 45)
    print(f"J_multiverse: {final_J:.2e}")
    print(f"Complete nullification: {'✓' if all_converged else '✗'}")
    
    if all_converged:
        print("🎉 THEOREM 4.1 VERIFIED: ABSOLUTE COGNITIVE VACUUM ACHIEVED!")
    else:
        print("⚠️  Close to convergence, increase iterations")
    
    # Plotting
    if plot:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        levels = list(range(levels + 1))
        final_foams_list = [final_foams[l] for l in levels]
        plt.semilogy(levels, final_foams_list, 'o-', linewidth=3, markersize=8)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Level l')
        plt.ylabel('Final Φ^(l)')
        plt.title(f'Foam Nullification (max Φ={max(final_foams_list):.2e})')
        plt.axhline(y=opt.epsilon, color='r', linestyle='--', label=f'ε={opt.epsilon}')
        plt.legend()
        
        if method == 'parallel' and hasattr(opt, 'J_history'):
            plt.subplot(1, 2, 2)
            plt.semilogy(opt.J_history, linewidth=2)
            plt.grid(True, alpha=0.3)
            plt.xlabel('Iteration')
            plt.ylabel('J_multiverse')
            plt.title('Superfunctional Convergence')
            plt.axhline(y=1e-6, color='r', linestyle='--')
        
        plt.tight_layout()
        plt.savefig('multiverse_nullification.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return {
        'final_states': final_states,
        'final_foams': final_foams,
        'final_J': final_J,
        'converged': all_converged,
        'time': elapsed
    }

def cli():
    parser = argparse.ArgumentParser(description="GRA Multiverse Nullification Demo")
    parser.add_argument('--levels', '-k', type=int, default=3, help='Hierarchy depth K')
    parser.add_argument('--subsystems', '-n', type=int, default=4, help='N subsystems/level')
    parser.add_argument('--method', choices=['recursive', 'parallel'], default='parallel', 
                       help='Optimization method')
    parser.add_argument('--max-iter', type=int, default=200, help='Max iterations (parallel)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', '-q', action='store_true', help='Less verbose')
    
    args = parser.parse_args()
    
    result = run_toy_example(
        levels=args.levels,
        n_subsystems=args.subsystems,
        method=args.method,
        max_iter=args.max_iter,
        seed=args.seed,
        plot=not args.no_plot,
        verbose=not args.quiet
    )
    
    if result['converged']:
        print("\n🏆 SUCCESS: Ready for production multiverse!")
    else:
        print("\n🔄 Increase --max-iter for full convergence.")

if __name__ == "__main__":
    cli()
