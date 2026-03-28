import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass
class Poisson1DResult:
    M: int
    h: float
    x_grid: np.ndarray
    f_vals: np.ndarray
    A: np.ndarray
    classical_solution: np.ndarray
    spectral_solution: np.ndarray
    eigenvalues: np.ndarray
    rhs_state: np.ndarray
    solution_state: np.ndarray
    amplitudes_in_eigenbasis: np.ndarray
    scaled_inverse_eigs: np.ndarray
    success_probability: float
    rel_state_error: float


def interior_grid(M: int) -> np.ndarray:
    """
    Return the interior grid points x_j = j/M, j=1,...,M-1.
    In the paper, h = 1/M and there are M-1 interior unknowns.
    """
    if M < 2:
        raise ValueError("M must be at least 2.")
    return np.arange(1, M) / M



def poisson_matrix_1d(M: int) -> np.ndarray:
    """
    Discrete 1D Poisson matrix A = h^{-2} * tridiag(-1, 2, -1)
    for Dirichlet boundary conditions on (0,1).
    """
    n = M - 1
    h = 1.0 / M
    A = 2.0 * np.eye(n)
    A += -1.0 * np.eye(n, k=1)
    A += -1.0 * np.eye(n, k=-1)
    return A / (h * h)



def dst_matrix(M: int) -> np.ndarray:
    """
    Orthornormal discrete sine transform matrix S of size (M-1)x(M-1),
    with entries
        S_{jk} = sqrt(2/M) sin(j k pi / M),   j,k = 1,...,M-1.
    This diagonalizes the 1D Dirichlet Poisson matrix.
    """
    n = M - 1
    S = np.zeros((n, n), dtype=float)
    factor = np.sqrt(2.0 / M)
    for j in range(1, M):
        for k in range(1, M):
            S[j - 1, k - 1] = factor * np.sin(np.pi * j * k / M)
    return S



def poisson_eigenvalues_1d(M: int) -> np.ndarray:
    """
    Eigenvalues of the discrete 1D Poisson matrix:
        lambda_j = 4 M^2 sin^2(j pi / (2M)), j=1,...,M-1.
    """
    j = np.arange(1, M, dtype=float)
    return 4.0 * (M ** 2) * np.sin(np.pi * j / (2.0 * M)) ** 2



def spectral_poisson_solve_1d(f_vals: np.ndarray, M: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve A v = f using the DST diagonalization A = S Lambda S^T.

    Returns:
      v         : solution in the physical/grid basis
      coeffs    : RHS coefficients beta_j in the eigenbasis
      lambdas   : eigenvalues lambda_j
    """
    S = dst_matrix(M)
    lambdas = poisson_eigenvalues_1d(M)
    coeffs = S.T @ f_vals
    v = S @ (coeffs / lambdas)
    return v, coeffs, lambdas



def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return vec / norm



def hhl_like_state_demo(f_vals: np.ndarray, M: int, C_d: float = 1.0) -> Dict[str, np.ndarray | float]:
    """
    A tiny state-vector demonstration of the paper's HHL-like logic.

    We do NOT build the full quantum circuit. Instead we emulate the logical steps:
      1) Prepare |f> from the RHS values.
      2) Expand |f> in the eigenbasis of A.
      3) Apply the inverse eigenvalue weighting beta_j -> beta_j * (C_d / lambda_j).
      4) Interpret the result as the post-selected solution state.

    For 1D, C_d = 1 is natural. The success probability corresponds to the norm of the
    ancilla-|1> branch before renormalization.
    """
    S = dst_matrix(M)
    lambdas = poisson_eigenvalues_1d(M)

    rhs_state = normalize(f_vals.astype(float))
    beta = S.T @ rhs_state
    weighted = beta * (C_d / lambdas)
    success_prob = float(np.sum(np.abs(weighted) ** 2))
    sol_state = normalize(S @ weighted)

    return {
        "rhs_state": rhs_state,
        "beta": beta,
        "lambdas": lambdas,
        "scaled_inverse_eigs": C_d / lambdas,
        "success_probability": success_prob,
        "solution_state": sol_state,
    }



def sample_rhs_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Some simple right-hand sides for toy experiments.
    The exact continuous solution is not the point here; we want easy-to-see modes.
    """
    if name == "constant":
        return lambda x: np.ones_like(x)
    if name == "sine_pi":
        return lambda x: np.sin(np.pi * x)
    if name == "sine_2pi":
        return lambda x: np.sin(2.0 * np.pi * x)
    if name == "poly_bump":
        return lambda x: x * (1.0 - x)
    raise ValueError(f"Unknown rhs function: {name}")



def run_case(M: int = 8, rhs_name: str = "sine_pi") -> Poisson1DResult:
    """
    Run a minimal reproducible 1D Poisson example.

    M controls the mesh size h = 1/M, so there are M-1 unknowns.
    This mirrors the paper's notation.
    """
    x = interior_grid(M)
    h = 1.0 / M
    f_fun = sample_rhs_function(rhs_name)
    f_vals = f_fun(x)

    A = poisson_matrix_1d(M)
    classical_sol = np.linalg.solve(A, f_vals)
    spectral_sol, coeffs, lambdas = spectral_poisson_solve_1d(f_vals, M)

    demo = hhl_like_state_demo(f_vals, M=M, C_d=1.0)

    # Compare the HHL-like post-selected solution state to the normalized classical solution.
    classical_state = normalize(classical_sol)
    phase_fix = np.vdot(classical_state, demo["solution_state"])
    if phase_fix != 0:
        aligned_state = demo["solution_state"] * np.exp(-1j * np.angle(phase_fix))
    else:
        aligned_state = demo["solution_state"]
    rel_state_error = float(np.linalg.norm(aligned_state - classical_state))

    return Poisson1DResult(
        M=M,
        h=h,
        x_grid=x,
        f_vals=f_vals,
        A=A,
        classical_solution=classical_sol,
        spectral_solution=spectral_sol,
        eigenvalues=lambdas,
        rhs_state=demo["rhs_state"],
        solution_state=demo["solution_state"],
        amplitudes_in_eigenbasis=demo["beta"],
        scaled_inverse_eigs=demo["scaled_inverse_eigs"],
        success_probability=float(demo["success_probability"]),
        rel_state_error=rel_state_error,
    )



def pretty_print_result(res: Poisson1DResult) -> None:
    print("=" * 72)
    print("Toy reproduction of the paper: 1D Poisson + spectral/HHL-like demo")
    print("=" * 72)
    print(f"M = {res.M}, h = {res.h:.6f}, unknowns = {res.M - 1}")
    print()
    print("Interior grid points x_j:")
    print(np.array2string(res.x_grid, precision=6, suppress_small=True))
    print()
    print("Discrete RHS values f_h:")
    print(np.array2string(res.f_vals, precision=6, suppress_small=True))
    print()
    print("Eigenvalues lambda_j of the 1D Poisson matrix:")
    print(np.array2string(res.eigenvalues, precision=6, suppress_small=True))
    print()
    print("RHS amplitudes beta_j in the sine/eigen basis:")
    print(np.array2string(res.amplitudes_in_eigenbasis, precision=6, suppress_small=True))
    print()
    print("Scaled inverse eigenvalues C_d/lambda_j (here C_d = 1):")
    print(np.array2string(res.scaled_inverse_eigs, precision=6, suppress_small=True))
    print()
    print("Classical solution from np.linalg.solve(A, f_h):")
    print(np.array2string(res.classical_solution, precision=8, suppress_small=True))
    print()
    print("Spectral solution using S Lambda^{-1} S^T f_h:")
    print(np.array2string(res.spectral_solution, precision=8, suppress_small=True))
    print()
    print("Normalized RHS state |f_h>:")
    print(np.array2string(res.rhs_state, precision=8, suppress_small=True))
    print()
    print("Normalized solution state after inverse-eigenvalue weighting:")
    print(np.array2string(res.solution_state, precision=8, suppress_small=True))
    print()
    print(f"Post-selection success probability (toy model) = {res.success_probability:.8e}")
    print(f"State error vs normalized classical solution = {res.rel_state_error:.8e}")
    print()
    print("Check || classical_solution - spectral_solution ||:")
    print(np.linalg.norm(res.classical_solution - res.spectral_solution))
    print("=" * 72)


if __name__ == "__main__":
    # Recommended first run: a very small case that mirrors the paper's 1D setting.
    result = run_case(M=8, rhs_name="sine_pi")
    pretty_print_result(result)

    # You can also try:
    # result = run_case(M=8, rhs_name="constant")
    # result = run_case(M=16, rhs_name="poly_bump")
    # pretty_print_result(result)
