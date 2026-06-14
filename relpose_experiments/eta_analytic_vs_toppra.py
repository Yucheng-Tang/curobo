# Does an analytic time-optimal PTP surrogate match TOPPRA execution time?
#
# ETA-IK trains an MLP to approximate PTP execution time and uses it as an IK
# cost to resolve dual-arm redundancy toward fast motions. One of its two data
# generators is TOPPRA, which (per the paper) ignores collisions. For that
# target the execution time is a near-closed-form function of (q0, qT) under box
# velocity/acceleration limits. This script quantifies how well a cheap analytic
# surrogate tracks TOPPRA, to decide whether the MLP is needed for the
# collision-free part of the objective.
#
# Run in the v1 `neural-sdf` container (TOPPRA needs numpy<2):
#   docker exec neural-sdf bash -c "cd ~/ws/neural_sdf/curobo_v2 && python relpose_experiments/eta_analytic_vs_toppra.py"
import numpy as np

# Per-joint box velocity/acceleration limits per robot. NOTE: the surrogate-vs-
# TOPPRA ranking comparison is a PURE JOINT-SPACE PTP timing question — it depends
# ONLY on these limits, not on base placement. The "two FR3 parallel 0.92m apart"
# layout matters only for FK / self-collision (dataset gen + cost benchmark),
# not for this table. FR3 uses Panda kinematics with FR3 datasheet limits.
ROBOTS = {
    # 12-DoF homogeneous, ~uniform high accel -> velocity-bottlenecked.
    "dual_ur10e (12, homo)": dict(
        vmax=np.array([2.0944, 2.0944, 3.1416, 3.1416, 3.1416, 3.1416] * 2),
        amax=np.array([15.0] * 12),  # cuRobo dual_ur10e.yml max_acceleration
    ),
    # 13-DoF heterogeneous (UR5 6 + KUKA iiwa 7), low iiwa accel -> accel-bottlenecked.
    # ETA-IK paper TOPPRA-dataset limits.
    "robdekon UR5+iiwa (13, hetero)": dict(
        vmax=np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        amax=np.array([5.0, 5.0, 3.0, 2.0, 2.0, 2.0, 5.0, 5.0, 3.0, 2.0, 2.0, 2.0, 2.0]),
    ),
    # 14-DoF homogeneous (2x Franka Research 3, 7-DoF each). FR3 datasheet limits.
    "dual_fr3 (14, homo)": dict(
        vmax=np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26] * 2),
        amax=np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0] * 2),
    ),
}


def analytic_ptp_time(dq, vmax, amax):
    """Synchronized time-optimal PTP time, zero boundary velocity, box vel/acc.

    Per joint: trapezoidal profile if it reaches vmax, else triangular.
      d >= vmax^2/amax  ->  t = d/vmax + vmax/amax   (trapezoid)
      d <  vmax^2/amax  ->  t = 2*sqrt(d/amax)        (triangle)
    Synchronized multi-DoF time = max over joints (all joints finish together).
    """
    d = np.abs(dq)
    d_switch = vmax**2 / amax
    t_trap = d / vmax + vmax / amax
    t_tri = 2.0 * np.sqrt(d / amax)
    t_j = np.where(d >= d_switch, t_trap, t_tri)
    return t_j.max(axis=-1)


def softmax_time(dq, vmax, amax, beta=8.0):
    """Smooth (differentiable) surrogate of the max via log-sum-exp."""
    d = np.abs(dq)
    d_switch = vmax**2 / amax
    t_trap = d / vmax + vmax / amax
    t_tri = 2.0 * np.sqrt(d / amax)
    t_j = np.where(d >= d_switch, t_trap, t_tri)
    return np.log(np.sum(np.exp(beta * t_j), axis=-1)) / beta


def toppra_time(q0, qT, vmax, amax):
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint

    path = ta.SplineInterpolator([0.0, 1.0], np.stack([q0, qT]))
    pc_vel = constraint.JointVelocityConstraint(np.stack([-vmax, vmax], axis=1))
    pc_acc = constraint.JointAccelerationConstraint(np.stack([-amax, amax], axis=1))
    instance = algo.TOPPRA([pc_vel, pc_acc], path)
    jnt_traj = instance.compute_trajectory()
    if jnt_traj is None:
        return np.nan
    return float(jnt_traj.duration)


def run_robot(name, vmax, amax, N=400, seed=0):
    ndof = len(vmax)
    rng = np.random.default_rng(seed)
    t_analytic, t_soft, t_toppra, dq2 = [], [], [], []
    for _ in range(N):
        q0 = rng.uniform(-np.pi, np.pi, ndof)
        qT = rng.uniform(-np.pi, np.pi, ndof)
        tt = toppra_time(q0, qT, vmax, amax)
        if not np.isfinite(tt):
            continue
        t_toppra.append(tt)
        t_analytic.append(analytic_ptp_time(qT - q0, vmax, amax))
        t_soft.append(softmax_time(qT - q0, vmax, amax))
        dq2.append((qT - q0) ** 2)
    t_analytic = np.array(t_analytic)
    t_soft = np.array(t_soft)
    t_toppra = np.array(t_toppra)
    DQ2 = np.array(dq2)

    def spearman(est):
        ra = np.argsort(np.argsort(est))
        rb = np.argsort(np.argsort(t_toppra))
        return np.corrcoef(ra, rb)[0, 1]

    print(f"\n=== {name} | n={len(t_toppra)} valid PTP pairs ===")
    print(
        f"TOPPRA duration: mean {t_toppra.mean():.3f}s range [{t_toppra.min():.3f},{t_toppra.max():.3f}]"
    )
    rows = [
        ("L2 sum dist  Σd_j²", DQ2.sum(axis=-1), True),
        ("weighted L2  Σ(d_j/v_j)²", (DQ2 / vmax**2).sum(axis=-1), True),
        ("weighted L∞  max d_j/v_j (1st-order)", (np.sqrt(DQ2) / vmax).max(axis=-1), True),
        ("closed-form  max_j t_j (exact)", t_analytic, False),
        ("logsumexp(β=8) smooth closed-form", t_soft, False),
    ]
    for label, est, ranking_only in rows:
        rho = spearman(est)
        if ranking_only:
            print(f"  {label:38s} | Spearman {rho:.4f}")
        else:
            rel = np.abs(est - t_toppra) / t_toppra
            print(
                f"  {label:38s} | Spearman {rho:.4f} | rel-err mean {rel.mean()*100:5.2f}%"
                f" median {np.median(rel)*100:5.2f}% max {rel.max()*100:5.2f}%"
            )


def main():
    for name, lim in ROBOTS.items():
        run_robot(name, lim["vmax"], lim["amax"])


if __name__ == "__main__":
    main()
