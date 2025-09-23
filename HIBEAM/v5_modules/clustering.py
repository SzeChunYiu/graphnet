
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def robust_build_scaled_features(xyz: np.ndarray, t: np.ndarray | None, cfg: dict, use_time: bool) -> np.ndarray:
    sx, sy, sz = cfg.get("scale_xyz", (0.1, 0.1, 0.1))
    xyzc = xyz - np.median(xyz, axis=0, keepdims=True)
    X = np.column_stack([
        xyzc[:, 0] / (sx + 1e-12),
        xyzc[:, 1] / (sy + 1e-12),
        xyzc[:, 2] / (sz + 1e-12),
    ])
    if use_time and t is not None:
        t0 = t - np.median(t)
        q25, q75 = np.quantile(t0, [0.25, 0.75])
        iqr_t = max(q75 - q25, 1e-12)
        t_scaled = t0 / iqr_t
        amp_xyz = np.median(np.linalg.norm(X, axis=1)) + 1e-9
        amp_t = np.median(np.abs(t_scaled)) + 1e-9
        if amp_t / amp_xyz > 50.0:
            return X
        t_scaled = t_scaled / 5.0
        return np.column_stack([X, t_scaled])
    return X

def _adaptive_dbscan(Z: np.ndarray, base_eps: float, base_min_samples: int) -> np.ndarray:
    if len(Z) >= 5:
        k = min(4, len(Z) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        dists, _ = nbrs.kneighbors(Z)
        dk = dists[:, -1]
        eps_auto = float(np.median(dk) * 1.5)
        eps0 = max(float(base_eps), eps_auto)
    else:
        eps0 = float(base_eps)
    ms = max(2, int(base_min_samples))
    for mult in (1.0, 1.5, 2.0, 3.0, 4.0):
        labels = DBSCAN(eps=eps0 * mult, min_samples=max(2, ms - 1)).fit_predict(Z)
        if np.any(labels >= 0):
            return labels
    return labels

def _try_split_parallel(P: np.ndarray, t_sub: np.ndarray | None = None, min_size: int = 3, use_time_weight: float = 0.15, max_iter: int = 4, sep_thr: float = 1.7):
    N = P.shape[0]
    if N < 2 * min_size:
        return [np.arange(N)]
    C = P.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(P - C, full_matrices=False)
    u1, u2 = Vt[0], Vt[1]
    s1, s2 = S[0], S[1]
    if s2 / (s1 + 1e-12) < 0.08:
        return [np.arange(N)]
    proj2 = (P - C) @ u2
    if t_sub is not None:
        t0 = t_sub - np.median(t_sub)
        q25, q75 = np.quantile(t0, [0.25, 0.75])
        iqr = max(q75 - q25, 1e-12)
        t_feat = (t0 / iqr) * use_time_weight
        feat = np.column_stack([proj2, t_feat])
    else:
        feat = proj2.reshape(-1, 1)
    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gm.fit(feat)
        labels = gm.predict(feat)
        mu = gm.means_.ravel() if feat.shape[1] == 1 else gm.means_[:, 0]
        cov = gm.covariances_
        std = np.sqrt(cov.reshape(-1)) if feat.shape[1] == 1 else np.sqrt(cov[:, 0, 0])
    except Exception:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feat)
        labels = km.labels_
        mu = np.array([proj2[labels == 0].mean(), proj2[labels == 1].mean()])
        std = np.array([proj2[labels == 0].std() + 1e-9, proj2[labels == 1].std() + 1e-9])
    sep = abs(mu[0] - mu[1]) / (np.mean(std) + 1e-9)
    if sep < 1.2:
        return [np.arange(N)]
    labels_prev = None
    for _ in range(max_iter):
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        if len(idx0) < min_size or len(idx1) < min_size:
            return [np.arange(N)]
        def fit_line(Q: np.ndarray):
            Ck = Q.mean(axis=0, keepdims=True)
            _, _, Vt_k = np.linalg.svd(Q - Ck, full_matrices=False)
            dk = Vt_k[0]
            dk = dk / (np.linalg.norm(dk) + 1e-12)
            return Ck.squeeze(0), dk
        C0, d0 = fit_line(P[idx0]); C1, d1 = fit_line(P[idx1])
        def dist_to_line(pt: np.ndarray, Ck: np.ndarray, dk: np.ndarray) -> float:
            v = pt - Ck
            return np.linalg.norm(v - np.dot(v, dk) * dk)
        d0_all = np.array([dist_to_line(p, C0, d0) for p in P])
        d1_all = np.array([dist_to_line(p, C1, d1) for p in P])
        new_labels = (d1_all < d0_all).astype(int)
        if labels_prev is not None and np.array_equal(new_labels, labels_prev):
            break
        labels_prev = labels.copy()
        labels = new_labels
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    if len(idx0) < min_size or len(idx1) < min_size:
        return [np.arange(N)]
    mu0, mu1 = proj2[idx0].mean(), proj2[idx1].mean()
    std0, std1 = proj2[idx0].std() + 1e-9, proj2[idx1].std() + 1e-9
    sep_final = abs(mu0 - mu1) / ((std0 + std1) * 0.5)
    if sep_final < sep_thr:
        return [np.arange(N)]
    return [idx0, idx1]

def cluster_multiscale_union(xyz: np.ndarray, tvec: np.ndarray | None, ms_cfg: list[dict], use_time: bool) -> list[dict]:
    clusters = []
    for sid, cfg in enumerate(ms_cfg):
        Z = robust_build_scaled_features(xyz, tvec, cfg, use_time=use_time)
        labels = _adaptive_dbscan(Z, cfg.get("eps", 0.5), cfg.get("min_samples", 3))
        for tid in np.unique(labels):
            if tid < 0:
                continue
            idxs = np.where(labels == tid)[0]
            if len(idxs) < 2:
                continue
            clusters.append({"scale_id": sid, "idxs": idxs})
    # Jaccard 去重
    merged = []
    used = [False] * len(clusters)
    for i in range(len(clusters)):
        if used[i]: 
            continue
        A = set(clusters[i]["idxs"].tolist())
        group = [i]
        for j in range(i + 1, len(clusters)):
            if used[j]: 
                continue
            B = set(clusters[j]["idxs"].tolist())
            jacc = len(A & B) / max(1, len(A | B))
            if jacc > 0.6:
                used[j] = True
                group.append(j)
        used[i] = True
        best = max(group, key=lambda k: len(clusters[k]["idxs"]))
        merged.append(clusters[best])
    # 平行二分
    final_clusters = []
    for c in merged:
        idxs = c["idxs"]
        P_sub = xyz[idxs, :]
        T_sub = tvec[idxs] if (tvec is not None) else None
        splits = _try_split_parallel(P_sub, t_sub=T_sub, min_size=3, use_time_weight=0.15, max_iter=4, sep_thr=1.7)
        if len(splits) == 1:
            final_clusters.append(c)
        else:
            for sp in splits:
                final_clusters.append({"scale_id": c["scale_id"], "idxs": idxs[sp]})
    return final_clusters
