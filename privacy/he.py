"""Homomorphic-encrypted aggregation (optional, TenSEAL).

HE moves the aggregation inside the ciphertext domain: clients encrypt their
updates, the server adds ciphertexts, and only the decrypted final mean is
revealed. Adds Paillier-side costs but removes this security from the server.

Imports are deferred so the rest of the project runs without ``tenseal``.
"""

from __future__ import annotations

import numpy as np


def encrypt_and_aggregate(updates: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Encrypted server-side mean (demo path).

    With TenSEAL available this encrypts each client update with the server's
    public key, sums ciphertexts, and returns the decrypted mean. Without the
    library we fall back to plaintext averaging and log a warning.
    """
    try:
        import tenseal as ts  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env
        import logging
        logging.getLogger(__name__).warning(
            "tenseal not installed; falling back to plaintext average (%s)", exc
        )
        return _plaintext_mean(updates)

    # CKKS context: polynomial degree, coefficient modulus, scale
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096,
        coeff_mod_bit_sizes=[60, 40, 40, 60],  # pragmatic for toy updates
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    server_public = context  # demo: single party encrypts+decrypts

    n_layers = len(updates[0])
    ct_sum = [
        ts.ckks_vector(server_public, updates[0][lyr].reshape(-1).tolist())
        for lyr in range(n_layers)
    ]
    for upd in updates[1:]:
        for lyr in range(n_layers):
            ct_sum[lyr] += ts.ckks_vector(server_public, upd[lyr].reshape(-1).tolist())

    out = []
    for lyr in range(n_layers):
        dec = np.array(ct_sum[lyr].decrypt()) / len(updates)
        out.append(dec.reshape(updates[0][lyr].shape))
    return out


def _plaintext_mean(updates):
    n = len(updates)
    n_layers = len(updates[0])
    acc = [np.zeros_like(updates[0][i], dtype=np.float64) for i in range(n_layers)]
    for upd in updates:
        for i in range(n_layers):
            acc[i] += upd[i].astype(np.float64)
    return [a / n for a in acc]