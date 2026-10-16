"""CPU structural checks for the HDC decoder; no training or external data."""
import math
import torch

from model import CausalRoleBundler, Config, Decoder, PopulationEncoder


def assert_close(a, b, atol=1e-6, rtol=1e-6):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"max difference {float((a-b).abs().max())}")


def test_bundling_algebra_and_streaming():
    c = Config(channels=3, outputs=2, hidden=4, dimension=5,
               temporal_bins=3, levels=4, dropout=0.0)
    module = CausalRoleBundler(c).eval()
    q = torch.randn(2, 11, 2*c.dimension)
    full, full_history = module(q)
    # Independently spell out role binding and lag indexing.
    d, k = c.dimension, c.temporal_bins
    expected = torch.zeros_like(full)
    for t in range(q.shape[1]):
        for lag in range(k):
            if t < lag:
                continue
            qr, qi = q[:, t-lag, :d], q[:, t-lag, d:]
            rr, ri = module.role_real[lag], module.role_imag[lag]
            expected[:, t, :d] += qr*rr - qi*ri
            expected[:, t, d:] += qr*ri + qi*rr
    expected /= math.sqrt(k)
    assert_close(full, expected)
    assert full_history.shape == (2, k-1, 2*d)

    history = None
    pieces = []
    for start, stop in ((0, 1), (1, 6), (6, 7), (7, 11)):
        part, history = module(q[:, start:stop], history)
        pieces.append(part)
    assert_close(full, torch.cat(pieces, dim=1))
    assert_close(full_history, history)


def test_raw_lag_stack_algebra_and_streaming():
    from model import CausalRawStack
    c = Config(channels=3, outputs=2, hidden=4, dimension=5,
               temporal_bins=4, levels=4, dropout=0.0)
    module = CausalRawStack(c)
    x = torch.randn(2, 9, 3)
    full, full_history = module(x)
    expected = torch.zeros_like(full)
    for t in range(x.shape[1]):
        for lag in range(c.temporal_bins):
            if t >= lag:
                expected[:, t, lag*3:(lag+1)*3] = x[:, t-lag]
    assert_close(full, expected)
    history = None
    pieces = []
    for start, stop in ((0, 2), (2, 3), (3, 8), (8, 9)):
        part, history = module(x[:, start:stop], history)
        pieces.append(part)
    assert_close(full, torch.cat(pieces, dim=1))
    assert_close(full_history, history)


def test_codebook_seed_and_channel_role_ablation():
    x = torch.tensor([[[0.2, -0.4, 0.1], [0.8, 0.3, -0.5]]])
    c = Config(channels=3, outputs=2, hidden=4, dimension=5,
               temporal_bins=2, levels=4, dropout=0.0, family="hdc")
    a, b = PopulationEncoder(c), PopulationEncoder(c)
    different = PopulationEncoder(Config(**{**c.__dict__, "codebook_seed": c.codebook_seed + 1}))
    assert torch.equal(a.basis_matrix, b.basis_matrix)
    assert not torch.equal(a.basis_matrix, different.basis_matrix)
    weights = a.basis_weights(x)
    assert_close(weights.sum(dim=-1), torch.ones_like(x))
    manual = (weights[..., None] * a.basis_matrix.reshape(3, 4, -1)[None, None]).sum(dim=(2, 3))
    assert_close(a(x), manual)
    assert not torch.equal(a(x), a(torch.flip(x, dims=(-1,))))
    # Removing channel roles collapses channel identity in the bound codebook.
    no_channel_roles = PopulationEncoder(Config(**{**c.__dict__, "no_channel_roles": True}))
    assert not torch.equal(a.basis_matrix, no_channel_roles.basis_matrix)
    # Without channel roles, permuting channels leaves the superposition
    # unchanged; with independently bound roles it generally changes it.
    permuted = torch.flip(x, dims=(-1,))
    assert_close(no_channel_roles(x), no_channel_roles(permuted))
    assert not torch.allclose(a(x), a(permuted))
    # The real/imaginary representation retains state-dependent phase.
    encoded = a(x)
    assert encoded[..., :c.dimension].abs().sum() > 0
    assert encoded[..., c.dimension:].abs().sum() > 0
    assert not torch.equal(encoded[:, :1], encoded[:, 1:])


def test_all_families_are_causal_and_batch_isolated():
    torch.manual_seed(17)
    x = torch.randn(2, 13, 4)
    for family in ("hdc", "random", "mlp", "plain"):
        c = Config(family=family, channels=4, outputs=3, hidden=6,
                   dimension=7, temporal_bins=4, levels=5, dropout=0.0)
        model = Decoder(c).eval()
        y, _, _ = model(x)
        altered = x.clone()
        altered[:, 8:] += 100
        y2, _, _ = model(altered)
        assert_close(y[:, :8], y2[:, :8], atol=1e-6)

        # Chunked calls preserve both GRU and temporal code history.
        h = hist = None
        chunks = []
        for start, stop in ((0, 3), (3, 4), (4, 10), (10, 13)):
            p, h, hist = model(x[:, start:stop], h, hist)
            chunks.append(p)
        assert_close(y, torch.cat(chunks, dim=1), atol=2e-6)

        # A batch item is independent of the other item in the batch.
        p0, _, _ = model(x[:1])
        assert_close(y[:1], p0, atol=2e-6)


def test_role_ablation_preserves_temporal_support():
    c = Config(channels=2, outputs=1, hidden=3, dimension=4,
               temporal_bins=3, levels=3, family="hdc", no_roles=True)
    module = CausalRoleBundler(c)
    q = torch.randn(1, 6, 2*c.dimension)
    actual, _ = module(q)
    expected = torch.zeros_like(actual)
    for lag in range(c.temporal_bins):
        if lag < q.shape[1]:
            expected[:, lag:, :] += q[:, :q.shape[1]-lag, :]
    expected /= math.sqrt(c.temporal_bins)
    assert_close(actual, expected)


def test_family_shapes_and_parameter_accounting():
    counts = {}
    for family in ("hdc", "random", "mlp", "plain"):
        c = Config(family=family, channels=4, outputs=3, hidden=6,
                   dimension=7, temporal_bins=2, levels=4, dropout=0.0)
        model = Decoder(c)
        p, h, history = model(torch.randn(2, 5, 4))
        assert p.shape == (2, 5, 3)
        assert h.shape == (1, 2, 6)
        expected_width = 14 if family in ("hdc", "random") else 4
        assert history.shape == (2, 1, expected_width)
        assert model.parameter_count() == sum(v.numel() for v in model.parameters() if v.requires_grad)
        counts[family] = model.parameter_count()
    assert counts["hdc"] == counts["random"]
    # Prespecified study dimensions/windows have matched front-end projection
    # budgets within one percent; tiny toy dimensions are too coarse for the
    # integer MLP bottleneck-width rounding to satisfy that bound.
    for dimension in (256, 512):
        for bins in (1, 8):
            study = Config(dimension=dimension, temporal_bins=bins)
            hdc_count = Decoder(study).parameter_count()
            mlp_count = Decoder(Config(**{**study.__dict__, "family": "mlp"})).parameter_count()
            assert abs(mlp_count - hdc_count) / hdc_count < 0.01

    # Codebooks depend on their explicit seed, not on optimizer RNG state.
    c = Config(channels=4, outputs=3, hidden=6, dimension=7,
               temporal_bins=2, levels=4, family="hdc", codebook_seed=19)
    torch.manual_seed(1)
    first = PopulationEncoder(c)
    torch.manual_seed(999)
    second = PopulationEncoder(c)
    assert torch.equal(first.basis_matrix, second.basis_matrix)

    # The planned recurrence removal remains a valid causal ablation.
    nonrecurrent = Decoder(Config(family="hdc", channels=4, outputs=3, hidden=6,
                                 dimension=7, temporal_bins=2, levels=4,
                                 recurrence=False, dropout=0.0)).eval()
    p, h, history = nonrecurrent(torch.randn(1, 6, 4))
    assert p.shape == (1, 6, 3) and h is None and history.shape == (1, 1, 14)


def main():
    torch.set_num_threads(1)
    test_bundling_algebra_and_streaming()
    test_raw_lag_stack_algebra_and_streaming()
    test_codebook_seed_and_channel_role_ablation()
    test_all_families_are_causal_and_batch_isolated()
    test_role_ablation_preserves_temporal_support()
    test_family_shapes_and_parameter_accounting()
    print("All CPU HDC model checks passed.")


if __name__ == "__main__":
    main()
