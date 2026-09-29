import torch

from loglens.model.line_encoder import LineEncoder, LineEncoderConfig, count_params


def test_default_param_count_in_budget():
    n = count_params(LineEncoder(LineEncoderConfig()))
    print("line encoder params:", n)
    assert 25e6 < n < 40e6  # 16k vocab: ~33.6M


def test_embedding_shape_and_padding_invariance():
    cfg = LineEncoderConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64,
                            embed_dim=16, max_len=16, dropout=0.0)
    m = LineEncoder(cfg).eval()
    a = torch.tensor([[1, 5, 6, 7, 2, 0, 0, 0]])
    b = torch.tensor([[1, 5, 6, 7, 2, 0, 0, 0, 0, 0]])
    with torch.no_grad():
        ea, eb = m(a), m(b)
    assert ea.shape == (1, 16)
    assert torch.allclose(ea, eb, atol=1e-5)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    cfg = LineEncoderConfig(vocab_size=50, d_model=64, n_layers=2, n_heads=4, d_ff=128,
                            embed_dim=16, max_len=12, dropout=0.0)
    m = LineEncoder(cfg)
    ids = torch.randint(5, 50, (16, 10))
    tgt = ids.clone()
    masked = ids.clone()
    masked[:, ::3] = 3
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    first = None
    for _ in range(150):
        logits = m.mlm_logits(m.hidden(masked))
        loss = torch.nn.functional.cross_entropy(logits[:, ::3].reshape(-1, 50), tgt[:, ::3].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < 0.3 * first
