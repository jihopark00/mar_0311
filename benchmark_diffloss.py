"""
SimpleTransformer vs SimpleMLPAdaLN 파라미터 및 추론 속도 비교
"""
import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, '/home/ljeadec31/opt/ssl2gen-top/mar_0311')

from models.flowloss import SimpleTransformer, SimpleMLPAdaLN


def count_parameters(model):
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def benchmark_inference(model, x, t, c, num_warmup=10, num_runs=100, device='cuda'):
    """추론 속도 벤치마크"""
    model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = model(x, t, c)

        # CUDA 동기화
        if device == 'cuda':
            torch.cuda.synchronize()

        # 실제 벤치마크
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(x, t, c)

        if device == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / num_runs
    return avg_time * 1000  # ms 단위로 반환


def format_params(num_params):
    """파라미터 수를 읽기 쉬운 형식으로 변환"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 80)

    # Config에서 가져온 설정값들
    in_channels = 768       # token_embed_dim = 3 * 16^2 = 768
    z_channels = 768        # d_cond (conditioning dimension from MAR)
    batch_size = 64         # buffer_size from config

    # 공통 테스트 입력
    x = torch.randn(batch_size, in_channels, device=device)
    t = torch.rand(batch_size, device=device)  # Flow matching uses [0, 1]
    c = torch.randn(batch_size, z_channels, device=device)

    # =========================================================================
    # Config 기반 비교 (marflow_base_256.yaml 설정)
    # =========================================================================
    print("\n[Config 기반 비교 - marflow_base_256.yaml 설정]")
    print("-" * 80)

    # SimpleTransformer (config에서 사용)
    transformer_config = SimpleTransformer(
        in_channels=in_channels,
        z_channels=z_channels,
        d_model=384,
        d_cond=768,
        depth=8,
        num_heads=6,
        patch_size=4,
        vae_embed_dim=3,
        mlp_ratio=4.0,
        grad_checkpointing=False
    ).to(device)

    # SimpleMLPAdaLN - 비슷한 파라미터 수를 위한 설정
    # Transformer와 비슷한 파라미터 수를 맞추기 위해 depth/width 조정
    mlp_adaln_config = SimpleMLPAdaLN(
        in_channels=in_channels,
        model_channels=1024,  # 파라미터 수 맞추기
        out_channels=in_channels,
        z_channels=z_channels,
        num_res_blocks=12,
        grad_checkpointing=False
    ).to(device)

    # 파라미터 수 계산
    transformer_params = count_parameters(transformer_config)
    mlp_params = count_parameters(mlp_adaln_config)

    print(f"\nSimpleTransformer (config 설정):")
    print(f"  - d_model: 384, d_cond: 768, depth: 8, num_heads: 6, patch_size: 4")
    print(f"  - Parameters: {format_params(transformer_params)} ({transformer_params:,})")

    print(f"\nSimpleMLPAdaLN (유사 파라미터):")
    print(f"  - model_channels: 1024, num_res_blocks: 12")
    print(f"  - Parameters: {format_params(mlp_params)} ({mlp_params:,})")

    # 추론 속도 벤치마크
    print("\n[추론 속도 벤치마크]")
    print("-" * 80)

    transformer_time = benchmark_inference(transformer_config, x, t, c, device=device)
    mlp_time = benchmark_inference(mlp_adaln_config, x, t, c, device=device)

    print(f"SimpleTransformer: {transformer_time:.4f} ms/inference")
    print(f"SimpleMLPAdaLN:    {mlp_time:.4f} ms/inference")
    print(f"속도 비율 (MLP/Transformer): {mlp_time/transformer_time:.2f}x")

    # =========================================================================
    # 다양한 설정에서 비교
    # =========================================================================
    print("\n" + "=" * 80)
    print("[다양한 설정에서의 비교]")
    print("=" * 80)

    configs = [
        # (name, d_model/model_channels, depth, num_heads for transformer)
        ("Small", 256, 4, 4),
        ("Base", 384, 8, 6),
        ("Large", 512, 12, 8),
    ]

    results = []

    for name, dim, depth, num_heads in configs:
        print(f"\n--- {name} Config (dim={dim}, depth={depth}) ---")

        # SimpleTransformer
        transformer = SimpleTransformer(
            in_channels=in_channels,
            z_channels=z_channels,
            d_model=dim,
            d_cond=z_channels,
            depth=depth,
            num_heads=num_heads,
            patch_size=4,
            vae_embed_dim=3,
            mlp_ratio=4.0,
            grad_checkpointing=False
        ).to(device)

        # SimpleMLPAdaLN - 같은 depth 사용
        mlp = SimpleMLPAdaLN(
            in_channels=in_channels,
            model_channels=dim,
            out_channels=in_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=False
        ).to(device)

        t_params = count_parameters(transformer)
        m_params = count_parameters(mlp)

        t_time = benchmark_inference(transformer, x, t, c, device=device)
        m_time = benchmark_inference(mlp, x, t, c, device=device)

        results.append({
            'name': name,
            'transformer_params': t_params,
            'mlp_params': m_params,
            'transformer_time': t_time,
            'mlp_time': m_time
        })

        print(f"SimpleTransformer: {format_params(t_params):>8} params, {t_time:.4f} ms")
        print(f"SimpleMLPAdaLN:    {format_params(m_params):>8} params, {m_time:.4f} ms")
        print(f"파라미터 비율 (Transformer/MLP): {t_params/m_params:.2f}x")
        print(f"속도 비율 (MLP/Transformer): {m_time/t_time:.2f}x")

        del transformer, mlp
        torch.cuda.empty_cache()

    # =========================================================================
    # 파라미터 수를 맞춘 공정 비교
    # =========================================================================
    print("\n" + "=" * 80)
    print("[파라미터 수를 맞춘 공정 비교]")
    print("=" * 80)

    # 약 10M 파라미터로 맞추기
    target_params = 10_000_000

    # SimpleTransformer: d_model=384, depth=8 -> ~9.5M
    transformer_fair = SimpleTransformer(
        in_channels=in_channels,
        z_channels=z_channels,
        d_model=32,
        d_cond=z_channels,
        depth=8,
        num_heads=1,
        patch_size=4,
        vae_embed_dim=3,
        mlp_ratio=4.0,
        grad_checkpointing=False,
        use_fused_attn=True,
    ).to(device)

    # SimpleMLPAdaLN: model_channels를 조정해서 비슷한 파라미터 수
    # ResBlock당 파라미터: ~4 * channels^2, 대략적으로 계산
    mlp_fair = SimpleMLPAdaLN(
        in_channels=in_channels,
        model_channels=1024,
        out_channels=in_channels,
        z_channels=z_channels,
        num_res_blocks=12,
        grad_checkpointing=False
    ).to(device)

    t_params_fair = count_parameters(transformer_fair)
    m_params_fair = count_parameters(mlp_fair)

    t_time_fair = benchmark_inference(transformer_fair, x, t, c, device=device)
    m_time_fair = benchmark_inference(mlp_fair, x, t, c, device=device)

    print(f"\n목표 파라미터 수: ~{format_params(target_params)}")
    print(f"\nSimpleTransformer (d_model=384, depth=8):")
    print(f"  Parameters: {format_params(t_params_fair)} ({t_params_fair:,})")
    print(f"  Inference:  {t_time_fair:.4f} ms")

    print(f"\nSimpleMLPAdaLN (model_channels=1024, depth=12):")
    print(f"  Parameters: {format_params(m_params_fair)} ({m_params_fair:,})")
    print(f"  Inference:  {m_time_fair:.4f} ms")

    print(f"\n파라미터 비율: {t_params_fair/m_params_fair:.2f}x (Transformer/MLP)")
    print(f"속도 비율: {m_time_fair/t_time_fair:.2f}x (MLP/Transformer)")

    # 효율성 계산 (파라미터당 처리 시간)
    t_efficiency = t_time_fair / t_params_fair * 1e6
    m_efficiency = m_time_fair / m_params_fair * 1e6
    print(f"\n효율성 (ms / M params):")
    print(f"  SimpleTransformer: {t_efficiency:.4f}")
    print(f"  SimpleMLPAdaLN:    {m_efficiency:.4f}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 80)
    print("[요약 테이블]")
    print("=" * 80)
    print(f"{'Model':<20} {'Params':>12} {'Time (ms)':>12} {'Throughput':>15}")
    print("-" * 60)
    print(f"{'SimpleTransformer':<20} {format_params(t_params_fair):>12} {t_time_fair:>12.4f} {1000/t_time_fair:>12.2f} inf/s")
    print(f"{'SimpleMLPAdaLN':<20} {format_params(m_params_fair):>12} {m_time_fair:>12.4f} {1000/m_time_fair:>12.2f} inf/s")

    print("\n" + "=" * 80)
    print("벤치마크 완료!")


if __name__ == "__main__":
    main()
