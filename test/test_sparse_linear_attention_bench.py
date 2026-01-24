import torch
from ac4k_kernel.ops import SparseLinearAttention

attn = SparseLinearAttention(
    128,
    0.2,
    kernel_type="softmax",
    BLOCK_Q=64,
    BLOCK_KV=64,
).cuda()

B, H, L, D = 1, 40, 75600, 128
q = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')
k = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')
v = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')

# Create CUDA events for timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

warmup_runs = 5
repeat = 10

print("Warming up...")
for _ in range(warmup_runs):
    o = attn(q, k, v)

torch.cuda.synchronize()  # Make sure the warmup is done

print("Running benchmark...")
start_event.record()
for _ in range(repeat):
    o = attn(q, k, v)
end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded

elapsed_time = start_event.elapsed_time(end_event) / repeat

print(f"Average time per attention: {elapsed_time} ms")
