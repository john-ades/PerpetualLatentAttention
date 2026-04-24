import torch
import torch.nn as nn
import math

class M6LatentAdapter(nn.Module):
    def __init__(self, kv_lora_rank, num_hidden_layers=32, S=64, top_k=8, gamma=0.95):
        super().__init__()
        self.S = S
        self.top_k = top_k
        self.gamma = gamma
        self.kv_lora_rank = kv_lora_rank
        
        # P lives entirely in TransMLA's compressed latent space!
        # Use register_buffer so ZeRO-3 doesn't partition this dynamic state tensor
        self.register_buffer("P", torch.randn(1, S, kv_lora_rank) * 0.02)
        
        # Address and Candidate Projections
        self.W_a = nn.Linear(kv_lora_rank, kv_lora_rank, bias=False)
        self.W_u = nn.Linear(kv_lora_rank, kv_lora_rank, bias=False)

        # NEW: M.6 Layer-Specific Read Projections
        # These learn to map the global memory P into the specific orthogonal PCA basis of each layer
        self.W_read = nn.ModuleList([
            nn.Linear(kv_lora_rank, kv_lora_rank, bias=False) 
            for _ in range(num_hidden_layers)
        ])
        
        # ✅ FIX: Initialize to small random weights so gradients can flow to W_a!
        # (Zeros completely severs the gradient flow because dL/dX = dL/dY @ W_read = 0)
        for proj in self.W_read:
            nn.init.normal_(proj.weight, std=0.02)
            
        # ✅ CRITICAL FIX: Safe Startup Attention Gate
        # Memory slots now bypass RMSNorm and are small, but text dot-products 
        # can frequently drop into negative values. -10.0 is not deep enough!
        # -100.0 guarantees the memory is mathematically invisible at startup.
        self.memory_gate = nn.Parameter(torch.tensor(-100.0))

    def write(self, k_pass_evicted: torch.Tensor, P_curr: torch.Tensor = None):
        """
        Triggered when tokens slide out of the context window.
        k_pass_evicted: (B, N_evicted, kv_lora_rank)
        """
        B = k_pass_evicted.shape[0]
        
        if P_curr is None:
            P_curr = self.P
            
        # FIX: Dynamically match the batch size of the incoming evicted tokens
        P_batch = P_curr.expand(B, -1, -1)
        
        # 1. Compress the evicted chunk into a semantic trajectory
        z_bar = k_pass_evicted.mean(dim=1) # (B, kv_lora_rank)
        
        # 2. Address the slots
        addr_query = self.W_a(z_bar)
        scores = torch.softmax(
            (addr_query.unsqueeze(1) @ P_batch.transpose(-2, -1)) / math.sqrt(self.kv_lora_rank),
            dim=-1
        ).squeeze(1) # (B, S)
        
        # 3. Top-k Routing (Sparse Masking)
        _, topk_idx = scores.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(scores).scatter_(1, topk_idx, 1.0)
        
        # ✅ CRITICAL FIX: Straight-Through Estimator (STE)
        # Allows gradients to bypass the non-differentiable scatter_ and train W_a
        mask = mask.detach() - scores.detach() + scores
        
        # 4. Candidate Generation
        u_t = self.W_u(z_bar)
        mask_3d = mask.unsqueeze(-1)
        u_3d = u_t.unsqueeze(1).expand_as(P_batch)
        
        # 5. Gated Overwrite (Returns updated tensor)
        P_new = (1 - mask_3d) * P_batch + mask_3d * (self.gamma * P_batch + (1 - self.gamma) * u_3d)
        return P_new

    def read(self, P_state: torch.Tensor):
        """Projects the global memory bank into layer-specific PCA latent bases."""
        return [proj(P_state) for proj in self.W_read]

    def forward(self, P_state: torch.Tensor):
        """
        ZeRO-3 safe read path. Invoked via __call__ to trigger DeepSpeed's 
        parameter gathering hooks for raw parameters like memory_gate.
        """
        memory_latents = self.read(P_state)
        
        # ✅ CRITICAL FIX: .clone() creates a new computational graph node.
        # This prevents a use-after-free crash when ZeRO-3 partitions the 
        # parameter back to size (0,) immediately after this hook exits!
        return memory_latents, self.memory_gate.clone()
