# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file is a public-facing version of the Aether orchestration
#  framework. Core logic and integrated modules are redacted for safety.
# Aether v2.8 – (MPS-first; M4 optimized, stable core)
#   • SDPA: Tiled + Sliding Window + Global tokens (online softmax, MPS-safe)
#   • GQA/MQA: kv_heads for K/V head sharing
#   • INT8-base + LoRA (custom) / PEFT-LoRA / Hybrid
#   • CPU-AdamW(8bit-ish) optional
#   • LVI (light): teacher-based Z-bias + low-frequency cache
#   • Intention Contrastive Loss (teacher-positive/negative)
#   • ReLoRA cycle (periodic merge→re-apply)
#   • Streaming dataset / Curriculum / TorchScript trace
# ============================================================================

import os as _aos
import time as _atime


# ---------------------------
# 1) MPS watermark (safe gate)
# ---------------------------
def _aether_patch_mps_watermark():
    try:
        import torch

        if not hasattr(torch, "mps"):
            return
        ratio_s = _aos.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        if ratio_s is None:
            # Respect PyTorch default / user's runtime setting
            return
        try:
            r = float(ratio_s)
        except Exception:
            print(
                f"[MPS] Invalid PYTORCH_MPS_HIGH_WATERMARK_RATIO={ratio_s!r}; ignoring"
            )
            return
        if r <= 0.0:
            # "Unlimited" mode request: leave allocator untouched
            print("[MPS] watermark: unlimited requested (allocator untouched)")
            return
        try:
            torch.mps.set_per_process_memory_fraction(r)
            print(f"[MPS] watermark set to {r:.3f}")
        except Exception as e:
            print(f"[MPS] set_per_process_memory_fraction({r}) failed:", e)
    except Exception:
        # Non-fatal
        pass


if _aos.environ.get("AETHER_DISABLE_MPS_WM_PATCH", "0") != "1":
    _aether_patch_mps_watermark()


# -----------------------------
# 1a) MPS fast-path preferences
# -----------------------------
def _aether_mps_fastpath():
    try:
        import torch
    except Exception:
        return

    if not hasattr(torch, "mps"):
        return

    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return
    try:
        if not backend.is_available():
            return
    except Exception:
        return

    if _aos.environ.get("AETHER_MPS_ASYNC", "1") != "0":
        setter = getattr(torch.mps, "set_graphs_sync_enabled", None)
        if setter is not None:
            try:
                setter(False)
                print("[MPS] async graph execution enabled")
            except Exception as e:
                print("[MPS] async graph execution toggle failed:", e)


if _aos.environ.get("AETHER_DISABLE_MPS_FASTPATH", "0") != "1":
    _aether_mps_fastpath()


# ------------------------------
# 1b) MPS warmup (pump & preheat)
# ------------------------------
def _aether_mps_warmup():
    steps_s = _aos.environ.get("AETHER_MPS_WARMUP_STEPS")
    if steps_s is None:
        return

    try:
        steps = int(steps_s)
    except Exception:
        print(f"[MPS] Invalid AETHER_MPS_WARMUP_STEPS={steps_s!r}; ignoring")
        return

    if steps <= 0:
        return

    size_s = _aos.environ.get("AETHER_MPS_WARMUP_SIZE", "2048")
    try:
        size = int(size_s)
    except Exception:
        print(f"[MPS] Invalid AETHER_MPS_WARMUP_SIZE={size_s!r}; using 2048")
        size = 2048

    dtype_name = _aos.environ.get("AETHER_MPS_WARMUP_DTYPE", "float16").lower()

    try:
        import torch
    except Exception:
        return

    if not hasattr(torch, "mps"):
        return

    backend = getattr(torch, "backends", None)
    if backend is not None and hasattr(torch.backends, "mps"):
        try:
            if not torch.backends.mps.is_available():
                return
        except Exception:
            return

    dtype_map = {
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": getattr(torch, "bfloat16", torch.float16),
    }
    dtype = dtype_map.get(dtype_name)
    if dtype is None:
        print(f"[MPS] Unknown dtype {dtype_name!r}; defaulting to float16")
        dtype = torch.float16

    sync = _aos.environ.get("AETHER_MPS_WARMUP_SYNC", "1") != "0"
    attn_enable = _aos.environ.get("AETHER_MPS_WARMUP_ATTENTION", "1") != "0"
    attn_steps_env = _aos.environ.get("AETHER_MPS_WARMUP_ATTENTION_STEPS")
    attn_heads = max(1, int(_aos.environ.get("AETHER_MPS_WARMUP_HEADS", "16")))
    attn_seq = max(1, int(_aos.environ.get("AETHER_MPS_WARMUP_SEQ", str(min(size, 1024)))))
    attn_dim = max(8, int(_aos.environ.get("AETHER_MPS_WARMUP_HEAD_DIM", "128")))
    try:
        attn_steps_default = max(1, min(steps, 4))
    except Exception:
        attn_steps_default = 1
    attn_steps = attn_steps_default
    if attn_steps_env:
        try:
            attn_steps = max(1, int(attn_steps_env))
        except Exception:
            print(
                f"[MPS] Invalid AETHER_MPS_WARMUP_ATTENTION_STEPS={attn_steps_env!r}; using {attn_steps_default}"
            )
            attn_steps = attn_steps_default

    try:
        device = torch.device("mps")
        with torch.no_grad():
            x = torch.randn((size, size), device=device, dtype=dtype)
            y = torch.randn((size, size), device=device, dtype=dtype)
            matmul_start = _atime.perf_counter()
            for _ in range(steps):
                z = torch.matmul(x, y)
                x, y = y, z
            if sync and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            matmul_elapsed = _atime.perf_counter() - matmul_start

            attn_elapsed = 0.0
            attn_tok_per_sec = 0.0
            if attn_enable:
                try:
                    import torch.nn.functional as F

                    q = torch.randn(
                        (1, attn_heads, attn_seq, attn_dim),
                        device=device,
                        dtype=dtype,
                    )
                    k = torch.randn_like(q)
                    v = torch.randn_like(q)
                    attn_start = _atime.perf_counter()
                    for _ in range(attn_steps):
                        F.scaled_dot_product_attention(
                            q, k, v, is_causal=True, dropout_p=0.0
                        )
                    if sync and hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    attn_elapsed = _atime.perf_counter() - attn_start
                    approx_tokens = float(attn_steps * attn_seq * attn_heads)
                    attn_tok_per_sec = approx_tokens / max(attn_elapsed, 1e-6)
                except Exception as e:
                    print(f"[MPS] attention warmup skipped: {e}")

        total_elapsed = matmul_elapsed + attn_elapsed
        matmul_flop = 2.0 * float(size) * float(size) * float(size) * float(steps)
        matmul_gflops = matmul_flop / max(matmul_elapsed, 1e-9) / 1e9
        telemetry = {
            "device": "mps",
            "dtype": dtype_name,
            "matmul": {
                "size": size,
                "steps": steps,
                "elapsed_s": matmul_elapsed,
                "total_flop": matmul_flop,
                "gflops_per_s": matmul_gflops,
            },
        }
        msg = (
            f"[MPS] warmup pumped {steps}x matmul (size={size}, dtype={dtype_name}) "
            f"in {matmul_elapsed:.3f}s (~{matmul_gflops:,.1f} GFLOP/s)"
        )
        if attn_enable and attn_elapsed > 0.0:
            telemetry["attention"] = {
                "steps": attn_steps,
                "heads": attn_heads,
                "seq": attn_seq,
                "head_dim": attn_dim,
                "elapsed_s": attn_elapsed,
                "tokens_per_s": attn_tok_per_sec,
            }
            msg += (
                f"; attention warmup {attn_steps}x (seq={attn_seq}, heads={attn_heads}, "
                f"dim={attn_dim}) in {attn_elapsed:.3f}s (~{attn_tok_per_sec:,.0f} tok/s)"
            )
        msg += f" [total {total_elapsed:.3f}s]"
        print(msg)
        export_path = _aos.environ.get("AETHER_MPS_WARMUP_EXPORT")
        if export_path:
            try:
                import json as _json

                with open(export_path, "w", encoding="utf-8") as _fh:
                    _json.dump(telemetry, _fh, indent=2, sort_keys=True)
                print(f"[MPS] warmup telemetry exported to {export_path!r}")
            except Exception as e:
                print(
                    f"[MPS] failed to export telemetry to {export_path!r}: {e}"
                )
    except Exception as e:
        print(f"[MPS] warmup failed: {e}")


if _aos.environ.get("AETHER_DISABLE_MPS_WARMUP", "0") != "1":
    _aether_mps_warmup()


# ---------------------------------------
# 2) Backward guard (retain_graph retry)
# ---------------------------------------
def _aether_backward_guard():
    try:
        import torch
    except Exception:
        return

    _orig_autograd_backward = torch.autograd.backward
    _orig_tensor_backward = torch.Tensor.backward

    def _retry_with_retain_autograd(
        tensors, grad_tensors=None, retain_graph=None, create_graph=False, inputs=None
    ):
        try:
            return _orig_autograd_backward(
                tensors,
                grad_tensors=grad_tensors,
                retain_graph=retain_graph,
                create_graph=create_graph,
                inputs=inputs,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            # Typical messages: "Trying to backward through the graph a second time"
            if ("second backward" in msg or "retain_graph" in msg) and not retain_graph:
                return _orig_autograd_backward(
                    tensors,
                    grad_tensors=grad_tensors,
                    retain_graph=True,
                    create_graph=create_graph,
                    inputs=inputs,
                )
            raise

    def _retry_with_retain_tensor(
        self, gradient=None, retain_graph=None, create_graph=False, inputs=None
    ):
        try:
            return _orig_tensor_backward(
                self,
                gradient=gradient,
                retain_graph=retain_graph,
                create_graph=create_graph,
                inputs=inputs,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if ("second backward" in msg or "retain_graph" in msg) and not retain_graph:
                return _orig_tensor_backward(
                    self,
                    gradient=gradient,
                    retain_graph=True,
                    create_graph=create_graph,
                    inputs=inputs,
                )
            raise

    torch.autograd.backward = _retry_with_retain_autograd
    torch.Tensor.backward = _retry_with_retain_tensor
    print("[GUARD] backward guard enabled")


if _aos.environ.get("AETHER_BACKWARD_GUARD", "0") == "1":
    _aether_backward_guard()

# ---------------------------------------------------------
# 3) Light Paged-Attention wrapper for inference-time SDPA
# ---------------------------------------------------------
_AETHER_SDPA_ORIG = None
_AETHER_SDPA_WINDOW = int(_aos.environ.get("AETHER_SDPA_WINDOW", "0"))


def enable_tiled_sdpa(
    tiled_q: int = 0,
    tiled_k: int = 0,
    compute_in_fp32: bool = True,
    window_size: int = None,
):
    """
    Enable a light wrapper around torch.nn.functional.scaled_dot_product_attention.

    If window_size is provided (or env AETHER_SDPA_WINDOW>0), and Q len == 1 (typical during generation),
    we restrict K/V to the last `window_size` tokens (paged/streamed attention).

    tiled_q/tiled_k are accepted for API compatibility (no-ops here).
    """
    global _AETHER_SDPA_ORIG, _AETHER_SDPA_WINDOW
    try:
        import torch.nn.functional as F
    except Exception:
        return

    if window_size is not None:
        _AETHER_SDPA_WINDOW = int(window_size)

    if _AETHER_SDPA_ORIG is None:
        _AETHER_SDPA_ORIG = F.scaled_dot_product_attention

    def _sdpa_wrapper(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        w = _AETHER_SDPA_WINDOW
        # If generating (q_len==1) and window enabled, crop tail window
        if w and q.size(-2) == 1 and k.size(-2) > w:
            k = k[..., -w:, :]
            v = v[..., -w:, :]
            if attn_mask is not None and getattr(attn_mask, "dim", lambda: 0)() >= 2:
                attn_mask = attn_mask[..., -w:]
        if compute_in_fp32:
            qf, kf, vf = q.float(), k.float(), v.float()
            out = _AETHER_SDPA_ORIG(
                qf,
                kf,
                vf,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
            return out.to(q.dtype)
        else:
            return _AETHER_SDPA_ORIG(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

    F.scaled_dot_product_attention = _sdpa_wrapper
    print(
        f"[SDPA] wrapper enabled (window={_AETHER_SDPA_WINDOW}, fp32={compute_in_fp32})"
    )
    
def disable_tiled_sdpa():
    """Restore the original SDPA implementation."""
    global _AETHER_SDPA_ORIG
    try:
        import torch.nn.functional as F

        if _AETHER_SDPA_ORIG is not None:
            F.scaled_dot_product_attention = _AETHER_SDPA_ORIG
            print("[SDPA] wrapper disabled")
    finally:
        _AETHER_SDPA_ORIG = None

import os
import sys
import time
import math
import json
import glob
import threading
import random
import types
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union, Deque
from collections import deque

try:
    import numpy as _np
    _AETHER_NUMPY_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _np = None
    _AETHER_NUMPY_AVAILABLE = False
    print(
        "[Aether] numpy is not available. Optional analytics and AI controllers are disabled.",
        file=sys.stderr,
    )


def _require_numpy(feature: str = "this operation"):
    if _np is None:
        raise ModuleNotFoundError(
            f"numpy is required for {feature}. Install it or disable the corresponding feature."
        )
    return _np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "PyTorch is required to run Aether. Install it via `pip install torch`."
    ) from exc

from torch.amp import GradScaler

try:  # SentencePiece is optional; tokenizer gracefully degrades without it
    import sentencepiece as _sentencepiece
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _sentencepiece = None

try:  # TikToken-style encodings are optional as well
    import tiktoken as _tiktoken
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _tiktoken = None


def _aether_detect_mps() -> bool:
    try:
        backend = getattr(torch, "backends", None)
        if backend is None:
            return False
        mps = getattr(torch.backends, "mps", None)
        if mps is None:
            return False
        return bool(mps.is_available())
    except Exception:
        return False


_AETHER_MPS_AVAILABLE = _aether_detect_mps()
_AETHER_DEFAULT_LORA_R = int(
    _aos.environ.get(
        "AETHER_LORA_R",
        "160" if _AETHER_MPS_AVAILABLE else "16",
    )
)


class FlashAttentionRuntime:
    def __init__(self):
        self._eps = 1e-6
        self._fallbacks = 0
        self._compiled_impl = None
        self._compile_failed = False
        self._allow_compile = _aos.environ.get("AETHER_FLASH_COMPILE", "0") == "1"
        self._compile_backend = _aos.environ.get(
            "AETHER_FLASH_COMPILE_BACKEND", "inductor"
        )

    def _block_sizes(self):
        bq = max(1, int(_aos.environ.get("AETHER_FLASH_BLOCK_Q", "128")))
        bk = max(1, int(_aos.environ.get("AETHER_FLASH_BLOCK_K", "256")))
        return bq, bk

    def should_use(self, q: torch.Tensor) -> bool:
        if _aos.environ.get("AETHER_FLASH_DISABLE", "0") == "1":
            return False
        if not _AETHER_MPS_AVAILABLE:
            return False
        if q.device.type != "mps":
            return False
        return True

    def _slice_mask(self, mask: torch.Tensor, i0: int, i1: int, j0: int, j1: int):
        if mask is None:
            return None
        if mask.dim() == 4:
            return mask[:, :, i0:i1, j0:j1]
        if mask.dim() == 3:
            return mask[:, i0:i1, j0:j1].unsqueeze(1)
        if mask.dim() == 2:
            blk = mask[:, j0:j1]
            return blk.view(blk.shape[0], 1, 1, blk.shape[1]).expand(
                -1, 1, i1 - i0, -1
            )
        return None

    def _pad_mask_slice(
        self,
        pad_mask: Optional[torch.Tensor],
        i0: int,
        i1: int,
        j0: int,
        j1: int,
        device: torch.device,
    ):
        if pad_mask is None:
            return None
        if pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.to(torch.bool)
        key_valid = pad_mask[:, j0:j1]
        if key_valid.numel() == 0:
            return None
        key_block = (~key_valid).view(key_valid.shape[0], 1, 1, key_valid.shape[1])
        return key_block.expand(-1, 1, i1 - i0, -1).to(device)

    def _call_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        training: bool = False,
        scale_override: Optional[float] = None,
    ) -> torch.Tensor:
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        block_q, block_k = self._block_sizes()
        out = torch.empty((B, H, Tq, D), dtype=q.dtype, device=q.device)
        scale = (
            float(scale_override)
            if scale_override is not None
            else 1.0 / math.sqrt(max(1, D))
        )
        scale_tensor = torch.tensor(
            scale, dtype=torch.float32, device=q.device
        )
        eps = float(_aos.environ.get("AETHER_FLASH_EPS", self._eps))
        k_t = k.transpose(-1, -2).contiguous()
        q_positions = (
            torch.arange(Tq, device=q.device, dtype=torch.int32)
            if is_causal
            else None
        )
        k_positions = (
            torch.arange(Tk, device=q.device, dtype=torch.int32)
            if is_causal
            else None
        )
        pad_mask_bool = None
        if pad_mask is not None:
            pad_mask_bool = pad_mask.to(torch.bool)
        attn_mask_local = attn_mask
        for i0 in range(0, Tq, block_q):
            i1 = min(i0 + block_q, Tq)
            q_blk = q[:, :, i0:i1, :]
            blk_len = q_blk.shape[2]
            q_pos = None
            if is_causal and q_positions is not None:
                q_pos = q_positions[i0:i1].to(q.device)
            m_i = torch.full(
                (B, H, blk_len),
                -1e9,
                dtype=torch.float32,
                device=q.device,
            )
            l_i = torch.zeros((B, H, blk_len), dtype=torch.float32, device=q.device)
            o_i = torch.zeros((B, H, blk_len, D), dtype=q.dtype, device=q.device)
            for j0 in range(0, Tk, block_k):
                j1 = min(j0 + block_k, Tk)
                k_blk = k_t[:, :, :, j0:j1]
                scores = torch.matmul(q_blk, k_blk).to(torch.float32)
                scores.mul_(scale_tensor)
                mask_block = None
                if attn_mask_local is not None:
                    mask_block = self._slice_mask(attn_mask_local, i0, i1, j0, j1)
                if mask_block is None:
                    mask_block = self._pad_mask_slice(
                        pad_mask_bool, i0, i1, j0, j1, scores.device
                    )
                if mask_block is not None:
                    if mask_block.dtype == torch.bool:
                        scores = scores.masked_fill(mask_block, float("-inf"))
                    else:
                        scores = scores + mask_block.to(scores.dtype)
                if is_causal:
                    if q_pos is None:
                        q_pos = q_positions[i0:i1].to(q.device)
                    k_pos = k_positions[j0:j1].to(q.device)
                    causal = k_pos.view(1, 1, 1, -1) > q_pos.view(1, 1, -1, 1)
                    scores = scores.masked_fill(causal, float("-inf"))
                block_max = torch.max(scores, dim=-1).values
                block_max = torch.where(
                    torch.isfinite(block_max),
                    block_max,
                    torch.full_like(block_max, -1e9),
                )
                m_new = torch.maximum(m_i, block_max)
                scores = scores - m_new.unsqueeze(-1)
                scores = torch.where(
                    torch.isfinite(scores),
                    scores,
                    torch.full_like(scores, -1e9),
                )
                p = torch.exp(scores)
                if training and dropout_p > 0.0:
                    keep = torch.rand_like(p)
                    p = p * (keep > dropout_p) / max(1e-6, 1.0 - dropout_p)
                exp_m = torch.exp(m_i - m_new)
                l_i = exp_m * l_i + p.sum(dim=-1)
                v_blk = v[:, :, j0:j1, :]
                o_i = exp_m.unsqueeze(-1).to(q.dtype) * o_i + torch.matmul(
                    p.to(q.dtype), v_blk
                )
                m_i = m_new
            denom = l_i.clamp_min(eps).unsqueeze(-1).to(q.dtype)
            o_blk = o_i / denom
            if pad_mask_bool is not None:
                valid = pad_mask_bool[:, i0:i1]
                if valid.dtype != torch.bool:
                    valid = valid.to(torch.bool)
                o_blk = o_blk * valid.view(B, 1, i1 - i0, 1)
            out[:, :, i0:i1, :] = o_blk
        return out

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        pad_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        training: bool = False,
        scale_override: Optional[float] = None,
    ) -> torch.Tensor:
        impl = self._call_impl
        want_compile = (
            self._allow_compile
            and self.should_use(q)
            and self._compiled_impl is None
            and not self._compile_failed
            and hasattr(torch, "compile")
        )
        if want_compile:
            try:
                self._compiled_impl = torch.compile(
                    self._call_impl,
                    dynamic=True,
                    backend=self._compile_backend,
                )
                print("[MPS][FLASH] compiled attention kernel active")
            except Exception:
                self._compile_failed = True
                print(
                    "[MPS][FLASH] compile disabled after failure; using eager attention"
                )
        if self._compiled_impl is not None and self.should_use(q):
            impl = self._compiled_impl
        return impl(
            q,
            k,
            v,
            pad_mask=pad_mask,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            training=training,
            scale_override=scale_override,
        )


_FLASH_ATTENTION = FlashAttentionRuntime()


# === Aether injected: GaLore-like low-rank optimizer (matrix params only) ==============
class GaLoreAdamW(torch.optim.Optimizer):
    # AdamW in a projected (low-rank) space for 2D weight matrices (e.g., Linear).
    # For a weight W in R^{O×I}, keep two fixed orthonormal projectors:
    #   P_out in R^{O×r},  P_in in R^{I×r}
    # Maintain Adam states only for the r×r core (m_core, v_core).
    # Update rule (decoupled weight decay):
    #   g_core = P_out^T (grad W) P_in
    #   m_core <- beta1 m_core + (1-beta1) g_core
    #   v_core <- beta2 v_core + (1-beta2) g_core ⊙ g_core
    #   ΔW = P_out ( m_hat / (sqrt(v_hat)+eps) ) P_in^T
    #   W <- (1 - lr*wd)*W - lr*ΔW
    # Non-matrix params (bias/LayerNorm) fall back to standard AdamW states.
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        rank=64,
        seed=1337,
        device=None,
        dtype=None,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid lr")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rank = int(rank)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self._galore = {}
        self._rng = torch.Generator(device="cpu").manual_seed(self.seed)

    @torch.no_grad()
    def _init_param(self, p):
        pid = id(p)
        if pid in self._galore:
            return
        dev = p.device if self.device is None else torch.device(self.device)
        dt = p.dtype if self.dtype is None else self.dtype
        st = {"t": 0, "matrix": False}
        if p.ndim == 2 and p.numel() >= self.rank * self.rank:
            O, I = p.shape
            r = min(self.rank, O, I)
            A = torch.randn((O, r), generator=self._rng, device=dev, dtype=dt)
            B = torch.randn((I, r), generator=self._rng, device=dev, dtype=dt)
            Q_out, _ = torch.linalg.qr(A, mode="reduced")
            Q_in, _ = torch.linalg.qr(B, mode="reduced")
            st["P_out"] = Q_out
            st["P_in"] = Q_in
            st["m"] = torch.zeros((r, r), device=dev, dtype=dt)
            st["v"] = torch.zeros((r, r), device=dev, dtype=dt)
            st["matrix"] = True
        else:
            st["m"] = torch.zeros_like(p, device=dev, dtype=dt)
            st["v"] = torch.zeros_like(p, device=dev, dtype=dt)
        self._galore[pid] = st

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._init_param(p)
                st = self._galore[id(p)]
                st["t"] += 1

                if wd != 0 and p.ndim >= 1:
                    p.mul_(1 - lr * wd)

                g = p.grad
                if st["matrix"]:
                    P_out = st["P_out"]
                    P_in = st["P_in"]
                    g_core = P_out.T @ g @ P_in
                    st["m"].mul_(beta1).add_(g_core, alpha=(1 - beta1))
                    st["v"].mul_(beta2).addcmul_(g_core, g_core, value=(1 - beta2))
                    m_hat = st["m"] / (1 - beta1 ** st["t"])
                    v_hat = st["v"] / (1 - beta2 ** st["t"])
                    core = m_hat / (v_hat.sqrt() + eps)
                    p.addmm_(P_out, core @ P_in.T, alpha=-lr)
                else:
                    st["m"].mul_(beta1).add_(g, alpha=(1 - beta1))
                    st["v"].mul_(beta2).addcmul_(g, g, value=(1 - beta2))
                    m_hat = st["m"] / (1 - beta1 ** st["t"])
                    v_hat = st["v"] / (1 - beta2 ** st["t"])
                    p.addcdiv_(m_hat, (v_hat.sqrt() + eps), value=-lr)

        return loss


# === end GaLore-like optimizer ===============================================


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

# === ANI-AI (numpy micro-agent; optional, opt-in via env) =====================
_ANI_AI_AVAILABLE = _AETHER_NUMPY_AVAILABLE


class _AetherNumpyAIController:
    """
    Lightweight bandit-like controller that reacts to numeric hazards and stabilizes training.
    - No structure changes: installed from TrainerBase.__init__ (opt-in via env).
    - Uses numpy only. Keeps small moving windows and picks actions via UCB.
    - Actions (all guarded & reversible): loss-scale backoff, dynamic grad-clip, LR backoff,
      skip-step, safe softmax/log_softmax patch, optimizer state sanitization.
    """

    def __init__(self, trainer):
        self.tr = trainer
        self.enabled = bool(int(os.environ.get("AETHER_AI_ENABLE", "0")))
        self.ucb_c = float(os.environ.get("AETHER_AI_UCB_C", "1.25"))
        self.min_ls = float(
            os.environ.get(
                "AETHER_AI_MIN_LOSS_SCALE",
                os.environ.get("AETHER_ANI_MIN_LOSS_SCALE", "0.015625"),
            )
        )
        self.ls_backoff = float(
            os.environ.get(
                "AETHER_AI_LS_BACKOFF",
                os.environ.get("AETHER_ANI_SCALE_BACKOFF", "0.5"),
            )
        )
        self.grad_clip_max = float(os.environ.get("AETHER_AI_MAX_CLIP", "1.0"))
        self.allow_lr_backoff = bool(
            int(os.environ.get("AETHER_AI_LR_BACKOFF_ENABLE", "1"))
        )
        self.lr_backoff = float(os.environ.get("AETHER_AI_LR_BACKOFF", "0.5"))
        self.safe_softmax_default = bool(
            int(os.environ.get("AETHER_AI_SAFE_SOFTMAX", "1"))
        )
        self.sanitize_opt_default = bool(
            int(os.environ.get("AETHER_AI_SANITIZE_OPT", "1"))
        )
        self.hazard_patience = int(os.environ.get("AETHER_AI_PATIENCE", "1"))
        self.cooldown = int(
            os.environ.get(
                "AETHER_AI_COOLDOWN", os.environ.get("AETHER_ANI_COOLDOWN", "300")
            )
        )
        self.allow_skip = bool(
            int(
                os.environ.get(
                    "AETHER_AI_SKIP_ON_HAZARD",
                    os.environ.get("AETHER_ANI_SKIP_ON_HAZARD", "1"),
                )
            )
        )
        self.hist_len = int(os.environ.get("AETHER_AI_HIST", "256"))
        self.action_mask = os.environ.get(
            "AETHER_AI_ACTIONS", "none,ls_down,clip_up,lr_down,softmax_on,sanitize,skip"
        ).split(",")
        self.global_safe_softmax_on = False
        self.global_softmax_patched = False
        self.skip_next_step = False
        self.last_action = "none"
        self.t = 0
        self.window_loss = []
        self.window_finite = []
        self.window_grad = []
        self.arms = [
            "none",
            "ls_down",
            "clip_up",
            "lr_down",
            "softmax_on",
            "sanitize",
            "skip",
        ]
        self.q = {a: 0.0 for a in self.arms}
        self.n = {a: 0 for a in self.arms}
        # lossscale bridge (if ANI scaler exists, prefer it)
        self._ls_get = getattr(self.tr, "_ani_get_loss_scale", None)
        self._ls_set = getattr(self.tr, "_ani_set_loss_scale", None)
        if self._ls_get is None or self._ls_set is None:
            # fallback: keep a local soft-scale multiplier
            self._local_ls = float(os.environ.get("AETHER_ANI_LOSS_SCALE", "1.0"))

            def _get():
                return self._local_ls

            def _set(v):
                self.__dict__.update(_local_ls=float(v))

            self._ls_get, self._ls_set = _get, _set
        # grad clip bridge
        self._set_grad_clip = getattr(self.tr, "_ani_set_grad_clip", None)
        # LR bridge
        self._lr_base = None

        if self.enabled and self.safe_softmax_default:
            self._patch_softmax(True)
        if self.enabled and self.sanitize_opt_default:
            self._sanitize_optimizer_states_safe()

    # ---------- patches ----------
    def _patch_softmax(self, on: bool):
        if on and not self.global_softmax_patched:
            self._orig_softmax = torch.softmax
            self._orig_logsoftmax = F.log_softmax

            def _safe_softmax(x, dim=-1, dtype=None):
                if not torch.is_floating_point(x):
                    return self._orig_softmax(x, dim=dim, dtype=dtype)
                x32 = x.float()
                x32 = x32 - x32.amax(dim=dim, keepdim=True)
                x32 = x32.clamp(min=-50.0, max=50.0)  # strong but safe
                y = torch.exp(x32)
                return (y / (y.sum(dim=dim, keepdim=True) + 1e-12)).to(dtype or x.dtype)

            def _safe_logsoftmax(x, dim=-1, dtype=None):
                if not torch.is_floating_point(x):
                    return F.log_softmax(x, dim=dim, dtype=dtype)
                x32 = x.float()
                x32 = x32 - x32.amax(dim=dim, keepdim=True)
                x32 = x32.clamp(min=-50.0, max=50.0)
                logsumexp = torch.log(torch.exp(x32).sum(dim=dim, keepdim=True) + 1e-12)
                return (x32 - logsumexp).to(dtype or x.dtype)

            torch.softmax = _safe_softmax
            F.log_softmax = _safe_logsoftmax
            self.global_softmax_patched = True
            self.global_safe_softmax_on = True
        elif (not on) and self.global_softmax_patched:
            try:
                torch.softmax = self._orig_softmax
                F.log_softmax = self._orig_logsoftmax
            except Exception:
                pass
            self.global_softmax_patched = False
            self.global_safe_softmax_on = False

    def _sanitize_optimizer_states_safe(self):
        opt = getattr(self.tr, "optimizer", None) or getattr(self.tr, "opt", None)
        if opt is None:
            return
        for p in opt.state.values():
            for k, v in list(p.items()):
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    if not torch.isfinite(v).all():
                        p[k] = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)

    # ---------- observation ----------
    def observe_forward(self, loss_tensor, logits=None):
        self.t += 1
        try:
            l = float(loss_tensor.detach().float().clamp(min=-20.0, max=20.0))
        except Exception:
            l = float("inf")
        self.window_loss.append(l)
        if len(self.window_loss) > self.hist_len:
            self.window_loss = self.window_loss[-self.hist_len :]
        finite = 1.0 if math.isfinite(l) else 0.0
        self.window_finite.append(finite)
        if len(self.window_finite) > self.hist_len:
            self.window_finite = self.window_finite[-self.hist_len :]

    def observe_grads(self, model):
        try:
            tot = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if not g.dtype.is_floating_point:
                    continue
                v = float(torch.linalg.norm(g.float()).cpu())
                if math.isfinite(v):
                    tot += v
            self.window_grad.append(tot)
            if len(self.window_grad) > self.hist_len:
                self.window_grad = self.window_grad[-self.hist_len :]
        except Exception:
            pass

    # ---------- policy ----------
    def _ucb_pick(self):
        # restrict by mask
        arms = [a for a in self.arms if a in self.action_mask]
        if not arms:
            arms = ["none"]
        t = max(1, sum(self.n[a] for a in arms))
        best_a, best_val = arms[0], -1e9
        for a in arms:
            q = self.q[a]
            n = max(1, self.n[a])
            u = q + self.ucb_c * math.sqrt(math.log(t + 1.0) / n)
            if u > best_val:
                best_val, best_a = u, a
        return best_a

    def _reward(self):
        # +1 for recent stability, -1 if instability / NaN likely (0 in loss is suspicious too)
        if not self.window_loss:
            return 0.0
        recent = self.window_loss[-min(8, len(self.window_loss)) :]
        finite_recent = self.window_finite[-len(recent) :]
        if sum(finite_recent) < len(recent):
            return -1.0
        if any(abs(x) < 1e-12 for x in recent):
            return -0.5  # suspicious zeros
        # gentle reward for decrease
        if len(recent) >= 4:
            d = (sum(recent[-2:]) - sum(recent[:2])) / 2.0
            return 0.5 if d < 0 else 0.1
        return 0.2

    def decide_and_act(self, step_idx: int):
        if not self.enabled:
            return
        # hazard = too many non-finite in tail or weird zero loss
        hazard = False
        if (
            self.window_finite
            and sum(self.window_finite[-self.hazard_patience :]) < self.hazard_patience
        ):
            hazard = True
        if self.window_loss and any(
            abs(x) < 1e-12 for x in self.window_loss[-self.hazard_patience :]
        ):
            hazard = True
        if not hazard:  # small chance to relax patches
            if (step_idx % max(1, self.cooldown)) == 0:
                self._patch_softmax(self.safe_softmax_default)  # may stay on by default
            # update Q for previous action
            r = self._reward()
            if self.last_action in self.q:
                self.q[self.last_action] = 0.9 * self.q[self.last_action] + 0.1 * r
                self.n[self.last_action] += 1
            return
        # --- hazard path: pick an action ---
        a = self._ucb_pick()
        self.last_action = a
        # apply
        if a == "none":
            return
        if a == "ls_down":
            cur = float(self._ls_get())
            new = max(self.min_ls, cur * self.ls_backoff)
            self._ls_set(new)
        elif a == "clip_up":
            if self._set_grad_clip:
                self._set_grad_clip(self.grad_clip_max)
        elif a == "lr_down" and self.allow_lr_backoff:
            opt = getattr(self.tr, "optimizer", None) or getattr(self.tr, "opt", None)
            if opt is not None:
                for g in opt.param_groups:
                    base = g.get("_base_lr", g["lr"])
                    g["_base_lr"] = base
                    g["lr"] = max(1e-7, float(base) * self.lr_backoff)
        elif a == "softmax_on":
            self._patch_softmax(True)
        elif a == "sanitize":
            self._sanitize_optimizer_states_safe()
        elif a == "skip" and self.allow_skip:
            self.skip_next_step = True
        # update stats for arm
        self.n[a] += 1

    # called by loop to check skip
    def should_skip(self):
        if self.skip_next_step:
            self.skip_next_step = False
            return True
        return False

    # ---------- flags & planning ----------
    def set_flag(self, name: str, value: bool):
        if not hasattr(self, "_flags"):
            self._flags = {}
        self._flags[name] = bool(value)

    def get_flag(self, name: str, default=False):
        return bool(getattr(self, "_flags", {}).get(name, default))

    def plan_pre_forward(self):
        # Decide pre-forward toggles from recent stability (e.g., fp32 logits)
        plan = {}
        recent_instab = (
            len(self.window_finite) >= max(2, self.hazard_patience)
            and sum(self.window_finite[-self.hazard_patience :]) < self.hazard_patience
        )
        if recent_instab:
            plan["fp32_logits"] = True
        if bool(int(os.environ.get("AETHER_FP32_LOGITS", "0"))):
            plan["fp32_logits"] = True
        return plan

    def post_forward_assess(self, logits, subx=None, suby=None, pad_id=None):
        # Inspect logits to catch early saturation before backward/step.
        out = {"hazard": False}
        try:
            Lmax = float(logits.detach().float().abs().amax().cpu())
        except Exception:
            Lmax = 0.0
        if Lmax > 75.0:  # conservative threshold for saturation risk
            self.set_flag("fp32_logits", True)
            out["hazard"] = True
            if self.allow_skip:
                self.skip_next_step = True
        try:
            if suby is not None and pad_id is not None:
                mask = suby != pad_id
                if int(mask.sum().item()) <= 1:
                    out["hazard"] = True
                    if self.allow_skip:
                        self.skip_next_step = True
        except Exception:
            pass
        return out

    def sanitize_gradients(self, model):
        # Nan->num for grads; light clamp to avoid wild spikes.
        try:
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad
                if not g.dtype.is_floating_point:
                    continue
                g.data = torch.nan_to_num(g.data, nan=0.0, posinf=1e4, neginf=-1e4)
        except Exception:
            pass


_CHRONO_BOOTSTRAP_MODE = os.environ.get("AETHER_CHRONO_BOOTSTRAP", "async").lower()
if os.environ.get("AETHER_DISABLE_CHRONO", "0") == "1":
    chrono = None
elif _CHRONO_BOOTSTRAP_MODE not in {"async", "sync", "off"}:
    chrono = None
    if os.environ.get("AETHER_SILENCE_OPTIONALS", "0") != "1":
        print(
            "[Chrono] Invalid AETHER_CHRONO_BOOTSTRAP mode; telemetry disabled.",
            file=sys.stderr,
        )
else:
    try:
        import spiral_chronostasis_v6_3 as chrono  # type: ignore
    except Exception as _chrono_exc:
        chrono = None
        if os.environ.get("AETHER_SILENCE_OPTIONALS", "0") != "1":
            print(
                "[Chrono] spiral_chronostasis_v6_3 unavailable; skipping optional telemetry.",
                file=sys.stderr,
            )
    else:
        _chrono_bootstrap_once = threading.Event()

        def _chrono_bootstrap() -> None:
            if _chrono_bootstrap_once.is_set():
                return
            _chrono_bootstrap_once.set()
            try:
                chrono.install_defaults()  # ~/.spiral/chronostasis.json を生成
            except Exception as _chrono_exc_install:
                print(
                    "[Chrono] install_defaults failed:",
                    _chrono_exc_install,
                    file=sys.stderr,
                )
            try:
                print(chrono.stats())
            except Exception as _chrono_exc_stats:
                print(
                    "[Chrono] stats unavailable:",
                    _chrono_exc_stats,
                    file=sys.stderr,
                )

        if _CHRONO_BOOTSTRAP_MODE == "sync":
            _chrono_bootstrap()
        elif _CHRONO_BOOTSTRAP_MODE == "async":
            threading.Thread(
                target=_chrono_bootstrap,
                name="aether-chrono-bootstrap",
                daemon=True,
            ).start()
# === k-bridge (optional) ======================================================
try:
    from kbridge.k_autograd import khuber_loss
    from kbridge.metrics import roc_auc_binary, pr_auc_binary
    from kbridge.kmetrics import (
        ece_and_hist_k,
        ndcg_at_k_seq_k,
        ece_multi_groups_k,
        ece_and_hist_k_bins,
    )
    from kbridge.preproc import byte_normalize_utf8

    _KBRIDGE_AVAILABLE = True
except Exception:
    _KBRIDGE_AVAILABLE = False


# Track tokenizer layout so analytics can adapt to hybrid vocabularies
_AETHER_TOKENIZER_LAYOUT: Dict[str, Any] = {
    "byte_base": 3,
    "byte_count": 256,
    "sp_base": None,
    "sp_size": 0,
    "tt_base": None,
    "tt_size": 0,
}


# === Class grouping helpers (ByteTokenizer-aware; fallback-safe) =============
def _build_classmap(vocab_size: int, scheme: str = "byte-basic", json_path: str = ""):
    """
    Returns: (classmap_np[int32, shape=(vocab_size,)], group_names[list[str]])
    scheme: "byte-basic" | "byte-compact" | "custom-json"
    - ByteTokenizer hybrid: PAD=0, BOS=1, EOS=2, optional SentencePiece/TikToken
      ranges, and (if enabled) byte fallback tokens. Byte ranges are detected via
      _AETHER_TOKENIZER_LAYOUT.
    Groups (byte-basic):
      0=control(<32,127 except whitespace), 1=whitespace, 2=digit, 3=alpha,
      4=punct, 5=ascii-other, 6=non-ascii(>=128), 7=special(PAD/BOS/EOS/others)
    """
    np_mod = _require_numpy("class grouping helpers")

    names = [
        "control",
        "whitespace",
        "digit",
        "alpha",
        "punct",
        "asciiOther",
        "nonASCII",
        "special",
    ]
    m = np_mod.full((vocab_size,), 7, dtype=np_mod.int32)  # default= special/other
    # specials
    for t in [0, 1, 2]:
        if t < vocab_size:
            m[t] = 7
    if scheme == "custom-json" and json_path:
        try:
            import json
            import os

            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        k = int(k)
                        g = int(v)
                        if 0 <= k < vocab_size:
                            m[k] = g
                elif isinstance(obj, list):
                    arr = np_mod.asarray(obj, dtype=np_mod.int32).ravel()
                    m[: min(vocab_size, arr.size)] = arr[: min(vocab_size, arr.size)]
                return m, names
        except Exception:
            pass
    layout = _AETHER_TOKENIZER_LAYOUT or {}
    byte_base = layout.get("byte_base", 3)
    byte_count = layout.get("byte_count", 256)
    if not isinstance(byte_base, int):
        byte_base = 3
    if not isinstance(byte_count, int):
        byte_count = 256
    byte_count = max(0, min(byte_count, max(0, vocab_size - byte_base)))
    if byte_count and byte_base < vocab_size:
        end = min(vocab_size, byte_base + byte_count)
        for tid in range(byte_base, end):
            b = tid - byte_base
            if b < 32 or b == 127:  # control
                if b in (9, 10, 11, 12, 13):
                    m[tid] = 1  # whitespace
                else:
                    m[tid] = 0
            elif b == 32 or b in (9, 10, 11, 12, 13):
                m[tid] = 1  # whitespace
            elif 48 <= b <= 57:
                m[tid] = 2  # digit
            elif 65 <= b <= 90 or 97 <= b <= 122:
                m[tid] = 3  # alpha
            elif b <= 127:
                m[tid] = 4 if chr(b) in string.punctuation else 5
            else:
                m[tid] = 6  # non-ASCII
    if scheme == "byte-compact":
        # merge some buckets: asciiOther->punct, control->whitespace
        m[np_mod.where(m == 5)] = 4
        m[np_mod.where(m == 0)] = 1
        names = ["ws", "digit", "alpha", "punct", "nonASCII", "special"]  # compact view
        # remap indices to 0..5
        # mapping: ws(1)->0, digit(2)->1, alpha(3)->2, punct(4 or 5)->3, nonASCII(6)->4, special(7)->5
        remap = np_mod.array([3, 0, 1, 2, 3, 3, 4, 5], dtype=np_mod.int32)
        m = remap[m]
    return m, names


import string
# === ultramem (optional drop-in) ===
try:
    import ultramem_patch as up
except Exception:
    up = None
# ========= Optional: external wasm bridge (safe import) =========
try:
    from poly_core_wasm_v12e_wasiX import WasmSIMDBridge  # noqa: F401
except Exception:
    WasmSIMDBridge = None
# ---- 外部ポンプ -------------------------------------------------------------
try:
    from spiral_pump_multi import SpiralPump as SpiralPumpEngine
    from spiral_pump_multi import detect_backend as pump_detect_backend
except Exception:
    # Fallback: allow training/inference without external engine
    SpiralPumpEngine = None

    def pump_detect_backend():
        try:
            import torch
        except Exception:
            return "cpu"

        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


#
# ====== Gradient Checkpointing (MPS-safe) ====================================
def enable_gradient_checkpointing(model, every: int = 1):
    """
    Wrap each N-th TransformerBlock.forward with torch.utils.checkpoint.
    Works on MPS (use_reentrant=False).
    """
    try:
        import torch.utils.checkpoint as ckpt
    except Exception:
        print("[CKPT] torch.utils.checkpoint not available; skipping")
        return
    # すでに適用済みなら何もしない
    for i, blk in enumerate(getattr(model, "blocks", [])):
        if (i + 1) % max(1, int(every)) != 0:
            continue
        if hasattr(blk, "_orig_forward"):
            continue
        blk._orig_forward = blk.forward  # keep original

        def _wrap(b):
            def _fw(self, x, pad_mask=None, **kwargs):
                pm = pad_mask if pad_mask is not None else kwargs.pop("pad_mask", None)

                def inner(_x):
                    if hasattr(b, "_orig_forward"):
                        return b._orig_forward(_x, pad_mask=pm, **kwargs)
                    return b.forward(_x, pad_mask=pm, **kwargs)

                return ckpt.checkpoint(
                    inner, x, use_reentrant=False, preserve_rng_state=False
                )

            return _fw

        blk.forward = types.MethodType(_wrap(blk), blk)
    print(f"[CKPT] enabled (every={every})")


def disable_gradient_checkpointing(model):
    """Restore original forward if wrapped."""
    for blk in getattr(model, "blocks", []):
        if hasattr(blk, "_orig_forward"):
            blk.forward = blk._orig_forward
            delattr(blk, "_orig_forward")
    print("[CKPT] disabled")


# --- misc utils --------------------------------------------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed)
    torch.manual_seed(seed)


def detect_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- numerics guard ----------------------------------------------------------
def zero_nan_(t: torch.Tensor):
    if not torch.isfinite(t).all():
        t.data = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


# ====== Tiled SDPA (replace F.sdpa) =========================================
_ORIG_SDPA = F.scaled_dot_product_attention
_DEFAULT_TILE_Q = int(os.environ.get("AETHER_MPS_FLASH_TILE_Q", "192"))
_DEFAULT_TILE_K = int(os.environ.get("AETHER_MPS_FLASH_TILE_K", "320"))
_DEFAULT_FLASH_FP32 = os.environ.get("AETHER_MPS_FLASH_FP32", "1") != "0"
_DEFAULT_WINDOW = int(os.environ.get("AETHER_MPS_FLASH_WINDOW", "0"))
_DEFAULT_GLOBAL_TOKENS = int(os.environ.get("AETHER_MPS_FLASH_GLOBALS", "0"))
_DEFAULT_GLOBAL_STRIDE = int(os.environ.get("AETHER_MPS_FLASH_STRIDE", "0"))

_PATCHED_SDPA = {
    "on": False,
    "tile_q": max(16, _DEFAULT_TILE_Q),
    "tile_k": max(32, _DEFAULT_TILE_K),
    "fp32": bool(_DEFAULT_FLASH_FP32),
    "window": max(0, _DEFAULT_WINDOW),  # 0=OFF, >0: local band (past window only)
    "global_tokens": max(0, _DEFAULT_GLOBAL_TOKENS),  # always-allowed keys from head
    "global_stride": max(0, _DEFAULT_GLOBAL_STRIDE),  # evenly spaced globals (0=off)
}


def _ensure_bhtd(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        return x
    if x.shape[1] < 8 and x.shape[2] > 8:  # (B,T,H,D) -> (B,H,T,D)
        return x.permute(0, 2, 1, 3).contiguous()
    return x


def _build_global_mask(T: int, g0: int, gstep: int, device, dtype=torch.bool):
    if T <= 0 or (g0 <= 0 and gstep <= 0):
        return torch.zeros(T, dtype=dtype, device=device)
    m = torch.zeros(T, dtype=dtype, device=device)
    if g0 > 0:
        m[: min(T, g0)] = True
    if gstep and gstep > 0:
        m[::gstep] = True
    return m


def streaming_sdpa_mps(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    tile_q=192,
    tile_k=320,
    compute_in_fp32=True,
    window_size: int = 0,
    global_tokens: int = 0,
    global_stride: int = 0,
):
    q = _ensure_bhtd(q)
    k = _ensure_bhtd(k)
    v = _ensure_bhtd(v)
    B, H, Tq, D = q.shape
    Tk = k.shape[2]
    dtype_in = q.dtype
    calc = (
        torch.float32
        if (compute_in_fp32 and dtype_in in (torch.float16, torch.bfloat16))
        else dtype_in
    )
    q = q.to(calc)
    k = k.to(calc)
    v = v.to(calc)
    out = torch.empty((B, H, Tq, D), dtype=calc, device=q.device)
    scale = 1.0 / math.sqrt(max(1, D))

    def slicemask(qs, qe, ks, ke):
        if attn_mask is None:
            return None
        m = attn_mask
        try:
            return m[(..., slice(qs, qe), slice(ks, ke))]
        except Exception:
            while m.dim() < 4:
                m = m.unsqueeze(0)
            return m.expand(B, 1, qe - qs, ke - ks)

    gmask_full = _build_global_mask(
        Tk, int(global_tokens), int(global_stride), device=q.device
    )

    for qs in range(0, Tq, tile_q):
        qe = min(Tq, qs + tile_q)
        q_blk = q[:, :, qs:qe, :] * scale
        y = torch.zeros((B, H, qe - qs, D), dtype=calc, device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), dtype=calc, device=q.device)
        m = torch.full((B, H, qe - qs, 1), -float("inf"), dtype=calc, device=q.device)
        k_limit = qe if is_causal else Tk

        for ks in range(0, k_limit, tile_k):
            ke = min(k_limit, ks + tile_k)
            k_blk = k[:, :, ks:ke, :]
            v_blk = v[:, :, ks:ke, :]
            s = torch.einsum("bhtd,bhkd->bhtk", q_blk, k_blk)

            ms = slicemask(qs, qe, ks, ke)
            local_mask = None

            if is_causal or window_size > 0:
                q_idx = torch.arange(qs, qe, device=q.device)
                k_idx = torch.arange(ks, ke, device=q.device)
                causal = (
                    (k_idx.unsqueeze(0) > q_idx.unsqueeze(1)) if is_causal else None
                )
                too_far = (
                    ((q_idx.unsqueeze(1) - k_idx.unsqueeze(0)) > int(window_size))
                    if window_size > 0
                    else None
                )
                if causal is not None and too_far is not None:
                    local_mask = causal | too_far
                elif causal is not None:
                    local_mask = causal
                elif too_far is not None:
                    local_mask = too_far
                if local_mask is not None:
                    gsub = gmask_full[ks:ke].unsqueeze(0)
                    local_mask = local_mask & (~gsub)
                    local_mask = local_mask.unsqueeze(0).unsqueeze(0)

            if ms is not None and local_mask is not None:
                ms = ms | local_mask
            elif local_mask is not None:
                ms = local_mask

            if ms is not None:
                s = s.masked_fill(ms, -float("inf"))

            m_ij = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
            p = torch.exp(s - m_ij)
            if dropout_p and dropout_p > 0:
                p = F.dropout(p, p=float(dropout_p), training=True)
            y = y * torch.exp(m - m_ij) + torch.einsum("bhtk,bhkd->bhtd", p, v_blk)
            l = l * torch.exp(m - m_ij) + p.sum(dim=-1, keepdim=True)
            m = m_ij

        out[:, :, qs:qe, :] = y / torch.clamp_min(l, 1e-9)

    return out.to(dtype_in)


def _patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    window = int(_PATCHED_SDPA["window"])
    globals_n = int(_PATCHED_SDPA["global_tokens"])
    stride = int(_PATCHED_SDPA["global_stride"])
    prefer_flash = (
        window <= 0
        and globals_n <= 0
        and stride <= 0
        and _FLASH_ATTENTION.should_use(q)
    )

    if prefer_flash:
        try:
            return _FLASH_ATTENTION(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=float(dropout_p or 0.0),
                is_causal=bool(is_causal),
                training=bool(dropout_p and dropout_p > 0.0),
                scale_override=scale,
            )
        except Exception as err:
            if _FLASH_ATTENTION._fallbacks < 3:
                print(f"[MPS][FLASH] fallback to streaming SDPA: {err}")
            _FLASH_ATTENTION._fallbacks += 1

    return streaming_sdpa_mps(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        is_causal,
        _PATCHED_SDPA["tile_q"],
        _PATCHED_SDPA["tile_k"],
        _PATCHED_SDPA["fp32"],
        window,
        globals_n,
        stride,
    )


def enable_tiled_sdpa(tile_q=192, tile_k=320, compute_in_fp32=True):
    if not _PATCHED_SDPA["on"]:
        setattr(F, "scaled_dot_product_attention", _patched_sdpa)
        _PATCHED_SDPA["on"] = True
    _PATCHED_SDPA.update(
        {"tile_q": int(tile_q), "tile_k": int(tile_k), "fp32": bool(compute_in_fp32)}
    )


def set_sliding_window(window: int = 0, global_tokens: int = 0, global_stride: int = 0):
    _PATCHED_SDPA["window"] = max(0, int(window))
    _PATCHED_SDPA["global_tokens"] = max(0, int(global_tokens))
    _PATCHED_SDPA["global_stride"] = max(0, int(global_stride))


def disable_tiled_sdpa():
    if _PATCHED_SDPA["on"]:
        setattr(F, "scaled_dot_product_attention", _ORIG_SDPA)
        _PATCHED_SDPA["on"] = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        print(f"[ENV] invalid int for {name}={raw!r}; using {default}")
        return int(default)


def _auto_enable_mps_flash_attention():
    if os.environ.get("AETHER_DISABLE_MPS_FLASH", "0") == "1":
        return
    try:
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return
    except Exception:
        return

    tile_q = max(16, _env_int("AETHER_MPS_FLASH_TILE_Q", _PATCHED_SDPA["tile_q"]))
    tile_k = max(32, _env_int("AETHER_MPS_FLASH_TILE_K", _PATCHED_SDPA["tile_k"]))
    fp32 = os.environ.get("AETHER_MPS_FLASH_FP32", "1") != "0"
    enable_tiled_sdpa(tile_q=tile_q, tile_k=tile_k, compute_in_fp32=fp32)
    window = _env_int("AETHER_MPS_FLASH_WINDOW", _PATCHED_SDPA["window"])
    globals_n = _env_int("AETHER_MPS_FLASH_GLOBALS", _PATCHED_SDPA["global_tokens"])
    stride = _env_int("AETHER_MPS_FLASH_STRIDE", _PATCHED_SDPA["global_stride"])
    if window > 0 or globals_n > 0 or stride > 0:
        set_sliding_window(window, globals_n, stride)
    print(
        "[MPS][FLASH] defaulted to custom kernel "
        f"(tile_q={tile_q}, tile_k={tile_k}, window={window}, globals={globals_n}, stride={stride})"
    )


_auto_enable_mps_flash_attention()


# ====== INT8 base + LoRA (custom; PEFT排他側で利用) ==========================
def _per_channel_symmetric_quant(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # Fallback if weight is not 2D or in_features is zero
        if w.ndim != 2 or w.shape[1] == 0:
            out = w.shape[0] if w.ndim >= 1 else 1
            maxv = w.abs().amax() if w.numel() > 0 else w.new_tensor(1.0)
            scale_val = (maxv / 127.0).to(torch.float32).clamp_min(1e-6)
            wq = torch.zeros_like(w, dtype=torch.int8)
            s = torch.full(
                (out,), float(scale_val.item()), dtype=torch.float32, device=w.device
            )
            return wq, s
        maxv = w.abs().amax(dim=1) + 1e-8
        scale = (maxv / 127.0).to(torch.float32)
        wq = torch.clamp((w / scale.unsqueeze(1)).round_(), -127, 127).to(torch.int8)
        return wq, scale


class LinearInt8Base(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.w_q = nn.Parameter(
            torch.empty(out_f, in_f, dtype=torch.int8), requires_grad=False
        )
        self.s = nn.Parameter(
            torch.ones(out_f, dtype=torch.float32), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.zeros_(self.w_q)
        nn.init.ones_(self.s)

    def load_from_float(self, w: torch.Tensor, b=None):
        with torch.no_grad():
            wq, s = _per_channel_symmetric_quant(w.to(torch.float32))
            self.w_q.copy_(wq)
            self.s.copy_(s)
            if self.b is not None and b is not None:
                self.b.copy_(b)

    def forward(self, x):
        w = self.w_q.to(torch.float32) * self.s.unsqueeze(1)
        y = F.linear(x, w, bias=self.b)
        return y


class LinearInt8LoRA(nn.Module):
    def __init__(
        self,
        in_f,
        out_f,
        r: Optional[int] = None,
        alpha: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        rank = _AETHER_DEFAULT_LORA_R if r is None else int(r)
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        alpha_val = alpha if alpha is not None else max(1, rank * 2)
        self.rank = rank
        self.alpha = int(alpha_val)
        self.base = LinearInt8Base(in_f, out_f, bias=bias)
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scal = float(self.alpha / max(1, rank))
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        y = self.base(x)
        z = self.lora_B(self.drop(self.lora_A(x))) * self.scal
        return y + z


def convert_linear_to_int8_lora(
    model: nn.Module,
    r: int = 160,
    alpha: int = 320,
    dropout: float = 0.0,
    include_names: Optional[List[str]] = None,
    exclude_names: Tuple[str, ...] = ("emb", "head"),
    skip_if_out_equals: Optional[int] = None,
) -> int:
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, (LinearInt8LoRA, LinearInt8Base)):
            continue
        if isinstance(m, nn.Linear):
            if getattr(m, "in_features", 1) == 0 or getattr(m, "out_features", 1) == 0:
                print(
                    f"[INT8-LoRA] skip zero-sized Linear: {name} in={getattr(m, 'in_features', None)} out={getattr(m, 'out_features', None)}"
                )
                continue
            if (
                skip_if_out_equals is not None
                and getattr(m, "out_features", None) == skip_if_out_equals
            ):
                continue
            if include_names is not None and not any(
                (inc in name) for inc in include_names
            ):
                continue
            if any((exc in name) for exc in exclude_names):
                continue
            p = list(m.parameters())
            w = p[0].detach().to(torch.float32)
            b = p[1].detach() if (len(p) > 1 and p[1] is not None) else None
            q = LinearInt8LoRA(
                m.in_features,
                m.out_features,
                r=r,
                alpha=alpha,
                dropout=dropout,
                bias=(b is not None),
            )
            try:
                q = q.to(w.device)
            except Exception:
                pass
            q.base.load_from_float(w, b)
            # replace in parent
            parent = model
            path = name.split(".")
            for seg in path[:-1]:
                parent = getattr(parent, seg)
            setattr(parent, path[-1], q)
            n += 1
    return n


# ====== Tokenizer (Byte-level; PAD/BOS/EOS) ==================================
class ByteTokenizer:
    """Hybrid tokenizer that blends byte, SentencePiece, and TikToken encodings.

    The legacy byte-level behaviour is preserved as a fallback, while optional
    SentencePiece and TikToken vocabularies can be layered on top when the
    corresponding libraries/models are available. At runtime the tokenizer picks
    whichever sub-tokenization yields the fewest tokens for a given fragment,
    enabling compact yet expressive sequences without sacrificing robustness.
    """

    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(
        self,
        vocab_size: int = 32000,
        sp_model_path: Optional[str] = None,
        sp_model_proto: Optional[bytes] = None,
        tiktoken_encoding: Optional[str] = None,
        byte_fallback: bool = True,
    ):
        self._sp = None
        self._sp_vocab = 0
        self._tt = None
        self._tt_vocab = 0
        self._byte_fallback = bool(byte_fallback)

        info_bits: List[str] = []

        if _sentencepiece is not None and (sp_model_path or sp_model_proto):
            proc = _sentencepiece.SentencePieceProcessor()
            try:
                if sp_model_proto is not None:
                    proc.LoadFromSerializedProto(sp_model_proto)
                elif sp_model_path is not None:
                    proc.Load(sp_model_path)
                self._sp_vocab = int(proc.get_piece_size())
                if self._sp_vocab > 0:
                    self._sp = proc
                    info_bits.append(f"sentencepiece={self._sp_vocab}")
                else:
                    print("[TOK] SentencePiece model has no vocabulary; disabling")
            except Exception as exc:
                print(f"[TOK] Failed to load SentencePiece model: {exc}")
        elif sp_model_path and _sentencepiece is None:
            print(
                f"[TOK] sentencepiece package is unavailable; requested model {sp_model_path!r} ignored"
            )

        if tiktoken_encoding and _tiktoken is not None:
            try:
                enc = _tiktoken.get_encoding(tiktoken_encoding)
                vocab = int(getattr(enc, "n_vocab", 0))
                if vocab > 0:
                    self._tt = enc
                    self._tt_vocab = vocab
                    info_bits.append(f"tiktoken={tiktoken_encoding}:{vocab}")
                else:
                    print(
                        f"[TOK] TikToken encoding {tiktoken_encoding!r} has zero vocab; disabling"
                    )
            except Exception as exc:
                print(f"[TOK] Failed to load TikToken encoding {tiktoken_encoding!r}: {exc}")
        elif tiktoken_encoding and _tiktoken is None:
            print(
                f"[TOK] tiktoken package is unavailable; requested encoding {tiktoken_encoding!r} ignored"
            )

        self._sp_base = self.EOS + 1
        offset = self._sp_base
        if self._sp is not None and self._sp_vocab > 0:
            self._sp_base = offset
            offset += self._sp_vocab
        else:
            self._sp_base = offset

        self._tt_base = offset
        if self._tt is not None and self._tt_vocab > 0:
            self._tt_base = offset
            offset += self._tt_vocab

        self._byte_base = offset
        if self._byte_fallback:
            offset += 256
            info_bits.append("byte-fallback")
        else:
            info_bits.append("no-byte-fallback")

        self._total_vocab = offset
        self.vocab_size = max(int(vocab_size), self._total_vocab)
        if self.vocab_size > self._total_vocab:
            info_bits.append(f"pad={self.vocab_size - self._total_vocab}")

        if not info_bits:
            info_bits.append("pure-byte")

        global _AETHER_TOKENIZER_LAYOUT
        _AETHER_TOKENIZER_LAYOUT = {
            "byte_base": self._byte_base if self._byte_fallback else None,
            "byte_count": 256 if self._byte_fallback else 0,
            "sp_base": self._sp_base if self._sp is not None else None,
            "sp_size": self._sp_vocab if self._sp is not None else 0,
            "tt_base": self._tt_base if self._tt is not None else None,
            "tt_size": self._tt_vocab if self._tt is not None else 0,
        }

        print(
            f"[TOK] Hybrid tokenizer ready ({', '.join(info_bits)}) → vocab={self.vocab_size}"
        )

    # ------------------------------------------------------------------ helpers
    def _encode_bytes(self, segment: str) -> List[int]:
        if not self._byte_fallback:
            return []
        data = segment.encode("utf-8", errors="ignore")
        if not data:
            return []
        return [self._byte_base + int(b) for b in data]

    def _sp_piece_to_id(self, piece: str) -> Optional[int]:
        if self._sp is None:
            return None
        try:
            pid = int(self._sp.piece_to_id(piece))
        except Exception:
            return None
        return pid if pid >= 0 else None

    # ---------------------------------------------------------------- interface
    def encode(self, s: str) -> List[int]:
        if not isinstance(s, str):
            s = str(s)

        tokens: List[int] = [self.BOS]
        segments: List[str]
        if self._sp is not None:
            try:
                segments = self._sp.encode(s, out_type=str)
            except Exception:
                segments = [s]
        else:
            segments = [s]

        for seg in segments:
            best: Optional[List[int]] = None

            if self._sp is not None:
                pid = self._sp_piece_to_id(seg)
                if pid is not None and 0 <= pid < self._sp_vocab:
                    best = [self._sp_base + pid]

            if self._tt is not None:
                try:
                    tt_raw = self._tt.encode(seg)
                except Exception:
                    tt_raw = []
                tt_ids = [self._tt_base + int(tid) for tid in tt_raw if tid < self._tt_vocab]
                if tt_ids and (best is None or len(tt_ids) < len(best)):
                    best = tt_ids

            if not best:
                best = self._encode_bytes(seg)

            if not best:
                # Absolute fallback: character-wise (may still be empty for whitespace)
                for ch in seg:
                    ch_ids = self._encode_bytes(ch)
                    if ch_ids:
                        tokens.extend(ch_ids)
                continue

            tokens.extend(best)

        tokens.append(self.EOS)
        return tokens

    def decode(self, ids: List[int]) -> str:
        pieces: List[str] = []

        sp_buffer: List[int] = []
        tt_buffer: List[int] = []

        def _flush_sp():
            if not sp_buffer or self._sp is None:
                sp_buffer.clear()
                return
            try:
                pieces.append(self._sp.decode_ids(sp_buffer))
            except Exception:
                for pid in sp_buffer:
                    try:
                        pieces.append(self._sp.id_to_piece(pid))
                    except Exception:
                        continue
            finally:
                sp_buffer.clear()

        def _flush_tt():
            if not tt_buffer or self._tt is None:
                tt_buffer.clear()
                return
            try:
                pieces.append(self._tt.decode(tt_buffer))
            except Exception:
                for tid in tt_buffer:
                    try:
                        pieces.append(self._tt.decode([tid]))
                    except Exception:
                        continue
            finally:
                tt_buffer.clear()

        for tid in ids:
            if tid in (self.PAD, self.BOS, self.EOS):
                continue
            if self._sp is not None and self._sp_base <= tid < self._sp_base + self._sp_vocab:
                _flush_tt()
                sp_buffer.append(int(tid - self._sp_base))
                continue
            if self._tt is not None and self._tt_base <= tid < self._tt_base + self._tt_vocab:
                _flush_sp()
                tt_buffer.append(int(tid - self._tt_base))
                continue

            _flush_sp()
            _flush_tt()

            if self._byte_fallback and self._byte_base <= tid < self._byte_base + 256:
                pieces.append(
                    bytes([int(tid - self._byte_base)]).decode("utf-8", errors="ignore")
                )

        _flush_sp()
        _flush_tt()

        text = "".join(pieces)
        if self._sp is not None:
            text = text.replace("\u2581", " ")
        return text


# ====== RMSNorm / SwiGLU / Rotary ===========================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.g


class SwiGLU(nn.Module):
    def __init__(self, d, mult=4.0):
        super().__init__()
        h = int(d * mult)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base_theta: float = 10_000.0, scaling: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.theta = float(base_theta)
        self.scaling = float(scaling)

    def _freqs(self, T: int, device, dtype):
        dim = self.dim
        inv = 1.0 / (
            self.theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        t = torch.arange(T, device=device, dtype=torch.float32) * float(self.scaling)
        freqs = torch.einsum("i,j->ij", t, inv)
        return torch.cat([freqs, freqs], dim=-1)

    # --- after ---
    def apply_rotary(self, x):
        """
        x: [B, H, T, D] を想定。最後の次元 D を前半/後半に分け、半次元で回転をかける。
        D が奇数でもパディング→適用→元に戻すので安全。
        """
        import torch
        import torch.nn.functional as F

        B, H, T, D = x.shape
        half = D // 2
        odd = D % 2 == 1
        if odd:
            # 奇数 head_dim の場合は末尾に 1 を詰めてから処理
            x = F.pad(x, (0, 1))
            D += 1
            half = D // 2

        # 周波数は必ず「半次元」で作る（ここが要点！）
        device, dtype = x.device, x.dtype
        # self.theta は rope_theta、self.scaling は dict か None を想定
        inv = 1.0 / (
            self.theta
            ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
        )
        t = torch.arange(T, device=device, dtype=torch.float32)

        # freqs: [T, half]
        freqs = torch.einsum("t,d->td", t, inv)

        # rope_scaling（linear 等）を使っている場合のみスケール
        if isinstance(self.scaling, dict):
            sc = self.scaling.get("factor") or self.scaling.get("scale")
            if sc:
                freqs = freqs / float(sc)

        # cos/sin: [1, 1, T, half] に整形（H にブロードキャスト）
        cos = torch.cos(freqs).to(dtype).view(1, 1, T, half)
        sin = torch.sin(freqs).to(dtype).view(1, 1, T, half)

        # 前半/後半に分解して回転
        x1 = x[..., :half]
        x2 = x[..., half : half * 2]
        xr = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        # 奇数 head_dim の場合は詰めた 1 を落とす
        if odd:
            xr = xr[..., : D - 1]

        return xr


# ====== MHA / Block / Model ==================================================
class MHA(nn.Module):
    """single qkv proj + GQA/MQA kv_heads"""

    def __init__(
        self,
        d,
        heads,
        dropout=0.0,
        use_rope: bool = False,
        rope_theta: float = 10_000.0,
        rope_scaling: float = 1.0,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.hq = int(heads)
        self.hk = int(kv_heads) if kv_heads else int(heads)
        assert self.hq % self.hk == 0, "hq must be divisible by hk"
        self.d = int(d)
        self.dk = d // self.hq
        self.drop = nn.Dropout(dropout)
        self.qkv = nn.Linear(d, self.hq * self.dk + 2 * self.hk * self.dk, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.use_rope = bool(use_rope)
        self.rope = (
            RotaryEmbedding(self.dk, base_theta=rope_theta, scaling=rope_scaling)
            if self.use_rope
            else None
        )

    def forward(self, x, pad_mask=None, attn_mask=None, is_causal=True):
        B, T, D = x.shape
        z = self.qkv(x)
        q_end = self.hq * self.dk
        kv_end = q_end + 2 * self.hk * self.dk
        q = z[:, :, :q_end].view(B, T, self.hq, self.dk).permute(0, 2, 1, 3)
        kv = (
            z[:, :, q_end:kv_end].view(B, T, self.hk, 2, self.dk).permute(0, 2, 1, 3, 4)
        )
        k = kv[:, :, :, 0, :]
        v = kv[:, :, :, 1, :]
        if self.hk != self.hq:
            rep = self.hq // self.hk
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if self.use_rope:
            if self.rope is None:
                raise RuntimeError("rope not initialized")
            self.rope.to(x.device, x.dtype)
            if (self.dk % 2) != 0:
                q = F.pad(q, (0, 1))
                k = F.pad(k, (0, 1))
            q = self.rope.apply_rotary(q)
            k = self.rope.apply_rotary(k)
            if (self.dk % 2) != 0:
                q = q[..., : self.dk]
                k = k[..., : self.dk]

        attn_dropout = self.drop.p if self.training else 0.0
        if _FLASH_ATTENTION.should_use(q):
            y = _FLASH_ATTENTION(
                q,
                k,
                v,
                pad_mask=pad_mask,
                attn_mask=attn_mask,
                dropout_p=attn_dropout,
                is_causal=is_causal,
                training=self.training,
            )
        else:
            m = None
            if pad_mask is not None:
                m = (~pad_mask).unsqueeze(1).unsqueeze(2).expand(B, 1, T, T)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask if attn_mask is not None else m,
                dropout_p=attn_dropout,
                is_causal=is_causal,
            )
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d,
        heads,
        dropout=0.0,
        *,
        ff_mult: float = 2.6666667,
        use_rope: bool = False,
        rope_theta: float = 10_000.0,
        rope_scaling: float = 1.0,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = MHA(
            d,
            heads,
            dropout,
            use_rope=use_rope,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            kv_heads=kv_heads,
        )
        self.n2 = RMSNorm(d)
        self.ff = SwiGLU(d, mult=ff_mult)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.attn(self.n1(x), pad_mask=pad_mask))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class AetherPumpSimple(nn.Module):
    def __init__(
        self,
        vocab_size=32000,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        dropout=0.1,
        max_len=4096,
        pad_id=0,
        tie_weights=True,
        ff_mult: float = 2.6666667,
        use_rope: bool = True,
        rope_theta: float = 10_000.0,
        rope_scaling: float = 1.0,
        use_abs_pos: bool = False,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.use_abs_pos = bool(use_abs_pos)
        if self.use_abs_pos:
            self.pos = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout,
                    ff_mult=ff_mult,
                    use_rope=use_rope,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    kv_heads=kv_heads,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.emb.weight
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        device = input_ids.device
        pad_mask = input_ids != self.pad_id
        x = self.emb(input_ids)
        if self.use_abs_pos:
            x = x + self.pos[:T, :].to(device)
        for blk in self.blocks:
            x = blk(x, pad_mask=pad_mask)
        x = self.ln_f(x)
        return self.head(x)

    def forward_with_bias(
        self, input_ids: torch.Tensor, bias: torch.Tensor, attention_mask=None, **kwargs
    ):
        B, T = input_ids.shape
        device = input_ids.device
        pad_mask = input_ids != self.pad_id
        x = self.emb(input_ids) + bias.unsqueeze(1)
        if self.use_abs_pos:
            x = x + self.pos[:T, :].to(device)
        for blk in self.blocks:
            x = blk(x, pad_mask=pad_mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def set_rope_params(
        self,
        use_rope: bool = True,
        rope_theta: float = 10_000.0,
        rope_scaling: float = 1.0,
        max_pos: Optional[int] = None,
    ):
        mp = max_pos if max_pos is not None else getattr(self, "max_len", 4096)
        for blk in self.blocks:
            if hasattr(blk, "attn") and isinstance(blk.attn, MHA):
                blk.attn.use_rope = bool(use_rope)
                if blk.attn.use_rope:
                    rope_dim = (
                        blk.attn.dk if (blk.attn.dk % 2) == 0 else (blk.attn.dk + 1)
                    )
                    blk.attn.rope = RotaryEmbedding(
                        rope_dim,
                        base_theta=float(rope_theta),
                        scaling=float(rope_scaling),
                    )

# ====== Collate / Datasets ===================================================
def collate_lm_safe(batch, pad_id: int):
    # batch: List[List[int]]  or  List[Tuple[List[int], List[int]]]
    if (
        len(batch) > 0
        and isinstance(batch[0], (tuple, list))
        and isinstance(batch[0][0], list)
    ):
        seqs = [b[0] for b in batch]  # 入力側のみ使用
    else:
        seqs = batch
    mx = max(2, max(len(x) for x in seqs))
    X = torch.full((len(seqs), mx), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        X[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return X[:, : mx - 1], X[:, 1:mx]


class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        glob_pat: str,
        tok: ByteTokenizer,
        pack_len: int = 1024,
        buffer_size: int = 8192,
        infinite: bool = True,
        seed: int = 1337,
        shuffle_blocks: Optional[int] = None,
    ):
        super().__init__()
        self.glob = glob_pat
        self.tok = tok
        self.pack_len = pack_len
        self.buffer_size = buffer_size
        self.infinite = infinite
        self.seed = seed
        if shuffle_blocks is None:
            shuffle_blocks = int(os.environ.get("AETHER_STREAM_SHUFFLE_BLOCKS", "64"))
        self.shuffle_blocks = max(0, int(shuffle_blocks))

    def _files(self, worker_id: int = 0, num_workers: int = 1, epoch: int = 0):
        # 再帰で拾う。0件なら即エラーで可視化（無限待ち防止）
        fs = sorted(glob.glob(self.glob, recursive=True))
        if not fs:
            raise FileNotFoundError(
                f"[DATA] No files match: {self.glob}  (cwd={os.getcwd()})"
            )
        random.Random(self.seed + int(epoch)).shuffle(fs)
        if num_workers <= 1:
            return fs
        sharded = fs[int(worker_id) :: int(num_workers)]
        if not sharded:
            sharded = [fs[int(worker_id) % len(fs)]]
        return sharded

    def __iter__(self):
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = int(worker.num_workers) if worker is not None else 1
        rng = random.Random(self.seed + worker_id * 1009)
        shuffle_blocks = int(self.shuffle_blocks)
        pending_blocks: List[List[int]] = []
        epoch = 0

        def emit_block(block: List[int]):
            if len(block) < 2:
                return []
            if shuffle_blocks <= 1:
                return [block]
            pending_blocks.append(block)
            if len(pending_blocks) < shuffle_blocks:
                return []
            idx = rng.randrange(len(pending_blocks))
            return [pending_blocks.pop(idx)]

        while True:
            for f in self._files(worker_id, num_workers, epoch):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        buf = ""
                        for line in fh:
                            buf += line
                            if len(buf) >= self.buffer_size:
                                if (
                                    _KBRIDGE_AVAILABLE
                                    and os.environ.get("KBRIDGE_PREPROC", "0") == "1"
                                ):
                                    try:
                                        nbytes = byte_normalize_utf8(buf)
                                        buf = nbytes.decode("utf-8", errors="ignore")
                                    except Exception:
                                        pass
                                ids = self.tok.encode(buf)
                                buf = ""
                                for s in range(0, max(1, len(ids) - 1), self.pack_len):
                                    block = ids[s : s + self.pack_len]
                                    for out in emit_block(block):
                                        yield out, out
                        if buf:
                            if (
                                _KBRIDGE_AVAILABLE
                                and os.environ.get("KBRIDGE_PREPROC", "0") == "1"
                            ):
                                try:
                                    nbytes = byte_normalize_utf8(buf)
                                    buf = nbytes.decode("utf-8", errors="ignore")
                                except Exception:
                                    pass
                            ids = self.tok.encode(buf)
                            for s in range(0, max(1, len(ids) - 1), self.pack_len):
                                block = ids[s : s + self.pack_len]
                                for out in emit_block(block):
                                    yield out, out
                except Exception:
                    pass
            while pending_blocks:
                idx = rng.randrange(len(pending_blocks))
                block = pending_blocks.pop(idx)
                yield block, block
            if not self.infinite:
                break
            epoch += 1


# ====== Trainer utils ========================================================
@dataclass
class CurriculumStage:
    until_step: int
    max_len: int


@dataclass
class TrainConfig:
    epochs: int = 1
    max_steps: Optional[int] = 2000
    safetensor_every: int = 0
    batch_size: int = 1
    micro_batch: int = 1
    lr: float = 2e-4
    warmup_steps: int = 300
    grad_clip: Optional[float] = 1.0
    max_len: int = 2048
    out_dir: str = "runs/v28"
    seed: int = 1337
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 500
    use_tb: bool = False
    curriculum: List[CurriculumStage] = field(default_factory=list)
    token_dropout: float = 0.01
    byte_noise: float = 0.0
    span_mask_prob: float = 0.02
    span_mask_len: int = 8
    tiled_q: int = 192
    tiled_k: int = 320
    mps_sync_every: int = 0
    loss_guard: float = 16.0
    ppl_cap: float = 12.0
    ppl_smooth_beta: float = 0.90
    tps_target: float = 0.0
    tps_boost_patience: int = 24
    # LVI regs / switches
    lvi_mv_weight: float = 0.10
    lvi_two_view_weight: float = 0.05
    lvi_enable: bool = False
    lvi_k: int = 64
    lvi_alpha_mode: str = "sparsemax"
    lvi_every: int = 4
    # Attention runtime
    window_size: int = 0
    global_tokens: int = 0
    global_stride: int = 0
    # Optimizer runtime
    opt_cpu8bit: bool = False
    # Intention loss
    intent_weight: float = 0.0
    intent_margin: float = 0.10
    intent_sample_frac: float = 0.25
    intent_every: int = 4
    # ReLoRA
    relora_every: int = 0

    # LoRA default rank
    lora_r: int = _AETHER_DEFAULT_LORA_R

    # GaLore-like optimizer
    opt_galore: bool = False
    galore_rank: int = 64
    # GQA
    kv_heads: Optional[int] = None
    # M4 / MPS runtime controls
    prefer_bfloat16: bool = True
    matmul_precision: str = "high"
    compile: bool = False
    compile_backend: str = "aot_eager"
    compile_dynamic: bool = False
    adaptive_microbatch: bool = True
    adaptive_micro_max: int = 16
    adaptive_micro_recover: int = 256
    oom_retries: int = 3
    grad_scaler: bool = True
    empty_cache_every: int = 0
    # Data pipeline / host→device controls
    loader_num_workers: int = 0
    loader_prefetch_factor: int = 2
    loader_persistent_workers: bool = False
    loader_pin_memory: bool = False
    prefetch_to_device: bool = True
    disallow_mps_fallback: bool = True
    auto_mps_7b: bool = True
    auto_mps_param_threshold: int = 5_000_000_000
    auto_mps_target_seq: int = 4096
    auto_mps_token_budget: int = 8192
    auto_mps_min_micro_batch: int = 1
    mps_7b_lora_rank: Optional[int] = None
    mps_7b_lora_alpha: Optional[int] = None
    mps_7b_lora_dropout: float = 0.05
    mps_7b_int8_exclude: Tuple[str, ...] = ("emb", "head")
    mps_7b_skip_if_out_equals: Optional[int] = None
    # Turbo governor (MPS throughput tuning)
    turbo_target_tok: float = 2000.0
    turbo_window: int = 6
    turbo_cooldown: int = 48
    turbo_seq_floor: int = 512
    turbo_seq_step: int = 256
    turbo_disable_metrics_ratio: float = 0.75
    turbo_micro_floor: int = 1


def _auto_tune_for_mps_7b(
    model: nn.Module, cfg: TrainConfig, device: torch.device
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if not getattr(cfg, "auto_mps_7b", False):
        return report
    if getattr(cfg, "_auto_mps_7b_done", False):
        return report
    cfg._auto_mps_7b_done = True  # type: ignore[attr-defined]
    total_params = int(sum(p.numel() for p in model.parameters()))
    report["params"] = total_params
    if device.type != "mps":
        report["skipped"] = "non_mps_device"
        print("[MPS][7B] auto-tune skipped: device is not MPS")
        return report
    threshold = int(getattr(cfg, "auto_mps_param_threshold", 5_000_000_000))
    if total_params < threshold:
        report["skipped"] = "below_threshold"
        print(
            f"[MPS][7B] auto-tune skipped: params={total_params:,} < threshold={threshold:,}"
        )
        return report

    adjustments: List[str] = []

    def _update(name: str, current, new_val):
        if current != new_val:
            adjustments.append(f"{name}:{current}->{new_val}")
            report[name] = {"old": current, "new": new_val}
        return new_val

    cfg_max_len = int(getattr(cfg, "max_len", 4096))
    target_seq = int(getattr(cfg, "auto_mps_target_seq", cfg_max_len))
    target_seq = max(512, min(cfg_max_len, target_seq))
    token_budget = int(
        getattr(cfg, "auto_mps_token_budget", target_seq * max(1, int(cfg.batch_size)))
    )
    token_budget = max(target_seq, token_budget)
    min_micro = max(1, int(getattr(cfg, "auto_mps_min_micro_batch", 1)))
    report["token_budget"] = {
        "target": token_budget,
        "sequence": target_seq,
        "min_micro": min_micro,
    }

    cfg.max_len = _update("max_len", cfg.max_len, min(cfg_max_len, target_seq))

    ideal_batch = max(1, math.ceil(token_budget / target_seq))
    cfg.batch_size = _update(
        "batch_size", cfg.batch_size, max(1, min(cfg.batch_size, ideal_batch))
    )

    desired_micro = max(min_micro, min(cfg.micro_batch, cfg.batch_size))
    if cfg.batch_size <= 2:
        desired_micro = max(1, int(cfg.batch_size))
    cfg.micro_batch = _update("micro_batch", cfg.micro_batch, max(1, int(desired_micro)))

    micro = max(1, int(cfg.micro_batch))
    chunk_size = max(1, math.ceil(max(1, int(cfg.batch_size)) / micro))
    effective_tokens = max(1, int(cfg.batch_size)) * target_seq
    chunk_tokens = chunk_size * target_seq
    report["token_budget"].update(
        {
            "effective": effective_tokens,
            "chunk_size": chunk_size,
            "chunk_tokens": chunk_tokens,
            "micro": micro,
            "max_len": int(cfg.max_len),
        }
    )

    new_micro_cap = max(cfg.adaptive_micro_max, micro * 4, max(1, int(cfg.batch_size)) * 4)
    cfg.adaptive_micro_max = _update(
        "adaptive_micro_max", cfg.adaptive_micro_max, new_micro_cap
    )
    new_recover = max(cfg.adaptive_micro_recover, max(256, micro * 32), 512)
    cfg.adaptive_micro_recover = _update(
        "adaptive_micro_recover",
        cfg.adaptive_micro_recover,
        new_recover,
    )
    if float(getattr(cfg, "tps_target", 0.0)) <= 0.0:
        cfg.tps_target = _update("tps_target", cfg.tps_target, float(chunk_tokens))
    report["token_budget"]["tps_target"] = float(getattr(cfg, "tps_target", 0.0))
    cfg.loader_num_workers = _update(
        "loader_num_workers", cfg.loader_num_workers, max(cfg.loader_num_workers, 2)
    )
    cfg.loader_prefetch_factor = _update(
        "loader_prefetch_factor",
        cfg.loader_prefetch_factor,
        max(cfg.loader_prefetch_factor, 6),
    )
    cfg.loader_pin_memory = _update(
        "loader_pin_memory", cfg.loader_pin_memory, True
    )
    if cfg.loader_num_workers > 0 and not cfg.loader_persistent_workers:
        cfg.loader_persistent_workers = True
        adjustments.append("loader_persistent_workers:False->True")
        report["loader_persistent_workers"] = {"old": False, "new": True}
    if not cfg.loader_pin_memory:
        cfg.loader_pin_memory = True
        adjustments.append("loader_pin_memory:False->True")
        report["loader_pin_memory"] = {"old": False, "new": True}
    if not cfg.prefetch_to_device:
        cfg.prefetch_to_device = True
        adjustments.append("prefetch_to_device:False->True")
    if not cfg.disallow_mps_fallback:
        cfg.disallow_mps_fallback = True
        adjustments.append("disallow_mps_fallback:False->True")
    if not cfg.grad_scaler:
        cfg.grad_scaler = True
        adjustments.append("grad_scaler:False->True")
    if not cfg.prefer_bfloat16:
        cfg.prefer_bfloat16 = True
        adjustments.append("prefer_bfloat16:False->True")
    matmul_mode = getattr(cfg, "matmul_precision", None)
    if not matmul_mode:
        cfg.matmul_precision = "high"
        adjustments.append("matmul_precision:None->high")
        report["matmul_precision"] = {"old": None, "new": "high"}
    target_window = int(target_seq)
    if target_window > 0:
        new_window = cfg.window_size if cfg.window_size > 0 else target_window
        new_window = max(new_window, target_window)
        cfg.window_size = _update("window_size", cfg.window_size, new_window)
    if cfg.global_tokens <= 0:
        cfg.global_tokens = max(1, target_window // 512)
        adjustments.append(f"global_tokens:0->{cfg.global_tokens}")
    cfg.global_stride = _update(
        "global_stride",
        cfg.global_stride,
        max(cfg.global_stride, max(1, target_window // max(1, cfg.global_tokens))),
    )
    cfg.mps_sync_every = _update(
        "mps_sync_every", cfg.mps_sync_every, max(cfg.mps_sync_every, 64)
    )
    cfg.empty_cache_every = _update(
        "empty_cache_every", cfg.empty_cache_every, max(cfg.empty_cache_every, 128)
    )
    cfg.oom_retries = _update(
        "oom_retries", cfg.oom_retries, max(cfg.oom_retries, 5)
    )
    cfg.tiled_q = _update("tiled_q", cfg.tiled_q, max(cfg.tiled_q, 224))
    cfg.tiled_k = _update("tiled_k", cfg.tiled_k, max(cfg.tiled_k, 384))
    cfg.loss_guard = _update("loss_guard", cfg.loss_guard, max(cfg.loss_guard, 18.0))
    cfg.ppl_cap = _update("ppl_cap", cfg.ppl_cap, max(cfg.ppl_cap, 12.0))
    cfg.ppl_smooth_beta = _update(
        "ppl_smooth_beta", cfg.ppl_smooth_beta, max(cfg.ppl_smooth_beta, 0.92)
    )
    cfg.tps_target = _update("tps_target", cfg.tps_target, max(cfg.tps_target, 200.0))
    cfg.tps_boost_patience = _update(
        "tps_boost_patience",
        cfg.tps_boost_patience,
        max(cfg.tps_boost_patience, 16),
    )
    cfg.turbo_target_tok = _update(
        "turbo_target_tok", cfg.turbo_target_tok, max(cfg.turbo_target_tok, 2000.0)
    )
    cfg.turbo_seq_floor = _update(
        "turbo_seq_floor", cfg.turbo_seq_floor, max(128, min(cfg.turbo_seq_floor, target_seq))
    )
    cfg.turbo_seq_step = _update(
        "turbo_seq_step",
        cfg.turbo_seq_step,
        max(cfg.turbo_seq_step, max(64, target_seq // 4)),
    )
    cfg.turbo_micro_floor = _update(
        "turbo_micro_floor",
        cfg.turbo_micro_floor,
        max(cfg.turbo_micro_floor, min_micro),
    )

    env_updates: Dict[str, Any] = {}

    def _env_default(key: str, val: Any):
        if os.environ.get(key) is None:
            os.environ[key] = str(val)
            env_updates[key] = val

    _env_default("AETHER_MPS_WARMUP_STEPS", 3)
    _env_default("AETHER_MPS_WARMUP_SIZE", max(1024, target_window))
    _env_default("AETHER_MPS_WARMUP_HEADS", max(16, getattr(model, "n_heads", 32)))
    _env_default("AETHER_MPS_WARMUP_SEQ", min(2048, target_window))
    _env_default("AETHER_MPS_WARMUP_ATTENTION", 1)
    if os.environ.get("AETHER_MPS_WARMUP_ATTENTION_STEPS") is None:
        attn_steps = (cfg.mps_sync_every // 16) or 1
        attn_steps = max(1, min(4, attn_steps))
        _env_default("AETHER_MPS_WARMUP_ATTENTION_STEPS", attn_steps)
    head_dim = None
    if hasattr(model, "head_dim"):
        try:
            head_dim = int(getattr(model, "head_dim"))
        except Exception:
            head_dim = None
    if head_dim is None and hasattr(model, "d_model") and getattr(model, "n_heads", 0):
        try:
            head_dim = int(getattr(model, "d_model")) // max(1, int(getattr(model, "n_heads")))
        except Exception:
            head_dim = None
    if head_dim is not None and head_dim > 0:
        _env_default("AETHER_MPS_WARMUP_HEAD_DIM", max(64, head_dim))
    _env_default("AETHER_MPS_ASYNC", 1)
    _env_default("AETHER_SDPA_WINDOW", cfg.window_size)
    _env_default("AETHER_MPS_FLASH_WINDOW", cfg.window_size)
    _env_default("AETHER_MPS_FLASH_TILE_Q", cfg.tiled_q)
    _env_default("AETHER_MPS_FLASH_TILE_K", cfg.tiled_k)
    _env_default("AETHER_MPS_FLASH_GLOBALS", max(0, cfg.global_tokens))
    _env_default(
        "AETHER_MPS_FLASH_STRIDE", max(0, cfg.global_stride)
    )
    if os.environ.get("AETHER_MPS_FLASH_FP32") is None:
        _env_default("AETHER_MPS_FLASH_FP32", 1)
    _env_default("PYTORCH_MPS_HIGH_WATERMARK_RATIO", 0.92)

    rank = cfg.mps_7b_lora_rank or max(16, _AETHER_DEFAULT_LORA_R // 2)
    alpha = cfg.mps_7b_lora_alpha or max(rank * 2, 64)
    dropout = float(getattr(cfg, "mps_7b_lora_dropout", 0.05))
    skip_out = cfg.mps_7b_skip_if_out_equals
    if skip_out is None and hasattr(model, "vocab_size"):
        skip_out = int(getattr(model, "vocab_size"))
    converted = convert_linear_to_int8_lora(
        model,
        r=rank,
        alpha=alpha,
        dropout=dropout,
        include_names=None,
        exclude_names=tuple(getattr(cfg, "mps_7b_int8_exclude", ("emb", "head"))),
        skip_if_out_equals=skip_out,
    )
    if converted > 0:
        report["int8_lora_converted"] = converted
        adjustments.append(f"int8_lora:{converted}")

    if env_updates:
        report["env"] = env_updates

    summary = ", ".join(adjustments) if adjustments else "defaults"
    token_note = ""
    tb = report.get("token_budget")
    if isinstance(tb, dict):
        eff = tb.get("effective")
        tgt = tb.get("target")
        chunk_tokens = tb.get("chunk_tokens")
        chunk_size = tb.get("chunk_size")
        seq_len = tb.get("sequence")
        micro = tb.get("micro")
        if eff and tgt:
            token_note = f"; tokens={int(eff):,}/{int(tgt):,}"
            if chunk_tokens and chunk_size and seq_len and micro is not None:
                token_note += (
                    f" (chunk={int(chunk_size)}×{int(seq_len)}, micro={int(micro)})"
                )
    print(
        f"[MPS][7B] auto-tune applied (params≈{total_params/1e9:.2f}B): {summary}{token_note}"
    )
    return report


class FabricLite:
    def __init__(self, outdir, use_tb=False):
        self.outdir = outdir
        self.use_tb = use_tb
        self.tb = None
        try:
            if use_tb:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=outdir)
        except Exception:
            self.tb = None

    def log(self, d: Dict[str, float], step: int = None):
        if self.tb:
            for k, v in d.items():
                try:
                    self.tb.add_scalar(k, float(v), global_step=step)
                except Exception:
                    pass
        ks = ", ".join([f"{k}={v:.4f}" for k, v in d.items()])
        print(f"[LOG] step={step} | {ks}")


@dataclass(frozen=True)
class ThroughputSnapshot:
    ema: float
    instant: float
    window_avg: float
    window_min: float
    window_max: float
    window_tokens: float
    window_seconds: float
    window_updates: int
    instant_tokens: float
    instant_seconds: float
    lifetime_avg: float
    lifetime_tokens: float
    lifetime_seconds: float
    samples: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "ema": float(self.ema),
            "instant": float(self.instant),
            "window_avg": float(self.window_avg),
            "window_min": float(self.window_min),
            "window_max": float(self.window_max),
            "window_tokens": float(self.window_tokens),
            "window_seconds": float(self.window_seconds),
            "window_updates": float(self.window_updates),
            "window_size": float(self.window_updates),
            "instant_tokens": float(self.instant_tokens),
            "instant_seconds": float(self.instant_seconds),
            "lifetime_avg": float(self.lifetime_avg),
            "lifetime_tokens": float(self.lifetime_tokens),
            "lifetime_seconds": float(self.lifetime_seconds),
            "samples": float(self.samples),
        }


class ThroughputMeter:
    def __init__(self, beta: float = 0.90, window: int = 8):
        self.beta = float(beta)
        self.value = 0.0
        self.ready = False
        self.window = max(1, int(window))
        self._history: Deque[Tuple[float, float, float]] = deque()
        self._window_tokens = 0.0
        self._window_seconds = 0.0
        self._window_min_q: Deque[float] = deque()
        self._window_max_q: Deque[float] = deque()
        self._last_inst = 0.0
        self._last_tokens = 0.0
        self._last_seconds = 0.0
        self._updates = 0
        self.total_tokens = 0.0
        self.total_seconds = 0.0

    def _evict_oldest(self):
        if len(self._history) < self.window:
            return
        old_tokens, old_seconds, old_inst = self._history.popleft()
        self._window_tokens -= old_tokens
        self._window_seconds -= old_seconds
        if self._window_min_q and self._window_min_q[0] == old_inst:
            self._window_min_q.popleft()
        if self._window_max_q and self._window_max_q[0] == old_inst:
            self._window_max_q.popleft()

    def update(self, tokens: float, seconds: float) -> float:
        if seconds <= 0:
            return self.value if self.ready else 0.0
        tokens = float(tokens)
        seconds = float(seconds)
        inst = tokens / max(seconds, 1e-6)
        self._evict_oldest()
        self._history.append((tokens, seconds, inst))
        self._window_tokens += tokens
        self._window_seconds += seconds
        while self._window_min_q and self._window_min_q[-1] > inst:
            self._window_min_q.pop()
        self._window_min_q.append(inst)
        while self._window_max_q and self._window_max_q[-1] < inst:
            self._window_max_q.pop()
        self._window_max_q.append(inst)
        self._last_inst = inst
        self._last_tokens = tokens
        self._last_seconds = seconds
        self._updates += 1
        self.total_tokens += tokens
        self.total_seconds += seconds
        if not self.ready:
            self.value = inst
            self.ready = True
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * inst
        return self.value

    def snapshot(self) -> ThroughputSnapshot:
        window_updates = len(self._history)
        window_avg = (
            self._window_tokens / self._window_seconds
            if self._window_seconds > 0
            else 0.0
        )
        window_min = self._window_min_q[0] if window_updates else 0.0
        window_max = self._window_max_q[0] if window_updates else 0.0
        lifetime_avg = (
            self.total_tokens / self.total_seconds
            if self.total_seconds > 0
            else 0.0
        )
        ema_val = self.value if self.ready else 0.0
        return ThroughputSnapshot(
            ema=ema_val,
            instant=self._last_inst,
            window_avg=window_avg,
            window_min=window_min,
            window_max=window_max,
            window_tokens=self._window_tokens,
            window_seconds=self._window_seconds,
            window_updates=window_updates,
            instant_tokens=self._last_tokens,
            instant_seconds=self._last_seconds,
            lifetime_avg=lifetime_avg,
            lifetime_tokens=self.total_tokens,
            lifetime_seconds=self.total_seconds,
            samples=self._updates,
        )

    def stats(self, as_dict: bool = True):
        snap = self.snapshot()
        return snap.as_dict() if as_dict else snap

    def reset(self):
        self.ready = False
        self.value = 0.0
        self._history.clear()
        self._window_tokens = 0.0
        self._window_seconds = 0.0
        self._window_min_q.clear()
        self._window_max_q.clear()
        self._last_inst = 0.0
        self._last_tokens = 0.0
        self._last_seconds = 0.0
        self._updates = 0
        self.total_tokens = 0.0
        self.total_seconds = 0.0


class MPSTurboGovernor:
    """Adaptive throughput governor targeting >2k tok/s on MPS."""

    def __init__(
        self,
        trainer,
        *,
        target_tok: float,
        window: int,
        cooldown: int,
        micro_floor: int,
        seq_floor: int,
        seq_step: int,
        disable_metrics_ratio: float,
    ):
        self.trainer = trainer
        self.target = max(0.0, float(target_tok))
        self.window = max(1, int(window))
        self.cooldown = max(1, int(cooldown))
        self.micro_floor = max(1, int(micro_floor))
        self.seq_floor = max(128, int(seq_floor))
        self.seq_step = max(32, int(seq_step))
        self.disable_metrics_ratio = float(min(max(disable_metrics_ratio, 0.0), 0.99))
        base_candidates = [
            int(getattr(trainer.cfg, "max_len", 0) or 0),
            int(getattr(trainer, "_token_sequence", 0) or 0),
        ]
        self.base_seq = max([c for c in base_candidates if c > 0] or [2048])
        self.enabled = self.target > 0.0 and getattr(trainer.device, "type", "cpu") == "mps"
        self.low_strike = 0
        self.high_strike = 0
        self.last_adjust = -10**9
        self.best = 0.0
        self.history: Deque[float] = deque(maxlen=self.window)

    def _shed_metrics(self, step: int, ema: float, low_bar: float):
        tr = self.trainer
        if not getattr(tr, "_turbo_metrics_suppressed", False):
            print(
                f"[TURBO] shedding auxiliary metrics (ema={ema:.1f} < {low_bar:.1f} tok/s)"
            )
        tr._turbo_metrics_suppressed = True
        tr._turbo_metrics_release_step = step + self.cooldown
        tr._turbo_disable_kbridge = True

    def _restore_metrics(self):
        tr = self.trainer
        if getattr(tr, "_turbo_metrics_suppressed", False):
            print("[TURBO] auxiliary metrics restored")
        tr._turbo_metrics_suppressed = False
        tr._turbo_disable_kbridge = False

    def update(
        self,
        step: int,
        instant_tps: float,
        ema_tps: float,
        stats: ThroughputSnapshot,
        total_tok: int,
        chunk_hint: float,
    ):
        if not self.enabled:
            return
        chunk_hint_val = float(chunk_hint) if chunk_hint is not None else float(total_tok)
        window_peak = getattr(stats, "window_max", float(instant_tps))
        self.best = max(self.best, float(max(window_peak, instant_tps)))
        self.history.append(float(ema_tps))
        low_bar = self.target * max(0.25, self.disable_metrics_ratio)
        high_bar = self.target * 0.97 if self.target > 0 else float("inf")

        if ema_tps < low_bar:
            self.low_strike += 1
            self.high_strike = 0
            self._shed_metrics(step, ema_tps, low_bar)
        elif (
            getattr(self.trainer, "_turbo_metrics_suppressed", False)
            and ema_tps >= self.target * 0.9
            and step >= getattr(self.trainer, "_turbo_metrics_release_step", 0)
        ):
            self._restore_metrics()
            self.low_strike = 0
            self.high_strike += 1
        elif ema_tps > high_bar:
            self.low_strike = 0
            self.high_strike += 1
        else:
            self.low_strike = 0
            self.high_strike = 0

        if (
            self.low_strike >= self.window
            and (step - self.last_adjust) >= self.cooldown
        ):
            tr = self.trainer
            current_micro = max(1, int(getattr(tr, "_micro_active", 1)))
            if current_micro > max(self.micro_floor, int(getattr(tr, "_micro_base", 1)) // 2):
                new_micro = max(self.micro_floor, current_micro - 1)
                if new_micro < current_micro:
                    tr._micro_active = new_micro
                    tr._micro_stable = 0
                    print(
                        f"[TURBO] micro_batch tightened to {new_micro} (ema={ema_tps:.1f} tok/s, target={self.target:.1f}, chunk≈{chunk_hint_val:.0f})"
                    )
                    self.last_adjust = step
                    return
            base_seq = max(self.base_seq, int(getattr(tr.cfg, "max_len", self.base_seq)))
            current_cap = int(getattr(tr, "_turbo_max_len", 0) or base_seq)
            new_cap = max(self.seq_floor, current_cap - self.seq_step)
            if new_cap < current_cap:
                tr._turbo_set_sequence_cap(new_cap)
                self.last_adjust = step
                return

        if (
            self.high_strike >= self.window
            and (step - self.last_adjust) >= self.cooldown
        ):
            tr = self.trainer
            changed = False
            if getattr(tr, "_turbo_max_len", 0):
                changed = tr._turbo_relax_sequence_cap(self.seq_step)
            if not changed and int(getattr(tr, "_micro_active", 1)) < int(
                getattr(tr, "_micro_base", 1)
            ):
                tr._micro_active = min(
                    int(tr._micro_base), int(tr._micro_active) + 1
                )
                tr._micro_stable = 0
                print(
                    f"[TURBO] micro_batch relaxed to {tr._micro_active} (ema={ema_tps:.1f} tok/s, chunk≈{chunk_hint_val:.0f})"
                )
                changed = True
            if changed:
                self.last_adjust = step


class EMAMeter:
    def __init__(self, beta: float = 0.95, clamp: Optional[Tuple[float, float]] = None):
        self.beta = float(beta)
        self.clamp = clamp
        self.value = 0.0
        self.ready = False

    def update(self, val: float) -> float:
        if not math.isfinite(val):
            return self.value if self.ready else 0.0
        if self.clamp is not None:
            lo, hi = self.clamp
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
        if not self.ready:
            self.value = val
            self.ready = True
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * val
        return self.value

    def reset(self):
        self.ready = False
        self.value = 0.0


class ChronoScheduler:
    def __init__(self, opt: torch.optim.Optimizer, cfg: TrainConfig, total_steps: int):
        self.opt = opt
        self.cfg = cfg
        self.total = total_steps
        self.step_id = 0
        self.base_lr = cfg.lr
        self.warm = cfg.warmup_steps
        self.last_mult = 0.0
        self._setlr(0.0)

    def _setlr(self, mult: float):
        for g in self.opt.param_groups:
            g["lr"] = self.base_lr * mult

    def step(self):
        self.step_id += 1
        s = self.step_id
        if s <= self.warm:
            m = s / max(1, self.warm)
        else:
            t = (s - self.warm) / max(1, self.total - self.warm)
            m = 0.5 * (1 + math.cos(math.pi * t))
        self.last_mult = m
        self._setlr(m)
        return m


# ====== Augment (light) ======================================================
class PsyAugment:
    def __init__(self, tok: ByteTokenizer):
        self.tok = tok
        self.token_dropout = 0.0
        self.byte_noise = 0.0
        self.span_mask_prob = 0.0
        self.span_len = 8
        self._gen = torch.Generator(device="cpu")
        seed_env = _aos.environ.get("AETHER_PSY_SEED")
        try:
            if seed_env is not None:
                self._gen.manual_seed(int(seed_env))
            else:
                self._gen.seed()
        except Exception:
            self._gen.seed()

    def update(self, token_dropout=0.0, byte_noise=0.0, span_mask_prob=0.0, span_len=8):
        self.token_dropout = float(token_dropout)
        self.byte_noise = float(byte_noise)
        self.span_mask_prob = float(span_mask_prob)
        self.span_len = int(span_len)

    @torch.no_grad()
    def apply_input(self, x: torch.Tensor, pad_id: int):
        if (
            self.token_dropout <= 0
            and self.byte_noise <= 0
            and self.span_mask_prob <= 0
        ):
            return x
        B, T = x.shape
        x = x.clone()
        if self.token_dropout > 0:
            if self._gen.device == x.device:
                mask = torch.rand(
                    x.shape, device=x.device, dtype=torch.float32, generator=self._gen
                ) < self.token_dropout
            else:
                mask = torch.rand_like(x, dtype=torch.float32) < self.token_dropout
            x.masked_fill_(mask, pad_id)
        # byte_noise / span_mask は必要なら追加
        return x


# ====== PEFT: LoRA (optional) ===============================================
PEFT_AVAILABLE = False
try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType

    PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    PeftModel = None  # type: ignore


def apply_peft_lora(
    model: nn.Module,
    r: int = _AETHER_DEFAULT_LORA_R,
    alpha: int = 320,
    dropout: float = 0.05,
    targets: Optional[List[str]] = None,
) -> nn.Module:
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft not installed. `pip install peft`")
    if targets is None:
        targets = ["qkv", "proj"]  # unified projection
    conf = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, conf)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def save_peft_adapter_if_any(model: nn.Module, out_dir: str):
    if PEFT_AVAILABLE and isinstance(model, PeftModel):
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        print(f"[PEFT] saved adapter to: {out_dir}")


def merge_peft_for_inference(model: nn.Module) -> nn.Module:
    if PEFT_AVAILABLE and isinstance(model, PeftModel):
        model = model.merge_and_unload()
        print("[PEFT] merged and unloaded adapters.")
    return model


def apply_hybrid_lora(
    model: nn.Module,
    peft_targets: List[str],
    int8_include: Optional[List[str]],
    r: int = 160,
    alpha: int = 320,
    dropout: float = 0.05,
) -> nn.Module:
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft not installed for hybrid mode")
    model = apply_peft_lora(
        model, r=r, alpha=alpha, dropout=dropout, targets=peft_targets
    )
    if int8_include:
        n = convert_linear_to_int8_lora(
            model.base_model,
            r=r,
            alpha=alpha,
            dropout=0.0,
            include_names=int8_include,
            exclude_names=("emb", "head"),
            skip_if_out_equals=getattr(model.base_model, "vocab_size", None),
        )
        print(f"[HYBRID] INT8+LoRA injected: {n}")
    return model


# ====== CPU AdamW (8bit-ish) ===============================================
class CPUAdamW8(torch.optim.Optimizer):
    """m,v を CPU に保持。v を int8 量子化（対数尺度）して更新コストと常駐RAMを軽量化。"""

    def __init__(
        self,
        params,
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        quantize_v=True,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.quantize_v = bool(quantize_v)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd and wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                g = p.grad.detach()
                if not torch.isfinite(g).all():
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                g_cpu = g.to("cpu", dtype=torch.float16, non_blocking=False)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        g_cpu, memory_format=torch.preserve_format
                    )
                    if self.quantize_v:
                        state["v_q"] = torch.zeros_like(g_cpu, dtype=torch.int8)
                        state["v_s"] = torch.ones((), dtype=torch.float32)
                    else:
                        state["v"] = torch.zeros_like(
                            g_cpu, memory_format=torch.preserve_format
                        )

                m = state["m"]
                if self.quantize_v:
                    v_q = state["v_q"]
                    v_s = state["v_s"]
                    v = v_q.float() * v_s
                else:
                    v = state["v"]

                state["step"] += 1
                t = state["step"]
                m.mul_(beta1).add_(g_cpu, alpha=(1.0 - beta1))
                v.mul_(beta2).addcmul_(g_cpu, g_cpu, value=(1.0 - beta2))

                if self.quantize_v:
                    vmax = v.abs().amax()
                    v_s = (vmax / 127.0).clamp_min(1e-6)
                    state["v_s"] = v_s
                    v_q.copy_((v / v_s).clamp_(-127, 127).round_().to(torch.int8))

                bc1 = 1 - beta1**t
                bc2 = 1 - beta2**t
                mhat = m / bc1
                vhat = v / bc2
                upd = (mhat / (vhat.sqrt() + eps)).to(torch.float16)
                p.data.add_(-lr * upd.to(device=p.data.device, dtype=p.data.dtype))
        return loss


# ====== Teacher / LVI / Intention loss ======================================
def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).clamp(-1.0, 1.0)


class _SimpleTeacher(nn.Module):
    """固定写像: 2-gram ハッシュ → Embedding → Lin(D)"""

    def __init__(self, d_model: int, buckets: int = 65536, proj_dim: int = 256):
        super().__init__()
        self.table = nn.Embedding(buckets, proj_dim)
        with torch.no_grad():
            nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        self.proj = nn.Linear(proj_dim, d_model, bias=False)
        for p in self.parameters():
            p.requires_grad = False
        self.buckets = int(buckets)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        device = ids.device
        h = (
            ids[:, :-1].long() * 1315423911 + ids[:, 1:].long() * 2654435761
        ) % self.buckets
        if h.numel() == 0:
            h = torch.zeros(B, 1, dtype=torch.long, device=device)
        feat = self.table(h).mean(dim=1)
        return self.proj(feat)


class _LVIEngine(nn.Module):
    """軽量 LVI: teacherベクトルを微量 bias として注入"""

    def __init__(self, d_model: int, scale: float = 0.02):
        super().__init__()
        self.teacher = _SimpleTeacher(d_model)
        self.scale = float(scale)
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, ids: torch.Tensor):
        vec = self.teacher(ids)  # (B,D)
        bias = vec * self.scale
        logs = {
            "mv_reg": ids.new_tensor(0.0, dtype=torch.float32),
            "two_view": ids.new_tensor(0.0, dtype=torch.float32),
        }
        return bias, logs


def _make_negative_ids(ids: torch.Tensor) -> torch.Tensor:
    if ids.numel() == 0:
        return ids
    neg = ids.clone()
    core = neg[:, 1:].clone()
    if core.size(1) > 1:
        core = torch.roll(core, shifts=1, dims=1)
    neg[:, 1:] = core
    return neg


def _contrastive_intent_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    emb_weight: torch.Tensor,
    pad_id: int,
    t_pos: Optional[torch.Tensor],
    t_neg: Optional[torch.Tensor],
    margin: float = 0.10,
    sample_frac: float = 0.25,
) -> torch.Tensor:
    B, T, V = logits.shape
    if sample_frac < 1.0:
        stride = max(1, int(1.0 / max(1e-6, sample_frac)))
        idx = torch.arange(0, T, device=logits.device, step=stride)
        logits = logits[:, idx, :]
        targets = targets[:, idx]
    mask = (targets != pad_id).float()
    if mask.sum() <= 0:
        return logits.new_tensor(0.0)

    with torch.no_grad():
        maxv = logits.amax(dim=-1, keepdim=True)
        zero_nan_(maxv)
    probs = torch.softmax((logits - maxv).to(torch.float32), dim=-1).clamp_min(1e-6)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    E = emb_weight.to(probs.dtype)
    E_pred = torch.einsum("btv,vd->btd", probs, E)

    if t_pos is not None:
        pos = t_pos.to(E_pred.dtype).unsqueeze(1).expand_as(E_pred)
        cos_pos = _safe_cos(E_pred, pos).mean()
    else:
        gt = targets.view(-1).clamp_min(0)
        E_gt = F.embedding(gt, E).view_as(E_pred)
        cos_pos = _safe_cos(E_pred, E_gt).mean()

    loss = (1.0 - cos_pos).mean()
    if t_neg is not None:
        neg = t_neg.to(E_pred.dtype).unsqueeze(1).expand_as(E_pred)
        cos_neg = _safe_cos(E_pred, neg).mean()
        loss = loss + F.relu(cos_neg - cos_pos + float(margin)).mean()
    return loss


# ====== Trainer (MPS/Fabric) =============================================
class AetherTrainerBase:
    def __init__(self, model: AetherPumpSimple, tok: ByteTokenizer, cfg: TrainConfig):
        self.model = model
        self.tok = tok
        self.cfg = cfg
        self.device = detect_device()

        self._mps7b_report = _auto_tune_for_mps_7b(self.model, self.cfg, self.device)
        self._token_budget = 0
        self._token_chunk = 0
        self._token_budget_target = 0
        self._token_sequence = 0
        token_report = (
            self._mps7b_report.get("token_budget")
            if isinstance(self._mps7b_report, dict)
            else None
        )
        if isinstance(token_report, dict):
            self._token_budget = int(token_report.get("effective", 0) or 0)
            self._token_chunk = int(token_report.get("chunk_tokens", 0) or 0)
            self._token_budget_target = int(token_report.get("target", 0) or 0)
            self._token_sequence = int(token_report.get("sequence", 0) or 0)
            if self._token_budget > 0:
                seq = token_report.get("sequence")
                chunk = token_report.get("chunk_size")
                micro = token_report.get("micro")
                msg = (
                    f"[MPS][7B] token budget≈{self._token_budget:,}/{max(self._token_budget_target, 1):,}"
                )
                detail = []
                if self._token_chunk > 0:
                    detail.append(f"chunk≈{self._token_chunk:,}")
                if chunk and seq:
                    detail.append(f"layout={int(chunk)}×{int(seq)}")
                if micro:
                    detail.append(f"micro={int(micro)}")
                if detail:
                    msg += " (" + ", ".join(detail) + ")"
                print(msg)

        matmul_mode = getattr(self.cfg, "matmul_precision", None)
        if matmul_mode:
            try:
                torch.set_float32_matmul_precision(str(matmul_mode))
                print(f"[AMP] matmul precision set to {matmul_mode}")
            except Exception as e:
                print("[AMP] matmul precision set failed:", e)

        self.model.to(self.device)

        if getattr(self.cfg, "compile", False):
            backend = getattr(self.cfg, "compile_backend", "aot_eager")
            dynamic = bool(getattr(self.cfg, "compile_dynamic", False))
            try:
                self.model = torch.compile(
                    self.model, backend=str(backend), dynamic=dynamic
                )
                print(
                    f"[COMPILE] torch.compile enabled (backend={backend}, dynamic={dynamic})"
                )
            except Exception as e:
                print("[COMPILE] torch.compile failed:", e)
        enable_tiled_sdpa(cfg.tiled_q, cfg.tiled_k, compute_in_fp32=True)
        try:
            set_sliding_window(
                int(cfg.window_size), int(cfg.global_tokens), int(cfg.global_stride)
            )
        except Exception:
            pass
        self.fabric = FabricLite(cfg.out_dir, use_tb=cfg.use_tb)
        self.psy = PsyAugment(tok)
        self._global_step = 0
        beta_env = os.environ.get("AETHER_TPS_EMA_BETA", "")
        tps_beta = 0.90
        if beta_env:
            try:
                tps_beta = float(beta_env)
            except Exception:
                print(
                    f"[THROUGHPUT] Invalid AETHER_TPS_EMA_BETA={beta_env!r}; using {tps_beta}"
                )
        tps_window = int(getattr(self.cfg, "tps_window", 8) or 8)
        window_env = os.environ.get("AETHER_TPS_WINDOW", "")
        if window_env:
            try:
                tps_window = max(1, int(window_env))
            except Exception:
                print(
                    f"[THROUGHPUT] Invalid AETHER_TPS_WINDOW={window_env!r}; using {tps_window}"
                )
        self._tps_meter = ThroughputMeter(beta=tps_beta, window=tps_window)
        self._last_tok_per_sec = 0.0
        self._last_tok_per_sec_ema = 0.0
        ppl_beta = float(getattr(self.cfg, "ppl_smooth_beta", 0.90))
        self._ppl_cap = float(getattr(self.cfg, "ppl_cap", 0.0))
        clamp_hi = self._ppl_cap if self._ppl_cap > 0 else None
        self._loss_meter = EMAMeter(beta=ppl_beta, clamp=(0.0, clamp_hi))
        self._loss_guard = float(getattr(self.cfg, "loss_guard", 0.0))
        self._tps_target = float(getattr(self.cfg, "tps_target", 0.0))
        self._tps_patience = max(1, int(getattr(self.cfg, "tps_boost_patience", 24)))
        self._tps_underperf = 0
        try:
            spike_patience = int(_aos.environ.get("AETHER_LOSS_SPIKE_PATIENCE", "4"))
        except Exception:
            spike_patience = 4
        self._loss_spike_count = 0
        self._loss_spike_patience = max(1, spike_patience)

        wd = 0.01

        self._strict_loss_guard = os.environ.get("AETHER_STRICT_LOSS_GUARD", "1") != "0"
        self._strict_activation_guard = (
            os.environ.get("AETHER_STRICT_ACT_GUARD", "0") != "0"
        )
        self._suppress_numeric_traceback = (
            os.environ.get("AETHER_SUPPRESS_NUMERIC_TRACEBACK", "1") != "0"
        )
        self._nonfinite_event_verbose = (
            os.environ.get("AETHER_NONFINITE_VERBOSE", "1") != "0"
        )

        # ultramem optimizer factory (if autopatch installed)
        if (up is not None) and hasattr(self.model, "_ultramem_make_optimizer"):
            self.opt = self.model._ultramem_make_optimizer(lr=self.cfg.lr, wd=wd)
            print("[OPT] ultramem optimizer factory used")
        else:
            if getattr(self.cfg, "opt_cpu8bit", False):
                self.opt = CPUAdamW8(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=cfg.lr,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=wd,
                    quantize_v=True,
                )
                print("[OPT] CPUAdamW8 (v:int8, m:fp16)")
            else:
                self.opt = (
                    GaLoreAdamW(
                        [p for p in self.model.parameters() if p.requires_grad],
                        lr=cfg.lr,
                        betas=(0.9, 0.95),
                        eps=1e-8,
                        weight_decay=wd,
                        rank=int(getattr(cfg, "galore_rank", 64)),
                    )
                    if getattr(cfg, "opt_galore", False)
                    else torch.optim.AdamW(
                        [p for p in self.model.parameters() if p.requires_grad],
                        lr=cfg.lr,
                        betas=(0.9, 0.95),
                        eps=1e-8,
                        weight_decay=wd,
                    )
                )
                print("[OPT] AdamW on MPS")
        self.optimizer = self.opt

        if self.device.type == "mps" and getattr(self.cfg, "prefer_bfloat16", True):
            try:
                torch.zeros(1, device=self.device, dtype=torch.bfloat16)
                self.amp_dtype = torch.bfloat16
                print("[AMP] Using bfloat16 autocast on MPS")
            except Exception:
                self.amp_dtype = torch.float16
                print("[AMP] Falling back to float16 autocast")
        else:
            self.amp_dtype = torch.float16

        if not hasattr(self, "amp_dtype"):
            self.amp_dtype = torch.float16

        self._scaler_enabled = self.device.type == "mps" and bool(
            getattr(self.cfg, "grad_scaler", True)
        ) and self.amp_dtype == torch.float16
        self.scaler = GradScaler() if self._scaler_enabled else None
        if self.scaler is not None:
            print("[AMP] GradScaler(mps) enabled")
        elif (
            self.device.type == "mps"
            and bool(getattr(self.cfg, "grad_scaler", True))
            and self.amp_dtype == torch.bfloat16
        ):
            print("[AMP] GradScaler disabled for bfloat16 autocast")

        self._prefetch_to_device = (
            self.device.type == "mps"
            and bool(getattr(self.cfg, "prefetch_to_device", True))
        )
        if self._prefetch_to_device:
            print("[MPS] host batch prefetch enabled")

        self._prefetch_stream = None
        if self.device.type == "mps" and int(
            os.environ.get("AETHER_MPS_DISABLE_PREFETCH_STREAM", "0")
        ) != 1:
            try:
                stream_ctor = getattr(torch.mps, "Stream", None)
                stream_ctx = getattr(torch.mps, "stream", None)
                if stream_ctor is not None and stream_ctx is not None:
                    self._prefetch_stream = stream_ctor()
                    print("[MPS] dedicated copy stream ready")
            except Exception as e:
                print("[MPS] copy stream unavailable:", e)
                self._prefetch_stream = None

        if self.device.type == "mps" and bool(
            getattr(self.cfg, "disallow_mps_fallback", True)
        ):
            try:
                import torch.backends.mps as _mps_backend

                _mps_backend.fallback_allow_all(False)
                print("[MPS] CPU fallback disabled (strict MPS kernels)")
            except Exception as e:
                print("[MPS] fallback control unavailable:", e)

        self._teacher = None
        if (self.cfg.intent_weight > 0.0) or bool(
            getattr(self.cfg, "lvi_enable", False)
        ):
            try:
                self._teacher = _SimpleTeacher(getattr(self.model, "d_model", 256)).to(
                    self.device
                )
                print("[INTENT] SimpleTeacher enabled")
            except Exception:
                self._teacher = None

        self._use_lvi = bool(getattr(self.cfg, "lvi_enable", False))
        self._lvi = (
            _LVIEngine(self.model.d_model, scale=0.02).to(self.device)
            if self._use_lvi
            else None
        )
        self._lvi_cache = {"bias": None, "logs": None, "step": -1}
        # ANI-AI controller (opt-in via env)
        self._ai = _AetherNumpyAIController(self)
        # Adaptive micro-batch & memory controls
        self._adaptive_micro = bool(getattr(self.cfg, "adaptive_microbatch", True))
        self._micro_active = max(1, int(getattr(self.cfg, "micro_batch", 1)))
        self._micro_base = self._micro_active
        self._micro_cap = max(
            self._micro_active,
            int(getattr(self.cfg, "adaptive_micro_max", self._micro_active)),
        )
        self._micro_recover_every = max(
            0, int(getattr(self.cfg, "adaptive_micro_recover", 256))
        )
        self._micro_stable = 0
        self._oom_retry_limit = max(0, int(getattr(self.cfg, "oom_retries", 3)))
        self._empty_cache_every = max(0, int(getattr(self.cfg, "empty_cache_every", 0)))
        if self._adaptive_micro and self._micro_cap > self._micro_base:
            print(
                f"[ADAPT] micro_batch base={self._micro_base}, cap={self._micro_cap}"
            )
        self._micro_floor = max(1, int(getattr(self.cfg, "turbo_micro_floor", 1)))
        if self._micro_active < self._micro_floor:
            self._micro_active = self._micro_floor
        if self._micro_base < self._micro_floor:
            self._micro_base = self._micro_floor
        if self._micro_cap < self._micro_floor:
            self._micro_cap = self._micro_floor
        self._turbo_seq_step = max(32, int(getattr(self.cfg, "turbo_seq_step", 256)))
        self._turbo_seq_floor = max(128, int(getattr(self.cfg, "turbo_seq_floor", 512)))
        base_seq_candidates = [
            int(getattr(self.cfg, "max_len", 0) or 0),
            int(self._token_sequence or 0),
        ]
        self._turbo_base_seq = max([c for c in base_seq_candidates if c > 0] or [self.cfg.max_len])
        self._turbo_max_len = 0
        self._turbo_metrics_suppressed = False
        self._turbo_metrics_release_step = 0
        self._turbo_disable_kbridge = False
        self._turbo = MPSTurboGovernor(
            self,
            target_tok=float(getattr(self.cfg, "turbo_target_tok", 0.0)),
            window=int(getattr(self.cfg, "turbo_window", 6)),
            cooldown=int(getattr(self.cfg, "turbo_cooldown", 48)),
            micro_floor=self._micro_floor,
            seq_floor=self._turbo_seq_floor,
            seq_step=self._turbo_seq_step,
            disable_metrics_ratio=float(
                getattr(self.cfg, "turbo_disable_metrics_ratio", 0.75)
            ),
        )
        # --- k-bridge integration knobs (safe opt-in; default on if installed) ---
        requested_k = bool(
            int(os.environ.get("KBRIDGE_ENABLE", "1" if _KBRIDGE_AVAILABLE else "0"))
        )
        self._k_enabled = (
            requested_k and _KBRIDGE_AVAILABLE and _AETHER_NUMPY_AVAILABLE
        )
        if requested_k and not self._k_enabled:
            reason = (
                "numpy unavailable"
                if not _AETHER_NUMPY_AVAILABLE
                else "kbridge package not available"
            )
            print(f"[KBRIDGE] disabled ({reason})")
        self._k_reg_w = float(os.environ.get("KBRIDGE_REG_W", "0.0"))
        self._kb_buf_cap = int(os.environ.get("KBRIDGE_BUF_CAP", "200000"))
        self._kb_ece_bins = int(os.environ.get("KBRIDGE_ECE_BINS", "15"))
        self._kb_metrics_every = int(
            os.environ.get(
                "KBRIDGE_METRICS_EVERY", str(max(1, getattr(cfg, "log_every", 10)))
            )
        )
        self._kb_collect_every = max(
            1,
            int(
                os.environ.get(
                    "KBRIDGE_COLLECT_EVERY",
                    str(max(1, getattr(cfg, "log_every", 10))),
                )
            ),
        )
        # classwise ECE config & length buckets
        self._kb_ece_by = (
            os.environ.get("KBRIDGE_ECE_BY", "pred").strip().lower()
        )  # "pred" | "true" | "predgrp" | "truegrp"
        self._kb_ece_c_top = int(os.environ.get("KBRIDGE_ECE_C_TOP", "5"))
        self._kb_ece_c_min = int(os.environ.get("KBRIDGE_ECE_C_MIN", "1000"))
        self._kb_ece_c_list = [
            int(x)
            for x in os.environ.get("KBRIDGE_ECE_C_LIST", "").split(",")
            if x.strip().isdigit()
        ]
        if self._k_enabled:
            np_mod = _require_numpy("knowledge-bridge configuration")
            _len_edges_env = os.environ.get(
                "KBRIDGE_LEN_BUCKETS", "0,128,512,2048,1000000"
            )
            try:
                _edges = [int(x) for x in _len_edges_env.split(",") if x.strip()]
                _edges = sorted(set([x for x in _edges if x >= 0]))
                if len(_edges) < 2:
                    _edges = [0, 1_000_000]
                self._kb_len_edges = np_mod.asarray(_edges, dtype=np_mod.int64)
            except Exception:
                self._kb_len_edges = np_mod.asarray([0, 1_000_000], dtype=np_mod.int64)
            # token-level buffers
            self._kb_buf_pmax = []
            self._kb_buf_labels = []
            self._kb_buf_predcls = []
            self._kb_buf_truecls = []
            self._kb_buf_lenbin = []
            # sequence-level NDCG buffer
            self._kb_seq_scores = []
            self._kb_seq_gains = []
            # class groups (built from vocab & scheme or custom json)
            self._kb_class_scheme = (
                os.environ.get("KBRIDGE_CLASSMAP", "byte-basic").strip().lower()
            )
            self._kb_class_json = os.environ.get("KBRIDGE_CLASSMAP_FILE", "").strip()
            try:
                self._kb_classmap, self._kb_group_names = _build_classmap(
                    vocab_size=getattr(self.model, "vocab_size", 32000),
                    scheme=self._kb_class_scheme,
                    json_path=self._kb_class_json,
                )
                self._kb_group_count = int(self._kb_classmap.max()) + 1
            except Exception:
                self._kb_classmap = np_mod.zeros(
                    (getattr(self.model, "vocab_size", 32000),), dtype=np_mod.int32
                )
                self._kb_group_names = []
                self._kb_group_count = int(self._kb_classmap.max()) + 1
        else:
            self._kb_len_edges = None
            self._kb_buf_pmax = []
            self._kb_buf_labels = []
            self._kb_buf_predcls = []
            self._kb_buf_truecls = []
            self._kb_buf_lenbin = []
            self._kb_seq_scores = []
            self._kb_seq_gains = []
            self._kb_class_scheme = os.environ.get("KBRIDGE_CLASSMAP", "byte-basic").strip().lower()
            self._kb_class_json = os.environ.get("KBRIDGE_CLASSMAP_FILE", "").strip()
            self._kb_classmap = None
            self._kb_group_names = []
            self._kb_group_count = 0
        # 2D ECE（len×class）や T sweep/quantile bins
        self._kb_enable_2d = bool(int(os.environ.get("KBRIDGE_ECE_2D", "0")))
        self._kb_2d_max_cells = int(os.environ.get("KBRIDGE_2D_MAX_CELLS", "24"))
        self._kb_tsweep = [
            float(x)
            for x in os.environ.get("KBRIDGE_TSWEEP", "").split(",")
            if x.strip()
        ]

    def _device_async_copy(self, *batch):
        if not batch:
            return tuple(), None
        if self.device.type != "mps":
            return tuple(t.to(self.device) for t in batch), None
        stream = self._prefetch_stream
        if stream is None:
            return tuple(t.to(self.device, non_blocking=True) for t in batch), None
        stream_ctx = getattr(torch.mps, "stream", None)
        if stream_ctx is None:
            return tuple(t.to(self.device, non_blocking=True) for t in batch), None
        with stream_ctx(stream):
            copied = tuple(t.to(self.device, non_blocking=True) for t in batch)
        return copied, stream

    def _wait_stream(self, stream):
        if stream is None or self.device.type != "mps":
            return
        current_fn = getattr(torch.mps, "current_stream", None)
        if callable(current_fn):
            try:
                current_fn().wait_stream(stream)
                return
            except Exception:
                pass
        sync_fn = getattr(torch.mps, "synchronize", None)
        if callable(sync_fn):
            try:
                sync_fn()
            except Exception:
                pass

    def _device_sync_copy(self, *batch):
        tensors, stream = self._device_async_copy(*batch)
        self._wait_stream(stream)
        return tensors

    def _turbo_set_sequence_cap(self, new_cap: int):
        base = max(1, int(self._turbo_base_seq))
        new_cap = int(new_cap)
        if new_cap <= 0 or new_cap >= base:
            if self._turbo_max_len != 0:
                print(
                    f"[TURBO] sequence cap cleared; full {base}-token context restored"
                )
            self._turbo_max_len = 0
            return
        if self._turbo_max_len == new_cap:
            return
        self._turbo_max_len = new_cap
        print(f"[TURBO] sequence cap set to {new_cap} tokens (base={base})")

    def _turbo_relax_sequence_cap(self, step_size: int) -> bool:
        if self._turbo_max_len <= 0:
            return False
        base = max(1, int(self._turbo_base_seq))
        step_size = max(1, int(step_size))
        new_cap = min(base, self._turbo_max_len + step_size)
        if new_cap >= base:
            self._turbo_max_len = 0
            print(
                f"[TURBO] sequence cap cleared; full {base}-token context restored"
            )
            return True
        if new_cap != self._turbo_max_len:
            self._turbo_max_len = new_cap
            print(f"[TURBO] sequence cap relaxed to {new_cap}")
            return True
        return False

    def _turbo_update(
        self,
        step: int,
        instant_tps: float,
        ema_tps: float,
        stats: ThroughputSnapshot,
        total_tok: int,
        chunk_hint: float,
    ):
        if isinstance(getattr(self, "_turbo", None), MPSTurboGovernor):
            self._turbo.update(step, instant_tps, ema_tps, stats, total_tok, chunk_hint)

    def _stabilize_loss_value(self, loss_avg: float) -> float:
        guard = max(0.0, self._loss_guard)
        if not math.isfinite(loss_avg):
            print("[LOSS] non-finite mean detected; resetting scaler and clamping")
            if self.scaler is not None:
                try:
                    self.scaler = GradScaler()
                    print("[LOSS] GradScaler reset after NaN loss")
                except Exception:
                    pass
            loss_avg = guard if guard > 0 else 0.0
        if guard > 0 and loss_avg > guard:
            self._loss_spike_count += 1
            print(
                f"[LOSS] spike detected ({loss_avg:.4f}); clipped to guard {guard:.2f}"
            )
            reset_msg = "[LOSS] GradScaler reset after loss spike"
            repeated = False
            if self._loss_spike_count >= self._loss_spike_patience:
                repeated = True
                reset_msg = "[LOSS] GradScaler reset after repeated spikes"
            if self.scaler is not None:
                try:
                    self.scaler = GradScaler()
                    print(reset_msg)
                except Exception:
                    pass
            if repeated and getattr(self, "_adaptive_micro", False):
                if self._micro_active < self._micro_cap:
                    prev = self._micro_active
                    self._micro_active = min(
                        self._micro_cap, max(prev + 1, prev * 2)
                    )
                    print(
                        "[LOSS] repeated spikes -> micro_batch raised to"
                        f" {self._micro_active}"
                    )
                self._loss_spike_count = 0
            if hasattr(self, "_ai"):
                try:
                    self._ai.skip_next_step = True
                except Exception:
                    pass
            loss_avg = guard
        else:
            self._loss_spike_count = 0
        return loss_avg

    def _maybe_boost_throughput(self, ema_tps: float):
        target = max(0.0, self._tps_target)
        if target <= 0:
            return
        if ema_tps >= target * 0.97:
            self._tps_underperf = 0
            return
        self._tps_underperf += 1
        if self._tps_underperf < self._tps_patience:
            return
        self._tps_underperf = 0
        if not self._adaptive_micro:
            return
        if self._micro_active > self._micro_base:
            new_micro = max(self._micro_base, self._micro_active // 2)
            if new_micro < self._micro_active:
                self._micro_active = new_micro
                print(
                    f"[THROUGHPUT] micro_batch tightened to {self._micro_active} (ema={ema_tps:.1f} < target={target:.1f})"
                )

    def _rebuild_optimizer(self):
        wd = 0.01
        if getattr(self.cfg, "opt_cpu8bit", False):
            self.opt = CPUAdamW8(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.cfg.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=wd,
                quantize_v=True,
            )
            print("[OPT] CPUAdamW8 (v:int8, m:fp16)")
        else:
            self.opt = (
                GaLoreAdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=self.cfg.lr,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=wd,
                    rank=int(getattr(self.cfg, "galore_rank", 64)),
                )
                if getattr(self.cfg, "opt_galore", False)
                else torch.optim.AdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=self.cfg.lr,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=wd,
                )
            )
            print("[OPT] AdamW on MPS")
        self.optimizer = self.opt
        sched = getattr(self, "_active_scheduler", None)
        if sched is not None:
            try:
                sched.opt = self.opt
                sched._setlr(float(getattr(sched, "last_mult", 1.0)))
            except Exception:
                pass

    def _relora_cycle(self):
        if not PEFT_AVAILABLE or not isinstance(self.model, PeftModel):
            return False
        try:
            self.model = merge_peft_for_inference(self.model).to(self.device)
            targets = [
                s.strip()
                for s in str(getattr(self.cfg, "peft_targets", "qkv,proj")).split(",")
                if s.strip()
            ]
            self.model = apply_peft_lora(
                self.model,
                r=int(getattr(self.cfg, "lora_r", 160)),
                alpha=int(getattr(self.cfg, "lora_alpha", 320)),
                dropout=float(getattr(self.cfg, "lora_dropout", 0.05)),
                targets=targets,
            ).to(self.device)
            self._rebuild_optimizer()
            print("[ReLoRA] merged & re-applied adapters; optimizer rebuilt.")
            return True
        except Exception as e:
            print("[ReLoRA] failed:", e)
            return False

    def _make_loader(self, ds, shuffle, max_len_for_collate):
        is_iter = isinstance(ds, IterableDataset)
        num_workers = max(0, int(getattr(self.cfg, "loader_num_workers", 0)))
        persistent = (
            num_workers > 0
            and bool(getattr(self.cfg, "loader_persistent_workers", False))
        )
        pin_memory = bool(getattr(self.cfg, "loader_pin_memory", False))
        loader_kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=(False if is_iter else bool(shuffle)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            collate_fn=lambda b: collate_lm_safe(b, pad_id=self.tok.PAD),
        )
        prefetch_factor = getattr(self.cfg, "loader_prefetch_factor", None)
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        return DataLoader(
            ds,
            **loader_kwargs,
        )

    def _ce_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, pad_id: int
    ) -> torch.Tensor:
        B, T, V = logits.shape
        logits_fp = (
            logits.float() if getattr(self, "_ai_fp32_logits", False) else logits
        )
        loss = F.cross_entropy(
            logits_fp.view(B * T, V),
            targets.view(B * T),
            ignore_index=pad_id,
            reduction="mean",
        )
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_logits(self, ids: torch.Tensor):
        if self._use_lvi and self._lvi is not None:
            cur = self._global_step
            every = max(1, int(getattr(self.cfg, "lvi_every", 1)))
            do_lvi = (
                (every <= 1) or (cur % every == 0) or (self._lvi_cache["bias"] is None)
            )
            if do_lvi:
                bias, logs = self._lvi(ids.to(self.device))
                self._lvi_cache = {"bias": bias.detach(), "logs": logs, "step": cur}
            else:
                B, T = ids.shape
                bias = self._lvi_cache["bias"]
                if bias.size(0) != B:
                    bias = bias[:B]
                logs = self._lvi_cache["logs"]
            logits = self.model.forward_with_bias(
                ids, bias.to(ids.device, dtype=self.model.emb.weight.dtype)
            )
            return logits, logs
        else:
            logits = self.model(ids)
            return logits, {
                "mv_reg": torch.tensor(0.0, device=ids.device),
                "two_view": torch.tensor(0.0, device=ids.device),
            }

    # External hook to feed sequence-level eval (scores/gains per sequence)
    def kb_push_seq_eval(self, scores_row, gains_row):
        try:
            np_mod = _require_numpy("sequence evaluation cache")

            s = np_mod.asarray(scores_row, dtype=np_mod.float64).ravel()
            g = np_mod.asarray(gains_row, dtype=np_mod.float64).ravel()
            if s.shape == g.shape and s.size > 0:
                self._kb_seq_scores.append(s)
                self._kb_seq_gains.append(g)
        except Exception:
            pass

class AetherTrainerMPS(AetherTrainerBase):
    def __init__(self, *a, ckpt_every: int = 1, **k):
        opt_override = k.pop("opt", None)
        super().__init__(*a, **k)
        if opt_override is not None:
            try:
                self.opt = opt_override
            except Exception:
                self.opt = opt_override
            self.optimizer = self.opt
        if ckpt_every > 0:
            enable_gradient_checkpointing(self.model, every=ckpt_every)

    def fit(self, train_ds, val_ds: Optional[Dataset] = None):
        # Curriculum
        curr = (
            self.cfg.curriculum
            if self.cfg.curriculum
            else [
                CurriculumStage(
                    until_step=self.cfg.warmup_steps, max_len=min(self.cfg.max_len, 512)
                ),
                CurriculumStage(
                    until_step=int((self.cfg.max_steps or 10_000) * 0.5),
                    max_len=min(self.cfg.max_len, 1024),
                ),
                CurriculumStage(
                    until_step=(self.cfg.max_steps or 10_000), max_len=self.cfg.max_len
                ),
            ]
        )

        def curr_params(step):
            for st in curr:
                if step <= st.until_step:
                    return {"max_len": st.max_len}
            return {"max_len": curr[-1].max_len}

        # Loaders
        train_dl = self._make_loader(
            train_ds, shuffle=True, max_len_for_collate=curr[0].max_len
        )
        val_dl = (
            self._make_loader(
                val_ds, shuffle=False, max_len_for_collate=curr[0].max_len
            )
            if val_ds is not None
            else None
        )

        try:
            steps_per_epoch = len(train_dl)
        except Exception:
            steps_per_epoch = self.cfg.max_steps or 10**9
        total_steps = self.cfg.max_steps or int(self.cfg.epochs * steps_per_epoch)
        sched = ChronoScheduler(self.opt, self.cfg, total_steps=total_steps)
        self._active_scheduler = sched
        lr_mult = float(getattr(sched, "last_mult", 0.0))
        self._tps_meter.reset()
        self._loss_meter.reset()
        self._last_tok_per_sec = 0.0
        self._last_tok_per_sec_ema = 0.0

        pad_id = self.tok.PAD
        val_iter = iter(val_dl) if val_dl is not None else None

        def _schedule_batch(loader_iter, step_hint: int):
            try:
                bx, by = next(loader_iter)
            except StopIteration:
                return None
            params = curr_params(int(step_hint))
            max_len = int(params.get("max_len", bx.size(1)))
            if getattr(self, "_turbo_max_len", 0):
                max_len = min(max_len, int(self._turbo_max_len))
            if max_len < bx.size(1):
                bx = bx[:, :max_len]
                by = by[:, :max_len]
            self.psy.update(
                token_dropout=self.cfg.token_dropout,
                byte_noise=self.cfg.byte_noise,
                span_mask_prob=self.cfg.span_mask_prob,
                span_len=self.cfg.span_mask_len,
            )
            bx = self.psy.apply_input(bx, pad_id)
            device_payload = None
            stream_payload = None
            if self._prefetch_to_device:
                try:
                    tensors, stream_payload = self._device_async_copy(bx, by)
                    if tensors:
                        device_payload = tuple(tensors)
                except RuntimeError as e:
                    print("[MPS] host batch prefetch disabled after error:", e)
                    self._prefetch_to_device = False
                    self._prefetch_stream = None
                    device_payload = None
                    stream_payload = None
            return {
                "cpu": (bx, by),
                "device": device_payload,
                "stream": stream_payload,
            }

        def _run_validation_snapshot(eval_step: int):
            nonlocal val_iter
            if val_dl is None or val_iter is None:
                return
            val_steps = max(1, int(os.environ.get("AETHER_VAL_STEPS", "4")))
            was_training = self.model.training
            self.model.eval()
            total_loss_tensor = torch.zeros((), device=self.device, dtype=torch.float32)
            total_tok_tensor = torch.zeros((), device=self.device, dtype=torch.float32)
            try:
                with torch.no_grad():
                    for _ in range(val_steps):
                        try:
                            bx, by = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_dl)
                            try:
                                bx, by = next(val_iter)
                            except StopIteration:
                                break
                        params = curr_params(int(eval_step))
                        max_len = int(params.get("max_len", bx.size(1)))
                        if getattr(self, "_turbo_max_len", 0):
                            max_len = min(max_len, int(self._turbo_max_len))
                        if max_len < bx.size(1):
                            bx = bx[:, :max_len]
                            by = by[:, :max_len]
                        B = bx.size(0)
                        idx = 0
                        current_micro = max(1, int(self._micro_active))
                        chunk_bs = max(1, math.ceil(B / current_micro))
                        while idx < B:
                            end = min(B, idx + chunk_bs)
                            subx, suby = self._device_sync_copy(bx[idx:end], by[idx:end])
                            with torch.autocast(
                                device_type="mps", dtype=self.amp_dtype
                            ):
                                logits_eval, _ = self._compute_logits(subx)
                            with torch.autocast(device_type="mps", enabled=False):
                                loss_eval = self._ce_loss(logits_eval, suby, pad_id)
                            tok_eval = (suby != pad_id).sum().to(total_tok_tensor.dtype)
                            total_tok_tensor = total_tok_tensor + tok_eval
                            total_loss_tensor = total_loss_tensor + loss_eval.detach().to(
                                total_loss_tensor.dtype
                            ) * tok_eval
                            idx = end
                total_tok_eval = int(total_tok_tensor.item())
                if total_tok_eval > 0:
                    val_loss = float(total_loss_tensor.item()) / max(1, total_tok_eval)
                    val_ppl = math.exp(min(max(val_loss, 0.0), 20.0))
                    self.fabric.log(
                        {
                            "val/loss": float(val_loss),
                            "val/ppl": float(val_ppl),
                            "val/tokens": float(total_tok_eval),
                        },
                        step=eval_step,
                    )
            finally:
                self.model.train(was_training)

        self.model.train()
        t0 = time.time()
        step = self._global_step
        for epoch in range(self.cfg.epochs if self.cfg.max_steps is None else 10**9):
            loader_iter = iter(train_dl)
            pending = _schedule_batch(loader_iter, step + 1)
            while pending is not None:
                if self.cfg.max_steps is not None and step >= self.cfg.max_steps:
                    break
                next_pending = None
                if self.cfg.max_steps is None or (step + 1) < self.cfg.max_steps:
                    next_pending = _schedule_batch(loader_iter, step + 2)

                bx_cpu, by_cpu = pending["cpu"]

                B, T = bx_cpu.shape
                total_loss, total_tok = 0.0, 0
                accum = 0
                logits = None
                lvi_logs = {
                    "mv_reg": torch.tensor(0.0, device=self.device),
                    "two_view": torch.tensor(0.0, device=self.device),
                }
                oom_count = 0
                last_oom_err = None
                device_cache = pending["device"]
                device_stream = pending["stream"]
                stream_waited = False

                while True:
                    total_loss = 0.0
                    total_tok = 0
                    accum = 0
                    idx = 0
                    total_loss_tensor = torch.zeros(
                        (), device=self.device, dtype=torch.float32
                    )
                    total_tok_tensor = torch.zeros(
                        (), device=self.device, dtype=torch.float32
                    )
                    restart_batch = False
                    batch_on_device = device_cache
                    batch_stream = None
                    if batch_on_device is not None and not stream_waited:
                        batch_stream = device_stream

                    if batch_on_device is None and self._prefetch_to_device:
                        try:
                            tensors, new_stream = self._device_async_copy(
                                bx_cpu, by_cpu
                            )
                            if tensors:
                                batch_on_device = tuple(tensors)
                                device_cache = batch_on_device
                                device_stream = new_stream
                                stream_waited = False
                                batch_stream = new_stream
                        except RuntimeError as e:
                            if self._maybe_handle_oom(e, B * T):
                                restart_batch = True
                                oom_count += 1
                                last_oom_err = e
                                self.opt.zero_grad(set_to_none=True)
                                if self._scaler_enabled:
                                    self.scaler = GradScaler()
                                    print("[OOM] GradScaler reset after OOM")
                            else:
                                print(
                                    "[MPS] host batch prefetch disabled after error:",
                                    e,
                                )
                                self._prefetch_to_device = False
                                self._prefetch_stream = None
                                batch_stream = None
                                device_cache = None
                                device_stream = None
                            batch_on_device = None

                    if restart_batch:
                        if self._oom_retry_limit and oom_count > self._oom_retry_limit:
                            raise last_oom_err
                        continue

                    if batch_on_device is not None and batch_stream is not None:
                        self._wait_stream(batch_stream)
                        stream_waited = True
                        batch_stream = None

                    while idx < B:
                        current_micro = max(1, int(self._micro_active))
                        chunk_bs = max(1, math.ceil(B / current_micro))
                        end = min(B, idx + chunk_bs)
                        if batch_on_device is not None:
                            subx = batch_on_device[0][idx:end]
                            suby = batch_on_device[1][idx:end]
                        else:
                            subx, suby = self._device_sync_copy(
                                bx_cpu[idx:end], by_cpu[idx:end]
                            )

                        # AI pre-forward planning
                        try:
                            plan = self._ai.plan_pre_forward()
                            self._ai_fp32_logits = bool(plan.get("fp32_logits", False))
                        except Exception:
                            self._ai_fp32_logits = bool(
                                int(os.environ.get("AETHER_FP32_LOGITS", "0"))
                            )

                        try:
                            with torch.autocast(
                                device_type="mps", dtype=self.amp_dtype
                            ):
                                logits, lvi_logs = self._compute_logits(subx)
                                if self._strict_activation_guard:
                                    logits = self._sanitize_tensor(
                                        logits, "forward.logits", step
                                    )
                                if isinstance(lvi_logs, dict):
                                    for _lk in ("mv_reg", "two_view"):
                                        if _lk in lvi_logs:
                                            lvi_logs[_lk] = self._sanitize_loss_value(
                                                lvi_logs[_lk], f"lvi.{_lk}", step
                                            )
                                _pf = {}
                                try:
                                    _pf = self._ai.post_forward_assess(
                                        logits, subx=subx, suby=suby, pad_id=pad_id
                                    )
                                except Exception:
                                    pass
                            # ★ CE は常に FP32・autocast 無効で計算（数値安定のため）
                            with torch.autocast(device_type="mps", enabled=False):
                                loss_ce = self._ce_loss(logits, suby, pad_id)
                                loss_ce = self._sanitize_loss_value(
                                    loss_ce, "loss_ce", step
                                )
                                loss = (
                                    loss_ce
                                    + float(self.cfg.lvi_mv_weight) * lvi_logs["mv_reg"]
                                    + float(self.cfg.lvi_two_view_weight)
                                    * lvi_logs["two_view"]
                                )

                                # ANI-AI observe forward
                                try:
                                    self._ai.observe_forward(loss, logits)
                                except Exception:
                                    pass

                                # --- K-bridge weak Huber regularizer (optional, safe default off)
                                if (
                                    self._k_enabled
                                    and _KBRIDGE_AVAILABLE
                                    and (self._k_reg_w > 0.0)
                                    and (self._teacher is not None)
                                ):
                                    try:
                                        with torch.no_grad():
                                            probs = torch.softmax(
                                                logits.float(), dim=-1
                                            )
                                            emb_w = self.model.emb.weight.float()
                                            E_pred = torch.einsum(
                                                "btv,vd->btd", probs, emb_w
                                            )
                                            E_bar = E_pred.mean(dim=1)
                                            t_pos = self._teacher(subx).to(
                                                E_bar.device, dtype=E_bar.dtype
                                            )
                                        reg = khuber_loss(E_bar, t_pos, delta=1.0)
                                        reg = self._sanitize_loss_value(
                                            reg, "kbridge", step
                                        )
                                        loss = loss + float(self._k_reg_w) * reg
                                    except Exception:
                                        pass

                                if getattr(self.cfg, "intent_weight", 0.0) > 0.0:
                                    every = max(1, int(getattr(self.cfg, "intent_every", 1)))
                                    do_int = (every <= 1) or (
                                        self._global_step % every == 0
                                    )
                                    if do_int and (self._teacher is not None):
                                        try:
                                            emb_w = getattr(self.model, "emb", None)
                                            if emb_w is not None:
                                                t_pos = self._teacher(subx)
                                                neg_ids = _make_negative_ids(subx)
                                                t_neg = self._teacher(neg_ids)
                                                loss_int = _contrastive_intent_loss(
                                                    logits,
                                                    suby,
                                                    emb_w.weight,
                                                    pad_id,
                                                    t_pos,
                                                    t_neg,
                                                    margin=float(
                                                        getattr(
                                                            self.cfg,
                                                            "intent_margin",
                                                            0.10,
                                                        )
                                                    ),
                                                    sample_frac=float(
                                                        getattr(
                                                            self.cfg,
                                                            "intent_sample_frac",
                                                            0.25,
                                                        )
                                                    ),
                                                )
                                                loss_int = self._sanitize_loss_value(
                                                    loss_int, "intent_loss", step
                                                )
                                                loss = (
                                                    loss
                                                    + float(self.cfg.intent_weight)
                                                    * loss_int
                                                )
                                        except Exception:
                                            pass

                                loss = self._sanitize_loss_value(loss, "loss", step)
                                loss = loss / max(1, current_micro)
                                loss = self._sanitize_loss_value(
                                    loss, "loss_scaled", step
                                )
                        except RuntimeError as e:
                            if self._maybe_handle_oom(e, B):
                                restart_batch = True
                                oom_count += 1
                                last_oom_err = e
                                self.opt.zero_grad(set_to_none=True)
                                if self._scaler_enabled:
                                    self.scaler = GradScaler()
                                    print("[OOM] GradScaler reset after OOM")
                                break
                            raise

                        if restart_batch:
                            break

                        backward_ok = True
                        if _pf.get("hazard", False):
                            try:
                                self._ai.sanitize_gradients(self.model)
                            except Exception:
                                pass
                            self.opt.zero_grad(set_to_none=True)
                            if self.scaler is not None:
                                self.scaler.update()
                        else:
                            backward_ok = self._backward_with_guard(loss, step)
                        if not backward_ok:
                            continue
                        # ANI-AI observe grads
                        try:
                            self._ai.observe_grads(self.model)
                        except Exception:
                            pass
                        self._zero_nonfinite_grads()
                        accum += 1

                        tok_tensor = (suby != pad_id).sum().to(total_tok_tensor.dtype)
                        total_tok_tensor = total_tok_tensor + tok_tensor
                        total_loss_tensor = total_loss_tensor + loss_ce.detach().to(
                            total_loss_tensor.dtype
                        ) * tok_tensor

                        current_micro = max(1, int(self._micro_active))
                        if accum >= current_micro:
                            if self.scaler is not None:
                                self.scaler.unscale_(self.opt)
                            self._prepare_ani_grads_for_step()
                            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), self.cfg.grad_clip
                                )
                            # ANI-AI: skip step if requested
                            if hasattr(self, "_ai") and self._ai.should_skip():
                                if self.scaler is not None:
                                    self.scaler.update()
                                self.opt.zero_grad(set_to_none=True)
                            else:
                                lr_mult = sched.step()
                                self._optimizer_step_with_guard(step)
                                self.opt.zero_grad(set_to_none=True)
                            # ANI-AI decide/act
                            try:
                                self._ai.decide_and_act(step)
                            except Exception:
                                pass
                            try:
                                if up is not None:
                                    up.mps_empty_cache_safe()
                            except Exception:
                                pass
                            if (
                                self._empty_cache_every > 0
                                and ((step + 1) % self._empty_cache_every == 0)
                                and self.device.type == "mps"
                            ):
                                try:
                                    torch.mps.empty_cache()
                                except Exception:
                                    pass
                            accum = 0

                        idx = end

                    if restart_batch:
                        if self._oom_retry_limit and oom_count > self._oom_retry_limit:
                            raise last_oom_err
                        continue
                    break

                if accum > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.opt)
                    self._prepare_ani_grads_for_step()
                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.grad_clip
                        )
                    # ANI-AI: skip step if requested
                    if hasattr(self, "_ai") and self._ai.should_skip():
                        if self.scaler is not None:
                            self.scaler.update()
                        self.opt.zero_grad(set_to_none=True)
                    else:
                        lr_mult = sched.step()
                        self._optimizer_step_with_guard(step)
                        self.opt.zero_grad(set_to_none=True)
                    # ANI-AI decide/act
                    try:
                        self._ai.decide_and_act(step)
                    except Exception:
                        pass
                    try:
                        if up is not None:
                            up.mps_empty_cache_safe()
                    except Exception:
                        pass
                    if (
                        self._empty_cache_every > 0
                        and ((step + 1) % self._empty_cache_every == 0)
                        and self.device.type == "mps"
                    ):
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass

                if self.cfg.mps_sync_every > 0 and (
                    step % self.cfg.mps_sync_every == 0
                ):
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass

                total_tok = int(total_tok_tensor.item())
                total_loss = float(total_loss_tensor.item())
                raw_loss_avg = total_loss / max(1, total_tok)
                loss_avg = self._stabilize_loss_value(raw_loss_avg)
                loss_avg = max(0.0, loss_avg)
                smooth_loss = self._loss_meter.update(loss_avg)
                ppl_base = smooth_loss
                if self._ppl_cap > 0:
                    ppl_base = min(ppl_base, self._ppl_cap)
                ppl = math.exp(ppl_base)
                elapsed = time.time() - t0
                t0 = time.time()
                tps = float(total_tok) / max(1e-6, elapsed)
                ema_tps = self._tps_meter.update(total_tok, elapsed)
                tps_snapshot = self._tps_meter.stats(as_dict=False)
                self._maybe_boost_throughput(ema_tps)
                self._last_tok_per_sec = float(tps_snapshot.instant)
                self._last_tok_per_sec_ema = float(ema_tps)
                step += 1
                self._global_step = step

                budget_ratio = (
                    float(total_tok) / float(self._token_budget)
                    if self._token_budget > 0
                    else 0.0
                )
                chunk_hint = (
                    float(self._token_chunk)
                    if self._token_chunk > 0
                    else float(total_tok)
                )

                self._turbo_update(step, tps, ema_tps, tps_snapshot, total_tok, chunk_hint)
                tps_stats = tps_snapshot.as_dict()

                if self._adaptive_micro:
                    if self._micro_active > self._micro_base:
                        self._micro_stable += 1
                        if (
                            self._micro_recover_every > 0
                            and self._micro_stable >= self._micro_recover_every
                        ):
                            new_micro = max(self._micro_base, self._micro_active // 2)
                            if new_micro < self._micro_active:
                                self._micro_active = new_micro
                                print(
                                    f"[ADAPT] micro_batch relaxed to {self._micro_active}"
                                )
                            self._micro_stable = 0
                    else:
                        self._micro_stable = 0

                if int(getattr(self.cfg, "relora_every", 0)) > 0 and (
                    step % int(self.cfg.relora_every) == 0
                ):
                    try:
                        self._relora_cycle()
                    except Exception:
                        pass

                        # --- K metrics: token-level buffers（pmax/correct/pred/true/lenbin + groups）
                if (
                    self._k_enabled
                    and _KBRIDGE_AVAILABLE
                    and not getattr(self, "_turbo_disable_kbridge", False)
                    and (step % max(1, self._kb_collect_every) == 0)
                ):
                    try:
                        with torch.no_grad():
                            p = torch.softmax(logits.float(), dim=-1)
                            pmax = p.amax(dim=-1)  # (B,T)
                            pred = p.argmax(dim=-1)  # (B,T)
                            correct = (pred == suby).float()  # (B,T)
                            mask = suby != pad_id
                            # to CPU np
                            pm = (
                                pmax[mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype("float64", copy=False)
                            )
                            lb = (
                                correct[mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype("int32", copy=False)
                            )
                            pr = (
                                pred[mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype("int32", copy=False)
                            )
                            tr = (
                                suby[mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype("int32", copy=False)
                            )
                            # length-bin per token
                            np_mod = _require_numpy(
                                "knowledge-bridge token analytics"
                            )

                            lens = mask.sum(dim=1).detach().cpu().tolist()
                            edges = self._kb_len_edges
                            lenbins = []
                            for i in range(mask.size(0)):
                                L = int(lens[i])
                                if L <= 0:
                                    continue
                                b = 0
                                for j in range(len(edges) - 1):
                                    if edges[j] <= L < edges[j + 1]:
                                        b = j
                                        break
                                lenbins.append(
                                    np_mod.full(
                                        (int(mask[i].sum().item()),), b, dtype=np_mod.int32
                                    )
                                )
                            lbins = (
                                np_mod.concatenate(lenbins, axis=0)
                                if len(lenbins) > 0
                                else np_mod.zeros((0,), dtype=np_mod.int32)
                            )
                            # groups (map token id -> group id)
                            cm = self._kb_classmap
                            prg = cm[np_mod.clip(pr, 0, cm.size - 1)]
                            trg = cm[np_mod.clip(tr, 0, cm.size - 1)]
                            # append
                            self._kb_buf_pmax.append(pm)
                            self._kb_buf_labels.append(lb)
                            self._kb_buf_predcls.append(pr)
                            self._kb_buf_truecls.append(tr)
                            self._kb_buf_lenbin.append(lbins)
                            # store groups
                            self._kb_buf_predgrp = getattr(self, "_kb_buf_predgrp", [])
                            self._kb_buf_truegrp = getattr(self, "_kb_buf_truegrp", [])
                            self._kb_buf_predgrp.append(prg)
                            self._kb_buf_truegrp.append(trg)
                            # ring-buffer compress
                            total = sum(a.size for a in self._kb_buf_pmax)
                            if total > self._kb_buf_cap:
                                keep = self._kb_buf_cap // 2
                                cur = 0
                                start = 0
                                for i, a in enumerate(self._kb_buf_pmax):
                                    cur += a.size
                                    if cur >= (total - keep):
                                        start = i
                                        break
                                if start > 0:
                                    self._kb_buf_pmax = self._kb_buf_pmax[start:]
                                    self._kb_buf_labels = self._kb_buf_labels[start:]
                                    self._kb_buf_predcls = self._kb_buf_predcls[start:]
                                    self._kb_buf_truecls = self._kb_buf_truecls[start:]
                                    self._kb_buf_lenbin = self._kb_buf_lenbin[start:]
                                    self._kb_buf_predgrp = self._kb_buf_predgrp[start:]
                                    self._kb_buf_truegrp = self._kb_buf_truegrp[start:]
                    except Exception:
                        pass

                if step % self.cfg.log_every == 0:
                    heavy_allowed = not getattr(
                        self, "_turbo_metrics_suppressed", False
                    )
                    # --- Temperature sweep ECE（current batch; snapshot only）
                    if (
                        heavy_allowed
                        and self._k_enabled
                        and _KBRIDGE_AVAILABLE
                        and self._kb_tsweep
                    ):
                        try:
                            with torch.no_grad():
                                for Tval in self._kb_tsweep:
                                    pT = torch.softmax(
                                        (logits.float() / float(Tval)), dim=-1
                                    )
                                    pmaxT = pT.amax(dim=-1)
                                    predT = pT.argmax(dim=-1)
                                    correctT = (predT == suby).float()
                                    mask = suby != pad_id
                                    pmT = (
                                        pmaxT[mask]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .astype("float64", copy=False)
                                    )
                                    lbT = (
                                        correctT[mask]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .astype("int32", copy=False)
                                    )
                                    eT, _, _, _ = ece_and_hist_k(
                                        pmT, lbT, n_bins=max(2, self._kb_ece_bins)
                                  )
                                    try:
                                        V = int(logits.shape[-1])
                                        mask = (suby != pad_id)
                                        tmin = int(suby[mask].min().item()) if mask.any() else -1
                                        tmax = int(suby[mask].max().item()) if mask.any() else -1
                                        lmax = float(logits.detach().float().abs().amax().item())
                                        with torch.no_grad():
                                            p = torch.softmax(logits.detach().float(), dim=-1)
                                            pm = p[mask]
                                            ty = suby[mask]
                                            ce_manual = float(
                                                (
                                                    -torch.log(
                                                        pm[torch.arange(pm.size(0)), ty.view(-1)] + 1e-12
                                                    )
                                                )
                                                .mean()
                                                .item()
                                            )
                                            top1 = float(pm.max(dim=-1).values.mean().item())
                                        print(
                                            f"[DBG] V={V} target=[{tmin},{tmax}] | logits|max|={lmax:.2e} "
                                            f"| ce_manual={ce_manual:.3f} | top1={top1:.3f}"
                                        )
                                        assert tmax < V, f"target id {tmax} >= vocab {V}"
                                    except Exception as e:
                                        print("[DBG] sanity failed:", e)

                                    self.fabric.log(
                                        {f"eval/ece_T{Tval:g}": float(eT)}, step=step
                                    )
                        except Exception:
                            pass
                    # autosave LoRA safetensor
                    if int(getattr(self.cfg, "safetensor_every", 0)) > 0 and (
                        step % int(self.cfg.safetensor_every) == 0
                    ):
                        try:
                            save_lora_safetensors_if_any(
                                self.model,
                                getattr(self.cfg, "out_dir", "runs/v285"),
                                step,
                            )
                        except Exception as _e:
                            print("[SAFE] autosave error:", _e)

                    self.fabric.log(
                        {
                            "train/loss": float(loss_avg),
                            "train/loss_raw": float(raw_loss_avg),
                            "train/loss_smooth": float(smooth_loss),
                            "train/ce": float(raw_loss_avg),
                            "train/ppl": float(ppl),
                            "train/tok_s": float(tps),
                            "train/tok_s_ema": float(ema_tps),
                            "train/tok_s_window": float(tps_stats["window_avg"]),
                            "train/tok_s_peak": float(tps_stats["window_max"]),
                            "train/tok_s_lifetime": float(tps_stats["lifetime_avg"]),
                            "train/tok_window_tokens": float(
                                tps_stats["window_tokens"]
                            ),
                            "train/tok_window_seconds": float(
                                tps_stats["window_seconds"]
                            ),
                            "train/tok_window_updates": float(
                                tps_stats["window_updates"]
                            ),
                            "train/tok_lifetime_tokens": float(
                                tps_stats["lifetime_tokens"]
                            ),
                            "train/tok_lifetime_seconds": float(
                                tps_stats["lifetime_seconds"]
                            ),
                            "train/lr_mult": float(lr_mult),
                            "train/token_budget_ratio": float(budget_ratio),
                            "train/token_chunk_hint": float(chunk_hint),
                            "train/token_budget_target": float(
                                self._token_budget_target
                                if self._token_budget_target > 0
                                else 0.0
                            ),
                        },
                        step=step,
                    )
                    if (
                        val_dl is not None
                        and int(getattr(self.cfg, "eval_every", 0)) > 0
                        and (step % int(getattr(self.cfg, "eval_every", 1)) == 0)
                    ):
                        _run_validation_snapshot(step)

                    # --- K-bridge: ECE + hist（overall）/ length-bucket / classwise（拡張）
                    if (
                        heavy_allowed
                        and self._k_enabled
                        and _KBRIDGE_AVAILABLE
                        and (step % max(1, self._kb_metrics_every) == 0)
                        and self._kb_buf_pmax
                    ):
                        try:
                            np_mod = _require_numpy("knowledge-bridge aggregate metrics")

                            pm = np_mod.concatenate(self._kb_buf_pmax, axis=0)
                            lb = np_mod.concatenate(self._kb_buf_labels, axis=0)
                            pr = (
                                np_mod.concatenate(self._kb_buf_predcls, axis=0)
                                if getattr(self, "_kb_buf_predcls", None)
                                else np_mod.zeros((0,), dtype=np_mod.int32)
                            )
                            tr = (
                                np_mod.concatenate(self._kb_buf_truecls, axis=0)
                                if getattr(self, "_kb_buf_truecls", None)
                                else np_mod.zeros((0,), dtype=np_mod.int32)
                            )
                            ln = (
                                np_mod.concatenate(self._kb_buf_lenbin, axis=0)
                                if self._kb_buf_lenbin
                                else np_mod.zeros((0,), dtype=np_mod.int32)
                            )
                            prg = (
                                np_mod.concatenate(
                                    getattr(self, "_kb_buf_predgrp", []), axis=0
                                )
                                if getattr(self, "_kb_buf_predgrp", None)
                                else np_mod.zeros((0,), dtype=np_mod.int32)
                            )
                            trg = (
                                np_mod.concatenate(
                                    getattr(self, "_kb_buf_truegrp", []), axis=0
                                )
                                if getattr(self, "_kb_buf_truegrp", None)
                                else np_mod.zeros((0,), dtype=np_mod.int32)
                            )
                            # overall ECE + hist（等幅）
                            ece, counts, mconf, macc = ece_and_hist_k(
                                pm, lb, n_bins=max(2, self._kb_ece_bins)
                            )
                            logs = {"eval/ece_top": float(ece)}
                            for i, ct in enumerate(counts.tolist()[:10]):
                                logs[f"eval/conf_hist/bin{i}"] = float(ct)
                                logs[f"eval/conf_bin_mean/conf{i}"] = float(mconf[i])
                                logs[f"eval/conf_bin_mean/acc{i}"] = float(macc[i])
                            # quantile bins（分位ヒスト）
                            qspec = self._kb_qbins
                            if qspec:
                                if qspec.startswith("q"):
                                    try:
                                        Q = max(2, int(qspec[1:]))
                                    except Exception:
                                        Q = 10
                                    qs = np_mod.linspace(0.0, 1.0, Q + 1)
                                    edges = np_mod.quantile(pm, qs)
                                else:
                                    edges = np_mod.asarray(
                                        [
                                            float(x)
                                            for x in qspec.split(",")
                                            if x.strip()
                                        ],
                                        dtype=np_mod.float64,
                                    )
                                edges[0] = 0.0
                                edges[-1] = 1.0
                                e_q, cnt_q, mc_q, ma_q, edges = ece_and_hist_k_bins(
                                    pm, lb, edges
                                )
                                logs[f"eval/ece_quantile@{len(edges) - 1}"] = float(e_q)
                                for i, ct in enumerate(cnt_q.tolist()[:5]):
                                    logs[f"eval/qhist/bin{i}"] = float(ct)
                            # classwise（group or raw class）
                            base_key = self._kb_ece_by
                            if base_key in ("predgrp", "truegrp"):
                                g = prg if base_key == "predgrp" else trg
                                G = int(self._kb_group_count)
                                e_g, c_g = ece_multi_groups_k(
                                    pm,
                                    lb,
                                    g,
                                    n_groups=G,
                                    n_bins=max(2, self._kb_ece_bins),
                                )
                                if self._kb_ece_c_list:
                                    idxs = [
                                        i for i in self._kb_ece_c_list if 0 <= i < G
                                    ]
                                else:
                                    order = np_mod.argsort(-c_g)
                                    idxs = [
                                        int(i)
                                        for i in order[: max(1, self._kb_ece_c_top)]
                                        if c_g[int(i)] >= self._kb_ece_c_min
                                    ]
                                for gi in idxs[: max(1, self._kb_ece_c_top)]:
                                    name = (
                                        self._kb_group_names[gi]
                                        if gi < len(self._kb_group_names)
                                        else f"g{gi}"
                                    )
                                    logs[f"eval/ece_grp_{base_key}[{name}]"] = float(
                                        e_g[gi]
                                    )
                                    logs[f"eval/count_grp_{base_key}[{name}]"] = float(
                                        c_g[gi]
                                    )
                            # 2D: length × class-group
                            if (
                                self._kb_enable_2d
                                and len(ln) > 0
                                and base_key in ("predgrp", "truegrp")
                            ):
                                g = prg if base_key == "predgrp" else trg
                                edges = self._kb_len_edges
                                G = int(self._kb_group_count)
                                _, c_all = ece_multi_groups_k(
                                    pm,
                                    lb,
                                    g,
                                    n_groups=G,
                                    n_bins=max(2, self._kb_ece_bins),
                                )
                                if self._kb_ece_c_list:
                                    sel_groups = [
                                        i for i in self._kb_ece_c_list if 0 <= i < G
                                    ]
                                else:
                                    order = np_mod.argsort(-c_all)
                                    sel_groups = [
                                        int(i)
                                        for i in order[: max(1, self._kb_ece_c_top)]
                                        if c_all[int(i)] >= self._kb_ece_c_min
                                    ]
                                cells = 0
                                for b in range(len(edges) - 1):
                                    idx = ln == b
                                    if not idx.any():
                                        continue
                                    e_b, c_b = ece_multi_groups_k(
                                        pm[idx],
                                        lb[idx],
                                        g[idx],
                                        n_groups=G,
                                        n_bins=max(2, self._kb_ece_bins),
                                    )
                                    lo, hi = edges[b], edges[b + 1]
                                    cnt_len = float(c_b.sum())
                                    logs[f"eval/ece2d_len[{lo},{hi})/count"] = cnt_len
                                    for gi in sel_groups:
                                        name = (
                                            self._kb_group_names[gi]
                                            if gi < len(self._kb_group_names)
                                            else f"g{gi}"
                                        )
                                        logs[
                                            f"eval/ece2d_len[{lo},{hi})/grp[{name}]"
                                        ] = float(e_b[gi])
                                        cells += 1
                                        if cells >= self._kb_2d_max_cells:
                                            break
                                    if cells >= self._kb_2d_max_cells:
                                        break
                            # AUC/PR-AUC（overall）
                            try:
                                auc = roc_auc_binary(lb, pm)
                                pauc = pr_auc_binary(lb, pm)
                                logs["eval/auc"] = float(auc)
                                logs["eval/prauc"] = float(pauc)
                            except Exception:
                                pass
                            self.fabric.log(logs, step=step)
                        except Exception:
                            pass

                    # --- K-bridge: sequence-level NDCG@k (external hook provided)
                    if (
                        heavy_allowed
                        and self._k_enabled
                        and _KBRIDGE_AVAILABLE
                        and self._kb_seq_scores
                    ):
                        try:
                            np_mod = _require_numpy("sequence-level NDCG evaluation")

                            kseq = int(os.environ.get("KBRIDGE_NDCG_K", "10"))
                            Lmax = max(s.size for s in self._kb_seq_scores)
                            S = np_mod.stack(
                                [
                                    np_mod.pad(s, (0, Lmax - s.size), constant_values=0.0)
                                    for s in self._kb_seq_scores
                                ],
                                axis=0,
                            )
                            R = np_mod.stack(
                                [
                                    np_mod.pad(g, (0, Lmax - g.size), constant_values=0.0)
                                    for g in self._kb_seq_gains
                                ],
                                axis=0,
                            )
                            nd = ndcg_at_k_seq_k(S, R, k=kseq)
                            self.fabric.log(
                                {"eval/ndcg_seq@{}".format(kseq): float(nd.mean())},
                                step=step,
                            )
                            self._kb_seq_scores.clear()
                            self._kb_seq_gains.clear()
                        except Exception:
                            pass
                pending = next_pending
                if self.cfg.max_steps and step >= self.cfg.max_steps:
                    break
            if self.cfg.max_steps and step >= self.cfg.max_steps:
                break

    def _maybe_handle_oom(self, err: RuntimeError, batch_tokens: int) -> bool:
        if not getattr(self, "_adaptive_micro", False):
            return False
        msg = str(err).lower()
        keywords = (
            "out of memory",
            "mps backend out of memory",
            "failed to allocate",
            "resource exhausted",
        )
        if not any(k in msg for k in keywords):
            return False
        if self._micro_active >= self._micro_cap:
            print(
                f"[OOM] {err} -- micro_batch cap reached ({self._micro_active}); aborting"
            )
            return False
        prev = self._micro_active
        self._micro_active = min(
            self._micro_cap, max(prev + 1, prev * 2)
        )
        self._micro_stable = 0
        approx_chunk = max(1, math.ceil(batch_tokens / self._micro_active))
        print(
            f"[OOM] {err} -- increasing micro_batch to {self._micro_active} (chunk≈{approx_chunk})"
        )
        if self.device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return True

    def _zero_nonfinite_grads(self):
        had_issue = False
        for p in self.model.parameters():
            if p.grad is None:
                continue
            finite = torch.isfinite(p.grad)
            if finite.all():
                continue
            p.grad.zero_()
            had_issue = True
        if had_issue:
            self._emit_nonfinite_event(
                "grad", self._global_step, detail="gradients zeroed"
            )

    def _ani_loss_scale(self) -> float:
        manager = getattr(self, "_ani_manager", None)
        if manager is None or not getattr(manager, "enabled", True):
            return 1.0
        try:
            scale = float(getattr(manager, "loss_scale", 1.0))
        except Exception:
            return 1.0
        if not math.isfinite(scale) or scale <= 0.0:
            return 1.0
        return scale

    def _ani_scaled_loss_for_backward(self, loss: torch.Tensor) -> torch.Tensor:
        scale = self._ani_loss_scale()
        if abs(scale - 1.0) < 1e-6:
            return loss
        return loss * scale

    def _prepare_ani_grads_for_step(self):
        manager = getattr(self, "_ani_manager", None)
        if manager is None or not getattr(manager, "enabled", True):
            return
        try:
            manager.unscale_grads()
        except Exception as err:
            if getattr(self, "_nonfinite_event_verbose", False):
                print("[ANI] grad unscale failed:", err)
        try:
            manager.relax_if_stable()
        except Exception as err:
            if getattr(self, "_nonfinite_event_verbose", False):
                print("[ANI] relax failed:", err)

    @property
    def tok_per_sec(self) -> float:
        return float(self._last_tok_per_sec)

    @property
    def tok_per_sec_ema(self) -> float:
        return float(self._last_tok_per_sec_ema)

    def _emit_nonfinite_event(self, where: str, step: int, detail: str = ""):
        if self._nonfinite_event_verbose:
            msg = f"[SAFE] non-finite sanitized at step={int(step)} ({where})"
            if detail:
                msg += f" :: {detail}"
            print(msg)
        evt = {"event": "nonfinite", "where": where, "step": int(step), "detail": detail}
        bus = globals().get("_AETHER_EVENT_BUS")
        if bus is not None:
            try:
                bus.emit(evt)
            except Exception:
                pass
        else:
            manager = getattr(self, "_ani_manager", None)
            if manager is not None:
                try:
                    manager.on_nonfinite(evt)
                except Exception:
                    pass

    def _sanitize_loss_value(self, value, name: str, step: int):
        if not self._strict_loss_guard or value is None:
            return value
        if torch.is_tensor(value):
            try:
                finite = torch.isfinite(value)
            except Exception:
                return value
            if finite.all():
                return value
            sanitized = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            detail = ""
            try:
                detail = f"value={float(value.detach().cpu().item()):.4g}"
            except Exception:
                try:
                    nf = int((~finite).sum().item())
                    detail = f"nonfinite={nf}/{int(finite.numel())}"
                except Exception:
                    detail = "nonfinite"
            self._emit_nonfinite_event(name, step, detail=detail)
            return sanitized
        if isinstance(value, (float, int)):
            if math.isfinite(float(value)):
                return value
            self._emit_nonfinite_event(name, step, detail=f"value={value}")
            return 0.0
        return value

    def _sanitize_tensor(self, tensor: torch.Tensor, name: str, step: int):
        if tensor is None or not torch.is_tensor(tensor):
            return tensor
        try:
            finite = torch.isfinite(tensor)
        except Exception:
            return tensor
        if finite.all():
            return tensor
        sanitized = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        detail = ""
        try:
            nf = int((~finite).sum().item())
            detail = f"nonfinite={nf}/{int(finite.numel())}"
        except Exception:
            detail = "nonfinite"
        self._emit_nonfinite_event(name, step, detail=detail)
        return sanitized

    def _is_numeric_error(self, err: BaseException) -> bool:
        if err is None:
            return False
        msg = str(err).lower()
        for key in ("nan", "inf", "overflow", "out of range", "invalid value"):
            if key in msg:
                return True
        return False

    def _backward_with_guard(self, loss: torch.Tensor, step: int) -> bool:
        try:
            loss_for_backward = self._ani_scaled_loss_for_backward(loss)
            if self.scaler is not None:
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            return True
        except RuntimeError as err:
            if self._suppress_numeric_traceback and self._is_numeric_error(err):
                self._emit_nonfinite_event("backward", step, detail=str(err))
                try:
                    self.opt.zero_grad(set_to_none=True)
                except Exception:
                    pass
                if self.scaler is not None:
                    try:
                        self.scaler.update()
                    except Exception:
                        pass
                return False
            raise

    def _optimizer_step_with_guard(self, step: int) -> bool:
        try:
            if self.scaler is not None:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            return True
        except RuntimeError as err:
            if self._suppress_numeric_traceback and self._is_numeric_error(err):
                self._emit_nonfinite_event("optimizer_step", step, detail=str(err))
                if self.scaler is not None:
                    try:
                        self.scaler.update()
                    except Exception:
                        pass
                return False
            raise


# ====== Lightning Fabric (optional placeholder) =============================
_HAVE_FABRIC = False
try:
    import lightning as L

    _HAVE_FABRIC = True
except Exception:
    L = None


class AetherTrainerFabric(AetherTrainerBase):
    def __init__(self, *a, ckpt_every: int = 1, precision: str = "16-mixed", **k):
        super().__init__(*a, **k)
        self.precision = precision

    def enable_lvi_for_trainer(self, cfg):
        pass


# ====== CLI ==================================================================


def save_lora_safetensors_if_any(model, out_dir: str, step: int):
    try:
        from safetensors.torch import save_file as _sf_save
    except Exception as _e:
        print("[SAFE] safetensors not available; skip save:", _e)
        return False
    try:
        import os

        os.makedirs(out_dir, exist_ok=True)
        sd = model.state_dict()
        sel = {
            k: v.detach().cpu()
            for k, v in sd.items()
            if ("lora_" in k) or ("adapter" in k)
        }
        if len(sel) == 0:
            print("[SAFE] no LoRA/adapter weights detected; skip")
            return False
        fn = os.path.join(out_dir, f"aether_lora_step{int(step)}.safetensors")
        _sf_save(sel, fn, metadata={"aether_ver": "2.8.5"})
        print(f"[SAFE] wrote LoRA safetensors: {fn}")
        return True
    except Exception as _e:
        print("[SAFE] save failed:", _e)
        return False


def _coerce_config_value(value: Any, template: Any) -> Any:
    if template is None:
        return value
    try:
        if isinstance(template, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if isinstance(template, int) and not isinstance(template, bool):
            if isinstance(value, bool):
                return int(value)
            return int(value)
        if isinstance(template, float):
            return float(value)
        if isinstance(template, str):
            return str(value)
    except Exception:
        return value
    return value


def _resolve_config_path(path: str) -> str:
    if not path:
        return ""
    expanded = os.path.expanduser(path)
    if os.path.isdir(expanded):
        for candidate in (
            "aether.config.json",
            "aether_config.json",
            "config.json",
        ):
            sub = os.path.join(expanded, candidate)
            if os.path.isfile(sub):
                return os.path.abspath(sub)
        return ""
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)
    return ""


def _auto_detect_config_path() -> str:
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, "aether.config.json"),
        os.path.join(cwd, "aether_config.json"),
        os.path.join(cwd, "config", "aether.json"),
        os.path.join(cwd, "config", "config.json"),
        os.path.join(cwd, "configs", "aether.json"),
        os.path.join(cwd, "config.json"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    config_dir = os.path.join(cwd, "configs")
    try:
        jsons = [
            os.path.join(config_dir, name)
            for name in os.listdir(config_dir)
            if name.lower().endswith(".json")
        ]
        if len(jsons) == 1 and os.path.isfile(jsons[0]):
            return os.path.abspath(jsons[0])
    except OSError:
        pass
    return ""


def _load_config_overrides(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[CONFIG] failed to load {path!r}: {exc}")
        return {}
    if not isinstance(data, dict):
        print(f"[CONFIG] {path!r} does not contain a JSON object; ignoring")
        return {}
    return data


def _auto_find_any(patterns: List[str]) -> str:
    for pat in patterns:
        recursive = "**" in pat
        try:
            it = glob.iglob(pat, recursive=recursive)
            next(it)
            return pat
        except StopIteration:
            continue
        except OSError:
            continue
    return ""


def _auto_discover_corpus(kind: str) -> str:
    if kind == "train":
        patterns = [
            "data/train/**/*.txt",
            "data/train/**/*.jsonl",
            "data/train/**/*.json",
            "datasets/train/**/*.txt",
            "datasets/train/**/*.jsonl",
            "datasets/**/*.txt",
            "datasets/**/*.jsonl",
            "data/**/*.txt",
            "data/**/*.jsonl",
        ]
    else:
        patterns = [
            "data/val/**/*.txt",
            "data/val/**/*.jsonl",
            "data/valid/**/*.txt",
            "data/valid/**/*.jsonl",
            "data/validation/**/*.txt",
            "data/validation/**/*.jsonl",
            "datasets/val/**/*.txt",
            "datasets/val/**/*.jsonl",
            "datasets/valid/**/*.txt",
            "datasets/valid/**/*.jsonl",
        ]
    return _auto_find_any(patterns)


def _auto_bootstrap_runtime(
    args: argparse.Namespace,
    defaults: argparse.Namespace,
    override_keys: set,
    device: torch.device,
) -> None:
    notes: List[str] = []

    def _is_default(name: str) -> bool:
        if name in override_keys:
            return False
        if not hasattr(args, name) or not hasattr(defaults, name):
            return False
        return getattr(args, name) == getattr(defaults, name)

    def _apply(name: str, value: Any) -> None:
        setattr(args, name, value)
        notes.append(f"{name}={value!r}")

    # Auto-discover training corpus
    if _is_default("train_glob") or not getattr(args, "train_glob", ""):
        train_pat = _auto_discover_corpus("train")
        if train_pat:
            _apply("train_glob", train_pat)
            if not getattr(args, "train_stream", False) and _is_default("train_stream"):
                _apply("train_stream", True)

    # Auto-discover validation corpus if unset
    if (_is_default("val_glob") or not getattr(args, "val_glob", "")) and getattr(
        args, "train_glob", ""
    ):
        val_pat = _auto_discover_corpus("val")
        if val_pat:
            _apply("val_glob", val_pat)

    # Ensure micro-batch does not exceed global batch
    if (
        hasattr(args, "batch_size")
        and hasattr(args, "micro_batch")
        and int(getattr(args, "micro_batch", 0)) > max(1, int(getattr(args, "batch_size", 1)))
        and _is_default("micro_batch")
    ):
        _apply("micro_batch", max(1, int(args.batch_size)))

    # Align pack length with target sequence length when possible
    if hasattr(args, "pack_len") and hasattr(args, "max_len") and _is_default("pack_len"):
        max_len = max(32, int(getattr(args, "max_len", 1024)))
        desired_pack = min(max_len, 2048)
        if desired_pack != int(getattr(args, "pack_len", desired_pack)):
            _apply("pack_len", desired_pack)
    if hasattr(args, "pack_len") and hasattr(args, "buffer_size") and _is_default(
        "buffer_size"
    ):
        pack_len = int(getattr(args, "pack_len", 1024))
        desired_buffer = max(int(getattr(args, "buffer_size", 8192)), pack_len * 4)
        if desired_buffer != int(getattr(args, "buffer_size", desired_buffer)):
            _apply("buffer_size", desired_buffer)

    # Tune auto MPS token budget if still default
    if device.type == "mps" and _is_default("auto_mps_token_budget"):
        target_seq = int(getattr(args, "auto_mps_target_seq", getattr(args, "max_len", 1024)))
        max_len = int(getattr(args, "max_len", target_seq))
        batch = max(1, int(getattr(args, "batch_size", 1)))
        desired_budget = max(target_seq, max_len * batch)
        if desired_budget != int(getattr(args, "auto_mps_token_budget", desired_budget)):
            _apply("auto_mps_token_budget", desired_budget)

    if notes:
        print("[AUTO] runtime tuned:", ", ".join(notes))


def __aether_main__():
    def _require_mps():
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise SystemExit("MPS is required. Run on macOS with Apple Silicon.")

    _require_mps()

    def _count_glob(pattern: str) -> int:
        if not pattern:
            return 0
        return sum(1 for _ in glob.iglob(pattern, recursive=True))

    ap = argparse.ArgumentParser()
    # data
    ap.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to config JSON (auto-detected if omitted)",
    )
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--train_stream", action="store_true")
    ap.add_argument("--train_glob", type=str, default="")
    ap.add_argument("--val_glob", type=str, default="")
    ap.add_argument("--pack_len", type=int, default=1024)
    ap.add_argument("--buffer_size", type=int, default=8192)
    ap.add_argument("--loader_workers", type=int, default=0)
    ap.add_argument("--loader_prefetch", type=int, default=2)
    ap.add_argument("--loader_persistent_workers", action="store_true")
    ap.add_argument("--loader_pin_memory", action="store_true")
    ap.add_argument("--no_prefetch_to_device", action="store_true")
    # model
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--sp_model", type=str, default="", help="SentencePiece model path")
    ap.add_argument(
        "--tiktoken_encoding",
        type=str,
        default="",
        help="Name of the TikToken encoding to blend (e.g. 'cl100k_base')",
    )
    ap.add_argument(
        "--disable_byte_fallback",
        action="store_true",
        help="Disable byte-level fallback tokens in the hybrid tokenizer",
    )
    ap.add_argument("--d_model", type=int, default=4096)
    ap.add_argument("--n_layers", type=int, default=32)
    ap.add_argument("--n_heads", type=int, default=32)
    ap.add_argument(
        "--kv_heads", type=int, default=0, help="0=disable GQA, >0=KV heads"
    )
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--use_rope", action="store_true")
    ap.add_argument("--rope_theta", type=float, default=10_000.0)
    ap.add_argument("--rope_scaling", type=float, default=1.0)
    ap.add_argument("--ff_mult", type=float, default=2.6666667)
    ap.add_argument("--use_abs_pos", action="store_true")
    # train
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--out_dir", type=str, default="runs/v28")
    ap.add_argument("--tiled_q", type=int, default=192)
    ap.add_argument("--tiled_k", type=int, default=320)
    # attention window
    ap.add_argument("--window_size", type=int, default=0)
    ap.add_argument("--global_tokens", type=int, default=0)
    ap.add_argument("--global_stride", type=int, default=0)
    # turbo governor
    ap.add_argument("--turbo_target_tok", type=float, default=2000.0)
    ap.add_argument("--turbo_window", type=int, default=6)
    ap.add_argument("--turbo_cooldown", type=int, default=48)
    ap.add_argument("--turbo_seq_floor", type=int, default=512)
    ap.add_argument("--turbo_seq_step", type=int, default=256)
    ap.add_argument("--turbo_disable_metrics_ratio", type=float, default=0.75)
    ap.add_argument("--turbo_micro_floor", type=int, default=1)
    ap.add_argument("--mps_sync_every", type=int, default=0)
    ap.add_argument("--allow_mps_cpu_fallback", action="store_true")
    ap.add_argument(
        "--ckpt_every",
        type=int,
        default=int(os.environ.get("AETHER_CKPT_EVERY", "1")),
        help="Gradient-checkpoint every Nth block; 0 disables checkpointing for higher tok/s",
    )
    # optimizer
    ap.add_argument("--opt_cpu8bit", action="store_true")
    ap.add_argument("--opt_galore", action="store_true")
    ap.add_argument("--galore_rank", type=int, default=64)
    # LVI
    ap.add_argument("--lvi", action="store_true")
    ap.add_argument("--lvi_k", type=int, default=64)
    ap.add_argument("--lvi_alpha_mode", type=str, default="sparsemax")
    ap.add_argument("--lvi_mv_weight", type=float, default=0.10)
    ap.add_argument("--lvi_two_view_weight", type=float, default=0.05)
    ap.add_argument("--lvi_every", type=int, default=4)
    # Intention
    ap.add_argument("--intent_weight", type=float, default=0.0)
    ap.add_argument("--intent_margin", type=float, default=0.10)
    ap.add_argument("--intent_sample_frac", type=float, default=0.25)
    ap.add_argument("--intent_every", type=int, default=4)
    # ReLoRA
    ap.add_argument("--relora_every", type=int, default=0)
    # LoRA/PEFT
    ap.add_argument("--peft_lora", action="store_true", help="Use PEFT-LoRA")
    ap.add_argument("--peft_targets", type=str, default="qkv,proj")
    ap.add_argument(
        "--int8_lora", action="store_true", help="Use custom INT8 base + LoRA"
    )
    ap.add_argument(
        "--hybrid_lora", action="store_true", help="PEFT + INT8-LoRA hybrid"
    )
    ap.add_argument(
        "--lora_r",
        type=int,
        default=_AETHER_DEFAULT_LORA_R,
        help=f"LoRA rank (default: {_AETHER_DEFAULT_LORA_R})",
    )
    ap.add_argument("--lora_alpha", type=int, default=320)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--auto_mps_7b", action="store_true")
    ap.add_argument("--no_auto_mps_7b", action="store_true")
    ap.add_argument("--auto_mps_target_seq", type=int, default=4096)
    ap.add_argument("--auto_mps_token_budget", type=int, default=8192)
    ap.add_argument("--auto_mps_min_micro_batch", type=int, default=1)
    ap.add_argument("--mps_7b_lora_rank", type=int, default=-1)
    ap.add_argument("--mps_7b_lora_alpha", type=int, default=0)
    ap.add_argument("--mps_7b_lora_dropout", type=float, default=0.05)
    ap.add_argument("--mps_7b_skip_if_out_equals", type=int, default=-1)
    ap.add_argument(
        "--freeze_embeddings",
        action="store_true",
        help="Freeze token embeddings/output head; adapter modes do this automatically",
    )
    ap.add_argument(
        "--train_embeddings",
        action="store_true",
        help="Force embeddings/output head trainable even in adapter modes",
    )
    ap.add_argument("--save_adapter", type=str, default="")
    # tracing
    ap.add_argument("--ts_trace", action="store_true")
    ap.add_argument("--ts_len", type=int, default=1024)
    ap.add_argument("--ts_out", type=str, default="runs/ts/aether_len{L}.pt")
    ap.add_argument("--ts_multi", type=str, default="")
    # memmap load
    ap.add_argument("--load_memmap", type=str, default="")
    ap.add_argument(
        "--safetensor_every",
        type=int,
        default=0,
        help="Save LoRA weights to safetensors every N steps (0=off)",
    )
    defaults = ap.parse_args([])
    args = ap.parse_args()

    config_candidates = []
    if getattr(args, "config", ""):
        config_candidates.append(args.config)
    env_cfg = os.environ.get("AETHER_CONFIG", "").strip()
    if env_cfg:
        config_candidates.append(env_cfg)
    auto_cfg = _auto_detect_config_path()
    if auto_cfg:
        config_candidates.append(auto_cfg)

    config_path = ""
    overrides: Dict[str, Any] = {}
    for cand in config_candidates:
        resolved = _resolve_config_path(cand)
        if not resolved:
            continue
        overrides = _load_config_overrides(resolved)
        config_path = resolved
        if overrides:
            break
    if config_path:
        args.config = config_path
    override_keys = set()
    if overrides:
        print(f"[CONFIG] loaded overrides from {config_path}")
        for key, value in overrides.items():
            if key == "config" or not hasattr(args, key):
                continue
            if not hasattr(defaults, key):
                continue
            current = getattr(args, key)
            default_value = getattr(defaults, key)
            if current != default_value:
                continue
            coerced = _coerce_config_value(value, default_value)
            try:
                setattr(args, key, coerced)
                override_keys.add(key)
                print(f"[CONFIG] {key} <- {coerced!r}")
            except Exception as exc:
                print(f"[CONFIG] failed to apply {key!r}: {exc}")
    elif config_path:
        print(f"[CONFIG] detected {config_path} (no overrides applied)")

    device = detect_device()

    _auto_bootstrap_runtime(args, defaults, override_keys, device)

    if getattr(args, "no_auto_mps_7b", False):
        args.auto_mps_7b = False
    elif (
        getattr(args, "auto_mps_7b", False) == getattr(defaults, "auto_mps_7b", False)
        and "auto_mps_7b" not in override_keys
        and device.type == "mps"
    ):
        args.auto_mps_7b = True
        print("[CONFIG] auto_mps_7b enabled by default (MPS detected)")

    set_seed(1337)

    tok = ByteTokenizer(
        vocab_size=args.vocab_size,
        sp_model_path=args.sp_model or None,
        tiktoken_encoding=args.tiktoken_encoding or None,
        byte_fallback=not getattr(args, "disable_byte_fallback", False),
    )
    if tok.vocab_size != args.vocab_size:
        print(
            f"[TOK] adjusted vocab size from {args.vocab_size} to {tok.vocab_size} to accommodate hybrid ranges"
        )
        args.vocab_size = tok.vocab_size
    kvh = args.kv_heads if args.kv_heads and args.kv_heads > 0 else None
    model = AetherPumpSimple(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=0.1,
        max_len=args.max_len,
        pad_id=tok.PAD,
        ff_mult=float(args.ff_mult),
        use_rope=bool(args.use_rope),
        rope_theta=float(args.rope_theta),
        rope_scaling=float(args.rope_scaling),
        use_abs_pos=bool(args.use_abs_pos),
        kv_heads=kvh,
    ).to(device)
    # ultramem autopatch from env (if enabled)
    if (up is not None) and (os.environ.get("ULTRAMEM_AUTOPATCH", "0") == "1"):
        up.install_autopatch_from_env(lambda: model)
        print("[ULTRAMEM] autopatch requested via env; model patched if configured")

    if args.load_memmap:
        sd = torch.load(args.load_memmap, map_location="cpu")
        model.load_state_dict(sd if isinstance(sd, dict) else sd["model"], strict=False)
        model.to(device)

    peft_targets = [s.strip() for s in args.peft_targets.split(",") if s.strip()]
    ultra_active = (up is not None) and (
        os.environ.get("ULTRAMEM_AUTOPATCH", "0") == "1"
    )
    if ultra_active:
        print("[ULTRAMEM] autopatch active; skipping built-in INT8/PEFT injections")
    else:
        if args.hybrid_lora:
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft not installed for --hybrid_lora")
            model = apply_hybrid_lora(
                model,
                peft_targets,
                int8_include=["w1", "w2", "w3"],
                r=int(args.lora_r),
                alpha=int(args.lora_alpha),
                dropout=float(args.lora_dropout),
            )
        elif args.peft_lora and not args.int8_lora:
            model = apply_peft_lora(
                model,
                r=int(args.lora_r),
                alpha=int(args.lora_alpha),
                dropout=float(args.lora_dropout),
                targets=peft_targets,
            )
        elif args.int8_lora and not args.peft_lora:
            n = convert_linear_to_int8_lora(
                model,
                r=int(args.lora_r),
                alpha=int(args.lora_alpha),
                dropout=0.0,
                include_names=None,
                exclude_names=("emb", "head"),
                skip_if_out_equals=getattr(model, "vocab_size", None),
            )
            print(f"[INT8+LoRA] replaced Linear -> LinearInt8LoRA: {n}")
    model = model.to(device)
    total_params_for_freeze = int(sum(p.numel() for p in model.parameters()))
    auto_adapter_expected = (
        bool(args.auto_mps_7b)
        and device.type == "mps"
        and total_params_for_freeze >= 5_000_000_000
    )
    freeze_embeddings = (
        bool(args.freeze_embeddings)
        or bool(args.peft_lora)
        or bool(args.int8_lora)
        or bool(args.hybrid_lora)
        or auto_adapter_expected
    )
    if bool(args.train_embeddings):
        freeze_embeddings = False
    if freeze_embeddings:
        try:
            mod = model.base_model if hasattr(model, "base_model") else model
            for p in getattr(mod, "emb").parameters():
                p.requires_grad = False
            # head.weight is tied to emb.weight, so freezing emb freezes output too.
            print("[FREEZE] embeddings frozen (adapter training)")
        except Exception as e:
            print("[FREEZE] freeze failed:", e)
    else:
        print("[FREEZE] embeddings trainable")
    cfg = TrainConfig(
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        lr=args.lr,
        warmup_steps=args.warmup,
        grad_clip=1.0,
        max_len=args.max_len,
        out_dir=args.out_dir,
        seed=1337,
        use_tb=False,
        tiled_q=args.tiled_q,
        tiled_k=args.tiled_k,
        mps_sync_every=int(args.mps_sync_every),
        lvi_mv_weight=args.lvi_mv_weight,
        lvi_two_view_weight=args.lvi_two_view_weight,
        window_size=int(args.window_size),
        global_tokens=int(args.global_tokens),
        global_stride=int(args.global_stride),
        turbo_target_tok=float(args.turbo_target_tok),
        turbo_window=int(args.turbo_window),
        turbo_cooldown=int(args.turbo_cooldown),
        turbo_seq_floor=int(args.turbo_seq_floor),
        turbo_seq_step=int(args.turbo_seq_step),
        turbo_disable_metrics_ratio=float(args.turbo_disable_metrics_ratio),
        turbo_micro_floor=int(args.turbo_micro_floor),
        opt_cpu8bit=bool(args.opt_cpu8bit),
        opt_galore=bool(args.opt_galore),
        galore_rank=int(args.galore_rank),
        intent_weight=float(args.intent_weight),
        intent_margin=float(args.intent_margin),
        intent_sample_frac=float(args.intent_sample_frac),
        intent_every=int(args.intent_every),
        relora_every=int(args.relora_every),
        kv_heads=kvh,
        lvi_enable=bool(args.lvi),
        lvi_k=int(args.lvi_k),
        lvi_alpha_mode=str(args.lvi_alpha_mode),
        lvi_every=int(args.lvi_every),
        loader_num_workers=int(args.loader_workers),
        loader_prefetch_factor=int(args.loader_prefetch),
        loader_persistent_workers=bool(args.loader_persistent_workers),
        loader_pin_memory=bool(args.loader_pin_memory),
        prefetch_to_device=not bool(args.no_prefetch_to_device),
        disallow_mps_fallback=not bool(args.allow_mps_cpu_fallback),
        auto_mps_7b=bool(args.auto_mps_7b),
        auto_mps_target_seq=int(args.auto_mps_target_seq),
        auto_mps_token_budget=max(1, int(args.auto_mps_token_budget)),
        auto_mps_min_micro_batch=max(1, int(args.auto_mps_min_micro_batch)),
        mps_7b_lora_rank=(
            int(args.mps_7b_lora_rank) if int(args.mps_7b_lora_rank) > 0 else None
        ),
        mps_7b_lora_alpha=(
            int(args.mps_7b_lora_alpha) if int(args.mps_7b_lora_alpha) > 0 else None
        ),
        mps_7b_lora_dropout=float(args.mps_7b_lora_dropout),
        mps_7b_skip_if_out_equals=(
            int(args.mps_7b_skip_if_out_equals)
            if int(args.mps_7b_skip_if_out_equals) >= 0
            else getattr(model, "vocab_size", None)
        ),
    )
    cfg.safetensor_every = int(getattr(args, "safetensor_every", 0))
    cfg.curriculum = [
        CurriculumStage(
            until_step=int(cfg.warmup_steps * 1.0),
            max_len=min(args.max_len, max(512, args.pack_len)),
        ),
        CurriculumStage(
            until_step=int(args.max_steps * 0.5) if args.max_steps else 10**9,
            max_len=min(args.max_len, max(1024, args.pack_len * 2)),
        ),
        CurriculumStage(until_step=args.max_steps or 10**9, max_len=args.max_len),
    ]
    cfg.peft_targets = args.peft_targets
    cfg.lora_r = int(args.lora_r)
    cfg.lora_alpha = int(args.lora_alpha)
    cfg.lora_dropout = float(args.lora_dropout)

    trainer = AetherTrainerMPS(model, tok, cfg, ckpt_every=max(0, int(args.ckpt_every)))

    # TorchScript trace
    if args.ts_trace:
        L = int(args.ts_len)
        x = torch.full((1, L), tok.PAD, dtype=torch.long, device=device)
        try:
            mt = torch.jit.trace(model, (x,), check_trace=False)
            os.makedirs(os.path.dirname(args.ts_out) or "runs/ts", exist_ok=True)
            torch.jit.save(mt, args.ts_out.replace("{L}", str(L)))
            print("[TS] saved:", args.ts_out)
        except Exception as e:
            print("[TS] trace failed:", e)
    if args.ts_multi:
        lens = [int(x.strip()) for x in args.ts_multi.split(",") if x.strip()]
        for L in lens:
            x = torch.full((1, L), tok.PAD, dtype=torch.long, device=device)
            try:
                mt = torch.jit.trace(model, (x,), check_trace=False)
                out_path = args.ts_out.replace("{L}", str(L))
                os.makedirs(os.path.dirname(out_path) or "runs/ts", exist_ok=True)
                torch.jit.save(mt, out_path)
                print("[TS] saved:", out_path)
            except Exception as e:
                print("[TS] trace failed:", e)

    # Train
    # --- Preflight: how many files will be used (見える化)
    n_tr = _count_glob(args.train_glob)
    n_va = _count_glob(args.val_glob)
    print(
        f"[DATA] train files={n_tr}  val files={n_va}  pattern_train={args.train_glob}  cwd={os.getcwd()}"
    )
    if args.train_stream and n_tr == 0:
        raise SystemExit(
            "[DATA] No training files found. Check --train_glob pattern or working directory."
        )
    if args.demo:
        s = "Hello, SpiralReality."
        ids = tok.encode(s)
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            y = model(x)
        print("demo logits:", y.shape)
    elif args.train_stream:
        assert args.train_glob, "--train_glob が必要です"
        ds_tr = StreamingTextDataset(
            args.train_glob,
            tok,
            pack_len=args.pack_len,
            buffer_size=args.buffer_size,
            infinite=True,
            seed=cfg.seed,
        )
        ds_va = (
            StreamingTextDataset(
                args.val_glob,
                tok,
                pack_len=min(4096, args.max_len),
                buffer_size=4096,
                infinite=False,
                seed=cfg.seed,
            )
            if args.val_glob
            else None
        )
        trainer.fit(ds_tr, ds_va)
        if args.save_adapter:
            save_peft_adapter_if_any(trainer.model, args.save_adapter)
    else:
        print(
            "Nothing to do. Use one of: --demo | --train_stream | --ts_trace | --ts_multi"
        )


# =============================================================================
if __name__ == "__main__":
    __aether_main__()
# =============================================================================


# ===================== Safety / ANI Extensions (Non-invasive hooks) =====================
# These utilities are appended without altering existing classes or logic.
# They are activated only if environment variables are set.

import os as _os
import json as _json
import types as _types
import time as _time


def _flatten_tensors(_x):
    if torch.is_tensor(_x):
        return [_x]
    elif isinstance(_x, (list, tuple)):
        out = []
        for z in _x:
            out.extend(_flatten_tensors(z))
        return out
    elif isinstance(_x, dict):
        out = []
        for z in _x.values():
            out.extend(_flatten_tensors(z))
        return out
    return []


def _safe_stats(t: torch.Tensor):
    try:
        if not torch.is_floating_point(t):
            return None
        finite = torch.isfinite(t)
        if finite.all():
            return None
        n = t.numel()
        nf = int((~finite).sum().item())
        ratio = nf / max(1, n)
        s = {"nf": nf, "n": n, "ratio": ratio}
        with torch.no_grad():
            s["min"] = float(torch.nan_to_num(t).min().item())
            s["max"] = float(torch.nan_to_num(t).max().item())
            s["mean"] = float(torch.nan_to_num(t).float().mean().item())
            s["std"] = float(
                torch.nan_to_num((t - t.float().mean()).float()).std().item()
            )
        return s
    except Exception:
        return {"nf": -1, "n": int(t.numel()) if t is not None else 0, "ratio": 1.0}


class _AetherEventBus:
    def __init__(self):
        self.handlers = []

    def emit(self, evt: dict):
        for h in list(self.handlers):
            try:
                h(evt)
            except Exception:
                pass

    def on(self, fn):
        self.handlers.append(fn)


_AETHER_EVENT_BUS = _AetherEventBus()


class _AetherNaNGuard:
    def __init__(
        self,
        model: torch.nn.Module,
        mode: str = "raise",
        check_inputs=True,
        check_outputs=True,
        check_grads=True,
        deep_patterns=None,
        deep_dump=False,
        deep_dump_on="event",
        outdir="runs/safety",
    ):
        self.model = model
        self.mode = mode
        self.check_inputs = bool(check_inputs)
        self.check_outputs = bool(check_outputs)
        self.check_grads = bool(check_grads)
        self.deep_patterns = [
            p.strip().lower() for p in (deep_patterns or []) if p.strip()
        ]
        self.deep_dump = bool(deep_dump)
        self.deep_dump_on = str(deep_dump_on or "event")
        self.outdir = outdir
        self._names = {id(m): n for n, m in model.named_modules()}
        self._pnames = {id(p): n for n, p in model.named_parameters()}
        self._installed = False
        _os.makedirs(_os.path.join(outdir, "tensors"), exist_ok=True)
        self._anomaly = False

    def _matches_deep(self, name: str) -> bool:
        if not self.deep_patterns:
            return False
        nl = name.lower()
        return any(p in nl for p in self.deep_patterns)

    def _nan2num_(self, t: torch.Tensor):
        if not torch.is_floating_point(t):
            return t
        torch.nan_to_num(
            t,
            nan=0.0,
            posinf=float(torch.finfo(t.dtype).max),
            neginf=float(torch.finfo(t.dtype).min),
            out=t,
        )
        return t

    def _clamp_(self, t: torch.Tensor, mn=None, mx=None):
        if not torch.is_floating_point(t):
            return t
        t.nan_to_num_(nan=0.0)
        if mn is None or mx is None:
            finfo = torch.finfo(t.dtype)
            mn = finfo.min if mn is None else mn
            mx = finfo.max if mx is None else mx
        t.clamp_(mn, mx)
        return t

    def _handle(self, where: str, name: str, tensors, stage=""):
        hazard = False
        stats = []
        for t in tensors:
            s = _safe_stats(t)
            if s is not None:
                hazard = True
                stats.append(s)
                if self.mode == "nan2num":
                    self._nan2num_(t)
                elif self.mode == "clamp":
                    self._clamp_(t)
        if hazard:
            evt = {
                "type": "nonfinite",
                "where": where,
                "name": name,
                "stage": stage,
                "stats": stats,
                "time": _time.time(),
            }
            _AETHER_EVENT_BUS.emit(evt)
            if (
                (self._matches_deep(name))
                and self.deep_dump
                and (self.deep_dump_on in ("always", "event"))
            ):
                # dump a small projection to save space
                try:
                    for i, t in enumerate(
                        [x for x in tensors if torch.is_tensor(x)][:2]
                    ):
                        path = _os.path.join(
                            self.outdir,
                            "tensors",
                            f"{int(evt['time'])}_{where}_{name.replace('.', '_')}_{stage}_{i}.pt",
                        )
                        torch.save(t.detach().cpu()[:2], path)
                        jpath = path.replace(".pt", ".json")
                        with open(jpath, "w") as f:
                            import json as __json

                            __json.dump(
                                {
                                    "name": name,
                                    "where": where,
                                    "stage": stage,
                                    "stats": stats,
                                },
                                f,
                            )
                except Exception:
                    pass
            if self.mode == "raise":
                raise FloatingPointError(
                    f"[NaNGuard] non-finite detected at {where}:{name} {stage} stats={stats[:1]}"
                )

    def _pre_hook(self, module, inputs):
        if not self.check_inputs:
            return
        name = self._names.get(id(module), module.__class__.__name__)
        tensors = _flatten_tensors(inputs)
        self._handle("forward_in", name, tensors, stage="pre")

    def _fwd_hook(self, module, inputs, outputs):
        if not self.check_outputs:
            return
        name = self._names.get(id(module), module.__class__.__name__)
        tensors = _flatten_tensors(outputs)
        # modify in-place for outputs if needed
        self._handle("forward_out", name, tensors, stage="post")

    def _grad_hook(self, p):
        def fn(g):
            if not self.check_grads:
                return g
            name = self._pnames.get(id(p), "param")
            s = _safe_stats(g)
            if s is not None:
                evt = {
                    "type": "nonfinite",
                    "where": "grad",
                    "name": name,
                    "stats": [s],
                    "time": _time.time(),
                }
                _AETHER_EVENT_BUS.emit(evt)
                if self.mode == "nan2num":
                    torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0, out=g)
                elif self.mode == "clamp":
                    self._clamp_(g)
                elif self.mode == "raise":
                    raise FloatingPointError(
                        f"[NaNGuard] non-finite grad at {name} stats={s}"
                    )
            return g

        return fn

    def install(self, anomaly=False):
        if self._installed:
            return
        self._anomaly = bool(anomaly)
        if self._anomaly:
            try:
                torch.autograd.set_detect_anomaly(True)
            except Exception:
                pass
        for _, m in self.model.named_modules():
            try:
                m.register_forward_pre_hook(self._pre_hook, with_kwargs=False)
                m.register_forward_hook(self._fwd_hook, with_kwargs=False)
            except Exception:
                pass
        if self.check_grads:
            for p in self.model.parameters():
                if p.requires_grad:
                    try:
                        p.register_hook(self._grad_hook(p))
                    except Exception:
                        pass
        self._installed = True


class _AdaptiveBatchIter:
    """Wraps an IterableDataset to yield dynamic batch sizes driven by a callback."""

    def __init__(self, ds: IterableDataset, collate_fn, get_bs, shuffle: bool):
        self.ds = ds
        self.collate_fn = collate_fn
        self.get_bs = get_bs
        self.shuffle = shuffle
        self._it = None

    def __iter__(self):
        self._it = iter(self.ds)
        return self

    def __next__(self):
        bs = max(1, int(self.get_bs()))
        batch = []
        while len(batch) < bs:
            try:
                item = next(self._it)
            except StopIteration:
                # Recreate iterator if dataset is infinite; otherwise propagate
                if getattr(self.ds, "infinite", False):
                    self._it = iter(self.ds)
                    continue
                if len(batch) == 0:
                    raise
                else:
                    break
            batch.append(item)
        if len(batch) == 0:
            raise StopIteration
        return self.collate_fn(batch)

    def __len__(self):
        # Unknown for streaming; let caller handle
        raise TypeError


class _AetherANIManager:
    """Adaptive Nonfinite Intervention: dynamic batch shrink + loss scaling + event-driven escalation."""

    def __init__(self, trainer, enable: bool = True, outdir="runs/safety"):
        self.trainer = trainer
        self.model = trainer.model
        self.cfg = trainer.cfg
        self.outdir = outdir
        self.enabled = bool(enable)
        self.base_bs = int(getattr(self.cfg, "batch_size", 1))
        self.bs_div = 1  # effective_bs = base_bs // bs_div
        self.level = 0
        self.max_level = int(_os.getenv("AETHER_ANI_MAX_ESCALATION", "3"))
        self.patience = int(_os.getenv("AETHER_ANI_PATIENCE", "2"))
        self.cooldown = int(_os.getenv("AETHER_ANI_COOLDOWN", "200"))
        self.loss_scale = float(_os.getenv("AETHER_ANI_LOSS_SCALE", "1.0"))
        self.min_loss_scale = float(_os.getenv("AETHER_ANI_MIN_LOSS_SCALE", "0.0625"))
        self.scale_backoff = float(_os.getenv("AETHER_ANI_SCALE_BACKOFF", "0.5"))
        self.scale_growth = float(_os.getenv("AETHER_ANI_SCALE_GROWTH", "2.0"))
        self.last_hazard_step = -(10**9)
        self.step_counter = 0
        self.hazard_in_window = 0
        self.lr_backoff = float(
            _os.getenv("AETHER_ANI_LR_BACKOFF", "1.0")
        )  # no LR change by default
        self.skip_on_hazard = bool(int(_os.getenv("AETHER_ANI_SKIP_ON_HAZARD", "0")))
        self.grad_clip_val = float(_os.getenv("AETHER_ANI_GRAD_CLIP", "0.0"))
        self._install_log_path = _os.path.join(outdir, "ani_events.jsonl")
        _os.makedirs(outdir, exist_ok=True)
        self._install_logging()
        self._patch_clip_and_step()
        self._subscribe_events()

    # ---- logging ----
    def _install_logging(self):
        try:
            with open(self._install_log_path, "a") as f:
                f.write(
                    _json.dumps(
                        {
                            "t": _time.time(),
                            "event": "_install",
                            "base_bs": self.base_bs,
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    def log(self, d):
        try:
            with open(self._install_log_path, "a") as f:
                f.write(_json.dumps(d) + "\n")
        except Exception:
            pass

    # ---- batch size control ----
    def effective_bs(self) -> int:
        return max(1, self.base_bs // max(1, self.bs_div))

    def relax_if_stable(self):
        # called periodically by caller (e.g., each optimizer step)
        self.step_counter += 1
        since = self.step_counter - self.last_hazard_step
        if since >= max(self.cooldown, 1) and self.level > 0:
            self.level -= 1
            self.bs_div = max(1, self.bs_div // 2)
            self.loss_scale = min(1.0, self.loss_scale * self.scale_growth)
            self.log(
                {
                    "t": _time.time(),
                    "event": "relax",
                    "level": self.level,
                    "bs": self.effective_bs(),
                    "scale": self.loss_scale,
                }
            )

    # ---- hazard handling ----
    def _escalate(self, reason: str):
        if self.level < self.max_level:
            self.level += 1
            self.bs_div = min(2**self.level, max(1, self.base_bs))  # ceiling
            self.loss_scale = max(
                self.min_loss_scale, self.loss_scale * self.scale_backoff
            )
            # optional LR backoff
            if self.lr_backoff < 1.0:
                for g in self.trainer.opt.param_groups:
                    g["lr"] = g.get("_base_lr", g["lr"]) * self.lr_backoff
                    g["_base_lr"] = g["lr"]
            self.log(
                {
                    "t": _time.time(),
                    "event": "escalate",
                    "reason": reason,
                    "level": self.level,
                    "bs": self.effective_bs(),
                    "scale": self.loss_scale,
                }
            )

    def on_nonfinite(self, evt):
        self.last_hazard_step = self.step_counter
        self.hazard_in_window += 1
        self._escalate(f"{evt.get('where')}:{evt.get('name')}")
        if self.skip_on_hazard:
            # zero grads + skip this step by zeroing grads (training loop will still step, but grads are null)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    # ---- gradient scaling integration ----
    def unscale_grads(self, params_iter=None):
        if abs(self.loss_scale - 1.0) < 1e-6:
            return
        with torch.no_grad():
            ps = (
                list(self.model.parameters())
                if params_iter is None
                else list(params_iter)
            )
            for p in ps:
                if p.grad is not None:
                    p.grad.div_(self.loss_scale)

    def _patch_clip_and_step(self):
        # Legacy compatibility hook. ANI grad preparation now lives at the
        # trainer's optimizer boundary, before clipping and stepping.
        return

    # ---- dataloader patch ----
    def make_loader(self, ds, shuffle, max_len_for_collate):
        # If dataset is streaming (IterableDataset), return adaptive loader.
        is_iter = isinstance(ds, IterableDataset)
        if not is_iter:
            # fall back to original behavior
            return DataLoader(
                ds,
                batch_size=self.trainer.cfg.batch_size,
                shuffle=bool(shuffle),
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                collate_fn=lambda b: collate_lm_safe(b, pad_id=self.trainer.tok.PAD),
            )
        collate_fn = lambda b: collate_lm_safe(b, pad_id=self.trainer.tok.PAD)
        return _AdaptiveBatchIter(
            ds, collate_fn, get_bs=self.effective_bs, shuffle=bool(shuffle)
        )


def _install_nan_guard_and_ani_from_env(trainer):
    """Installer. Call once after trainer is constructed."""
    if int(_os.getenv("AETHER_NAN_GUARD", "0")):
        mode = _os.getenv("AETHER_NAN_GUARD_MODE", "raise")
        chk_in = bool(int(_os.getenv("AETHER_NAN_GUARD_CHECK_INPUTS", "1")))
        chk_out = bool(int(_os.getenv("AETHER_NAN_GUARD_CHECK_OUTPUTS", "1")))
        chk_grads = bool(int(_os.getenv("AETHER_NAN_GUARD_CHECK_GRADS", "1")))
        deep_en = bool(int(_os.getenv("AETHER_DEEP_ENABLE", "0")))
        deep_pats = (
            _os.getenv("AETHER_DEEP_PATTERNS", "attn,layernorm,loss").split(",")
            if deep_en
            else []
        )
        deep_dump = bool(int(_os.getenv("AETHER_DEEP_DUMP", "0")))
        deep_on = _os.getenv("AETHER_DEEP_DUMP_ON", "event")
        guard = _AetherNaNGuard(
            trainer.model,
            mode,
            chk_in,
            chk_out,
            chk_grads,
            deep_pats,
            deep_dump,
            deep_on,
        )
        guard.install(anomaly=bool(int(_os.getenv("AETHER_NAN_GUARD_ANOMALY", "0"))))
    if int(_os.getenv("AETHER_ANI_ENABLE", "0")):
        manager = _AetherANIManager(trainer, enable=True)
        trainer._ani_manager = manager
        # subscribe to NaN guard events
        _AETHER_EVENT_BUS.on(manager.on_nonfinite)
        # patch loader (instance-level)
        try:

            def _patched_make_loader(self, ds, shuffle, max_len_for_collate):
                return manager.make_loader(ds, shuffle, max_len_for_collate)

            trainer._make_loader = _types.MethodType(_patched_make_loader, trainer)
        except Exception:
            pass
        # Loss scaling is applied by AetherTrainerMPS._backward_with_guard().
        # Avoid global torch.Tensor.backward monkey patches here.


# Inject installer call inside __aether_main__ dynamically by patching the function body at runtime is complex.
# Instead, we call the installer immediately after trainer creation by monkey-patching AetherTrainerMPS.__init__.
try:
    _orig_init = AetherTrainerMPS.__init__

    def _patched_init(self, *a, **k):
        _orig_init(self, *a, **k)
        try:
            _install_nan_guard_and_ani_from_env(self)
        except Exception as _e:
            print("[ANI/NAN] installer failed:", _e)

    AetherTrainerMPS.__init__ = _patched_init
except Exception as _e:
    print("[ANI/NAN] __init__ patch failed:", _e)

# ===================== End of Safety / ANI Extensions =====================
# =============================================================================
# SpiralGuardian — passive NaN/gradient failsafe
#  - ANI owns active treatment and loss-scale correction at trainer boundaries
#  - Guardian watches loss spikes and grad collapse/explosion
#  - optional last-resort escalation: lr backoff, clipping, step skip, snapshots
#  - trainer hooks are optional and local to the trainer instance
# =============================================================================
import collections
from typing import Dict, Optional
import torch
import torch.nn as nn


def _sg_now():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _sg_env_b(name: str, default: int = 0) -> bool:
    return str(os.environ.get(name, str(default))).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _sg_env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _sg_env_i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _sg_env_s(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _sg_cfg_from_env() -> Dict[str, Any]:
    droot = _sg_env_s("AETHER_SAFETY_DUMP_DIR", "runs/safety")
    os.makedirs(droot, exist_ok=True)
    os.makedirs(os.path.join(droot, "guardian"), exist_ok=True)
    os.makedirs(os.path.join(droot, "snaps"), exist_ok=True)
    return {
        "enable": _sg_env_b("AETHER_GUARDIAN", 1),
        # loss spike detector
        "ema_alpha": _sg_env_f("AETHER_GUARDIAN_EMA_ALPHA", 0.10),
        "spike_mult": _sg_env_f(
            "AETHER_GUARDIAN_LOSS_SPIKE_EMA_X", 1.75
        ),  # loss > ema*X
        "spike_delta": _sg_env_f(
            "AETHER_GUARDIAN_LOSS_SPIKE_DELTA", 0.20
        ),  # (loss-last)/max(last,eps) > delta
        "rise_window": _sg_env_i("AETHER_GUARDIAN_RISE_STEPS", 3),  # 連続上昇で発火
        # grad monitor
        "grad_min": _sg_env_f("AETHER_GUARDIAN_GRAD_MIN", 1e-7),
        "grad_max": _sg_env_f("AETHER_GUARDIAN_GRAD_MAX", 50.0),
        "grad_sample": _sg_env_i("AETHER_GUARDIAN_GRAD_SAMPLES", 128),  # 0=全件
        # staged actions
        "lr_backoff": _sg_env_f("AETHER_GUARDIAN_LR_BACKOFF", 0.5),
        "scale_backoff": _sg_env_f("AETHER_GUARDIAN_SCALE_BACKOFF", 0.5),
        "scale_min": _sg_env_f("AETHER_GUARDIAN_SCALE_MIN", 0.015625),
        "clip_on": _sg_env_b("AETHER_GUARDIAN_CLIP_ON", 1),
        "clip_value": _sg_env_f("AETHER_GUARDIAN_CLIP", 1.0),
        "skip_on": _sg_env_b("AETHER_GUARDIAN_SKIP", 1),
        "max_level": _sg_env_i("AETHER_GUARDIAN_MAX_LEVEL", 3),
        "patience": _sg_env_i("AETHER_GUARDIAN_PATIENCE", 1),  # 何件で次レベル
        "stop_after": _sg_env_i("AETHER_GUARDIAN_STOP_AFTER", 8),
        "cooldown": _sg_env_i("AETHER_GUARDIAN_COOLDOWN", 200),
        # snapshots
        "snap_every": _sg_env_i("AETHER_GUARDIAN_SNAP_EVERY", 500),
        "dump_dir": droot,
        "verbose": _sg_env_b("AETHER_GUARDIAN_VERBOSE", 1),
        # wire-up
        "wrap_opt": _sg_env_b("AETHER_GUARDIAN_WRAP_OPT", 1),
        "wrap_loss": _sg_env_b("AETHER_GUARDIAN_WRAP_LOSS", 1),
    }


class SpiralGuardian:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler=None,
        step_provider=None,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.opt = optimizer
        self.sched = scheduler
        self.cfg = cfg or _sg_cfg_from_env()
        self.step_provider = step_provider
        self.dump_dir = os.path.join(self.cfg["dump_dir"], "guardian")
        os.makedirs(self.dump_dir, exist_ok=True)
        # state
        self.level = 0
        self.haz_count = 0
        self._stable = 0
        self._rise = 0
        self._ema = None
        self._last_loss = None
        self._skip_budget = 0
        self._snaps = collections.deque(maxlen=5)
        # patch optimizer
        if self.cfg["wrap_opt"] and self.opt is not None:
            self._wrap_optimizer()
        if self.cfg["verbose"]:
            print(
                "[GUARD] SpiralGuardian active | cfg:",
                {
                    k: self.cfg[k]
                    for k in (
                        "spike_mult",
                        "spike_delta",
                        "grad_min",
                        "grad_max",
                        "lr_backoff",
                        "scale_backoff",
                        "clip_value",
                        "patience",
                    )
                },
            )

    # ---- public hooks --------------------------------------------------------
    def observe_loss(self, loss: torch.Tensor):
        try:
            val = float(loss.detach().to("cpu"))
        except Exception:
            return
        st = self.step()
        # ema
        if self._ema is None:
            self._ema = val
        a = float(self.cfg["ema_alpha"])
        self._ema = (1.0 - a) * self._ema + a * val
        # spike heuristics
        spike_ema = val > self._ema * float(self.cfg["spike_mult"])
        spike_jump = False
        if self._last_loss is not None:
            denom = max(1e-9, abs(self._last_loss))
            spike_jump = ((val - self._last_loss) / denom) > float(
                self.cfg["spike_delta"]
            )
            self._rise = self._rise + 1 if val > self._last_loss else 0
        sustained = self._rise >= int(self.cfg["rise_window"])
        self._last_loss = val

        if spike_ema or spike_jump or sustained:
            self._report(
                "loss_spike",
                {
                    "step": st,
                    "loss": val,
                    "ema": self._ema,
                    "spike_ema": bool(spike_ema),
                    "spike_jump": bool(spike_jump),
                    "sustained": bool(sustained),
                },
            )

    def after_backward(self):
        # grad stats (sampled)
        gmin, gmax, gmean = None, None, 0.0
        count = 0
        limit = int(self.cfg["grad_sample"])
        for p in self.model.parameters():
            if p.grad is None or not p.grad.is_floating_point():
                continue
            gn = float(p.grad.detach().data.norm().cpu())
            gmin = gn if gmin is None else min(gmin, gn)
            gmax = gn if gmax is None else max(gmax, gn)
            gmean += gn
            count += 1
            if limit > 0 and count >= limit:
                break
        if count > 0:
            gmean /= count
            if not math.isfinite(gmean) or (
                gmax is not None and not math.isfinite(gmax)
            ):
                self._report("grad_nonfinite", {"gmean": gmean, "gmax": gmax})
            else:
                if gmean < float(self.cfg["grad_min"]):
                    self._report("grad_collapse", {"gmean": gmean})
                if gmax is not None and gmax > float(self.cfg["grad_max"]):
                    self._report("grad_explosion", {"gmax": gmax})

    # ---- core ---------------------------------------------------------------
    def step(self) -> int:
        try:
            if self.step_provider:
                return int(self.step_provider())
        except Exception:
            pass
        return int(getattr(self, "_fallback_step", 0))

    def _report(self, kind: str, payload: Dict[str, Any]):
        self.haz_count += 1
        if self.cfg["verbose"]:
            print(f"[GUARD] hazard {kind} @step={self.step()} :: {payload}")
        # write context
        self._write_context(kind, payload)
        # escalate w/ patience
        if (self.haz_count % max(1, int(self.cfg["patience"]))) == 0:
            self._escalate(kind)

        # stop criteria
        if self.haz_count >= int(self.cfg["stop_after"]):
            fn = self._snapshot(reason="stop_after")
            print("[GUARD] stop-after reached; snapshot:", fn)
            raise SystemExit(2)

    def _escalate(self, reason: str):
        self.level = min(int(self.cfg["max_level"]), self.level + 1)
        if self.cfg["verbose"]:
            print(f"[GUARD] ESCALATE → L{self.level} ({reason})")
        # staged actions
        self._apply_lr_backoff()
        self._apply_scale_backoff()
        if self.cfg["clip_on"]:
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg["clip_value"])
                )
            except Exception:
                pass
        if self.cfg["skip_on"]:
            self._skip_budget = max(
                self._skip_budget, 1
            )  # skip next step at least once
        # snapshot on every escalation
        self._snapshot(reason=f"escalate_L{self.level}")

    def _forgive(self):
        # cooldown forgiveness: decay level, restore scales partly
        if self.level > 0:
            self.level -= 1
        if self.cfg["verbose"]:
            print(f"[GUARD] FORGIVE → L{self.level}")
        # allow loss_scale growth next steps; lr gradually returns via sched

    # ---- optimizer wrapping --------------------------------------------------
    def _wrap_optimizer(self):
        if getattr(self.opt, "_sg_wrapped", False):
            return
        orig_step = self.opt.step
        guardian = self

        def wrapped_step(*args, **kwargs):
            # pre
            if guardian._skip_budget > 0:
                if guardian.cfg["verbose"]:
                    print("[GUARD] skip optimizer.step() due to hazard")
                guardian._skip_budget -= 1
                guardian.opt.zero_grad(set_to_none=True)
                return
            # pre-check grads
            bad = False
            for p in guardian.model.parameters():
                g = p.grad
                if g is None or not g.is_floating_point():
                    continue
                if not torch.isfinite(g).all():
                    bad = True
                    break
            if bad:
                guardian._report("grad_nonfinite_pre_step", {})
                guardian.opt.zero_grad(set_to_none=True)
                return
            # step
            res = orig_step(*args, **kwargs)
            # post: sanity of params
            badp = False
            for p in guardian.model.parameters():
                if not p.is_floating_point():
                    continue
                if not torch.isfinite(p).all():
                    badp = True
                    break
            if badp:
                guardian._report("param_nonfinite_post_step", {})
                # restore last snap if any
                if guardian._snaps:
                    guardian._restore(guardian._snaps[-1])
                    guardian.opt.zero_grad(set_to_none=True)
            # periodic snap + cooldown
            guardian._cooldown_tick()
            guardian._periodic_snap()
            return res

        self.opt.step = wrapped_step
        self.opt._sg_wrapped = True
        if self.cfg["verbose"]:
            print("[GUARD] optimizer wrapped")

    def _apply_lr_backoff(self):
        try:
            for pg in self.opt.param_groups:
                base = pg.get("_base_lr", pg.get("lr", 1e-3))
                if "_base_lr" not in pg:
                    pg["_base_lr"] = base
                factor = float(self.cfg["lr_backoff"]) ** max(1, self.level)
                pg["lr"] = float(base) * factor
        except Exception:
            pass

    def _apply_scale_backoff(self):
        # AMP loss-scaler互換（trainer側にscaleがあれば使う）
        try:
            scaler = getattr(self, "scaler", None)
            if scaler is None and hasattr(self.opt, "loss_scaler"):
                scaler = getattr(self.opt, "loss_scaler", None)
            if scaler is None:
                # try global in trainer (AETHER_ANI_LOSS_SCALE 系と連携)
                scaler = getattr(self, "_trainer", None)
                if scaler is not None:
                    scaler = getattr(scaler, "loss_scaler", None)
            if scaler is None:
                return
            cur = float(getattr(scaler, "scale", getattr(scaler, "_scale", 1.0)))
            new = max(
                float(self.cfg["scale_min"]), cur * float(self.cfg["scale_backoff"])
            )
            try:
                setattr(scaler, "scale", new)
            except Exception:
                try:
                    scaler._scale = torch.tensor(new, device="cpu")
                except Exception:
                    pass
            if self.cfg["verbose"]:
                print(f"[GUARD] loss_scale backoff {cur} → {new}")
        except Exception:
            pass

    # ---- snapshots & logs ----------------------------------------------------
    def _snapshot(self, reason="manual"):
        try:
            st = self.step()
            path = os.path.join(
                self.cfg["dump_dir"],
                "snaps",
                f"{_sg_now()}_step{st}_L{self.level}_{reason}.pt",
            )
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.opt.state_dict()
                    if self.opt is not None
                    else None,
                    "meta": {"step": st, "level": self.level, "reason": reason},
                },
                path,
            )
            self._snaps.append(path)
            while len(self._snaps) > 5:
                self._snaps.popleft()
            if self.cfg["verbose"]:
                print("[GUARD] snapshot:", path)
            return path
        except Exception as e:
            print("[GUARD] snapshot failed:", e)

    def _restore(self, path: str):
        try:
            data = torch.load(path, map_location="cpu")
            self.model.load_state_dict(data.get("model", {}))
            if self.opt is not None and data.get("optimizer") is not None:
                try:
                    self.opt.load_state_dict(data["optimizer"])
                except Exception as e:
                    print("[GUARD] opt restore failed:", e)
            if self.cfg["verbose"]:
                print("[GUARD] restored:", path)
        except Exception as e:
            print("[GUARD] restore failed:", e)

    def _write_context(self, kind, payload):
        try:
            st = self.step()
            meta = {
                "t": _sg_now(),
                "step": st,
                "kind": kind,
                "payload": payload,
                "level": self.level,
                "lr_groups": [
                    pg.get("lr", None) for pg in getattr(self.opt, "param_groups", [])
                ],
                "clip": float(self.cfg["clip_value"]),
                "scale_min": float(self.cfg["scale_min"]),
            }
            with open(
                os.path.join(self.dump_dir, f"{_sg_now()}_step{st}_{kind}.json"), "w"
            ) as f:
                json.dump(meta, f)
        except Exception:
            pass

    def _cooldown_tick(self):
        self._stable += 1
        if self._stable >= int(self.cfg["cooldown"]):
            self._stable = 0
            # forgive one level
            self._forgive()

    def _periodic_snap(self):
        st = self.step()
        se = int(self.cfg["snap_every"])
        if se > 0 and (st % se) == 0:
            self._snapshot(reason="periodic")


# ---- installer ---------------------------------------------------------------
def _install_spiral_guardian_from_env(trainer):
    cfg = _sg_cfg_from_env()
    if not cfg["enable"]:
        return None
    try:
        model = getattr(trainer, "model", None)
        opt = getattr(trainer, "opt", None)
        sched = getattr(trainer, "sched", None) if hasattr(trainer, "sched") else None
        guard = SpiralGuardian(
            model,
            opt,
            sched,
            step_provider=lambda: int(getattr(trainer, "_global_step", 0)),
            cfg=cfg,
        )
        # wire loss observation if requested
        if cfg["wrap_loss"]:
            try:
                _orig = trainer._ce_loss

                def _wrap(self, logits, targets, pad_id):
                    out = _orig(logits, targets, pad_id)
                    guard.observe_loss(out)
                    return out

                import types as _types

                trainer._ce_loss = _types.MethodType(_wrap, trainer)
                print("[GUARD] loss wrapper installed")
            except Exception as e:
                print("[GUARD] loss wrapper failed:", e)
        # wire backward hook
        try:
            _orig_bw = (
                trainer.fit_one_batch if hasattr(trainer, "fit_one_batch") else None
            )
        except Exception:
            _orig_bw = None
        # we can call after_backward() from trainer loop; if not, offer a public handle
        setattr(trainer, "guardian_after_backward", lambda: guard.after_backward())
        # stash for external control
        setattr(trainer, "_spiral_guardian", guard)
        print("[GUARD] SpiralGuardian installed")
        return guard
    except Exception as e:
        print("[GUARD] install failed:", e)
        return None
