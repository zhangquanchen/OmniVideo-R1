# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn.functional as F


def _apply_conv3d_patch():
    """Apply Conv3d patch for PyTorch 2.9 to avoid performance regression."""
    if not torch.__version__.startswith("2.9"):
        return

    try:
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe

        def _patched_forward(self, x):
            """Use matrix multiplication instead of conv3d (reference: vLLM implementation)."""
            L, C = x.shape
            x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
            B, C, T, H, W = x.shape
            K = self.proj.kernel_size
            x = x.unfold(2, K[0], K[0]).unfold(3, K[1], K[1]).unfold(4, K[2], K[2])
            x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, C * K[0] * K[1] * K[2])
            x = F.linear(x, self.proj.weight.view(self.embed_dim, -1), self.proj.bias)
            return x.view(L, self.embed_dim)

        modeling_qwen3_omni_moe.Qwen3OmniMoeVisionPatchEmbed.forward = _patched_forward
        print("✅ [ms-swift] Patched Qwen3OmniMoeVisionPatchEmbed.forward for PyTorch 2.9 conv3d workaround")
    except ImportError:
        pass  # transformers version may not have this model
    except Exception as e:
        print(f"⚠️ [ms-swift] Failed to apply conv3d patch: {e}")


try:
    from transformers.utils import is_torch_npu_available

    if is_torch_npu_available():
        # Enable Megatron on Ascend NPU
        import mindspeed.megatron_adaptor  # F401
    from .init import init_megatron_env
    init_megatron_env()
    _apply_conv3d_patch()
except Exception:
    # allows lint pass.
    raise

from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .train import megatron_sft_main, megatron_pt_main, megatron_rlhf_main
    from .export import megatron_export_main
    from .convert import convert_hf2mcore, convert_mcore2hf
    from .utils import prepare_mcore_model, adapter_state_dict_context, convert_hf_config
    from .argument import MegatronTrainArguments, MegatronRLHFArguments, MegatronExportArguments, MegatronArguments
    from .model import MegatronModelType, MegatronModelMeta, get_megatron_model_meta, register_megatron_model
    from .trainers import MegatronTrainer, MegatronDPOTrainer
    from .tuners import LoraParallelLinear
else:
    _import_structure = {
        'train': ['megatron_sft_main', 'megatron_pt_main', 'megatron_rlhf_main'],
        'export': ['megatron_export_main'],
        'convert': ['convert_hf2mcore', 'convert_mcore2hf'],
        'utils': ['prepare_mcore_model', 'adapter_state_dict_context', 'convert_hf_config'],
        'argument': ['MegatronTrainArguments', 'MegatronRLHFArguments', 'MegatronExportArguments', 'MegatronArguments'],
        'model': ['MegatronModelType', 'MegatronModelMeta', 'get_megatron_model_meta', 'register_megatron_model'],
        'trainers': ['MegatronTrainer', 'MegatronDPOTrainer'],
        'tuners': ['LoraParallelLinear'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
