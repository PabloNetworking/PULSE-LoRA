"""
Model builder for GLUE tasks with LoRA adapter support.

Supports DeBERTa, BERT, RoBERTa, and other transformer models for
sequence classification and regression tasks on GLUE benchmarks.
"""
import logging
import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GLUEAdapterModel(nn.Module):
    """
    A wrapper class for sequence classification models with adapter (LoRA) support.

    Modelled on FederatedScope-LLM's ``AdapterModel``, for sequence classification.
    """

    def __init__(
        self,
        model: nn.Module,
        use_adapter: bool = False,
        **kwargs
    ):
        """
        Initialize the GLUE adapter model wrapper.

        Args:
            model: Base transformer model for sequence classification
            use_adapter: Whether to apply LoRA adapter
            **kwargs: Adapter configuration (adapter_package, adapter_method, etc.)
        """
        super().__init__()

        self.model = None
        if use_adapter:
            adapter_package = kwargs.pop('adapter_package', 'peft')
            adapter_method = kwargs.pop('adapter_method', 'lora')

            self.model = enable_glue_adapter(
                model, adapter_package, adapter_method, **kwargs
            )
        else:
            self.model = model

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model.forward(*args, **kwargs)

    def state_dict(self, return_trainable: bool = True, *args, **kwargs):
        """
        Get state dict, optionally filtered to trainable parameters only.

        Args:
            return_trainable: If True, return only parameters that require grad

        Returns:
            State dictionary
        """
        if return_trainable:
            return self.get_trainable_state_dict()
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = False):
        """Load state dict into the model."""
        return self.model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self):
        """Get only the trainable parameters from the model."""
        grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)

        model_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict

    def save_model(self, path: str, state: int = 0):
        """Save the model checkpoint."""
        ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)


def enable_glue_adapter(
    model: nn.Module,
    package: str,
    adapter: str,
    **kwargs
) -> nn.Module:
    """
    Enable a LoRA adapter for a GLUE classification model.

    Args:
        model: Pre-trained sequence classification model
        package: Adapter package ('peft')
        adapter: Adapter method ('lora')
        **kwargs: Adapter configuration (r, lora_alpha, target_modules, etc.)

    Returns:
        Model with LoRA adapter applied
    """
    adapter = adapter.lower()

    if package == 'peft':
        from peft import get_peft_model, TaskType

        if adapter == 'lora':
            from peft import LoraConfig

            target_modules = kwargs.pop('target_modules', None)
            if target_modules is None:
                # Without explicit target modules, choose them by architecture
                model_type = getattr(model.config, 'model_type', '').lower()
                if 'deberta' in model_type:
                    target_modules = ['query_proj', 'key_proj', 'value_proj',
                                     'dense', 'output.dense']
                elif 'bert' in model_type:
                    target_modules = ['query', 'key', 'value', 'dense']
                elif 'roberta' in model_type:
                    target_modules = ['query', 'key', 'value', 'dense']
                else:
                    # Generic fallback
                    target_modules = ['query', 'key', 'value']
                logger.info(f"Auto-detected target modules for {model_type}: {target_modules}")

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=kwargs.get('r', 8),
                lora_alpha=kwargs.get('lora_alpha', 16),
                lora_dropout=kwargs.get('lora_dropout', 0.05),
                target_modules=target_modules,
                bias="none",
            )

            model = get_peft_model(model, peft_config)

            model.print_trainable_parameters()

        else:
            raise NotImplementedError(f"Adapter {adapter} not implemented for GLUE")
    else:
        raise NotImplementedError(f"Package {package} not supported for GLUE")

    return model


def get_model_from_huggingface_for_glue(
    model_name: str,
    config,
    num_labels: int = 2,
) -> nn.Module:
    """
    Load a sequence classification model from HuggingFace.

    Supports DeBERTa, BERT, RoBERTa, and other AutoModelForSequenceClassification
    compatible models. The model is loaded in fp32 with gradient checkpointing
    enabled.

    Args:
        model_name: HuggingFace model name or path
        config: Configuration object
        num_labels: Number of output labels for classification

    Returns:
        Sequence classification model
    """
    from transformers import AutoModelForSequenceClassification

    kwargs = {}
    if hasattr(config, 'glue') and hasattr(config.glue, 'cache_dir'):
        kwargs['cache_dir'] = config.glue.cache_dir
    elif hasattr(config.data, 'root'):
        kwargs['cache_dir'] = config.data.root

    if num_labels == 1:
        kwargs['problem_type'] = 'regression'

    # For models like DeBERTa, avoid device_map="auto" and place the full model on a single device.
    device = config.device if hasattr(config, 'device') else 0
    if isinstance(device, int) and device < 0:
        device = "cpu"
    device_map = {"": device}

    logger.info(f"Loading {model_name} in fp32 on {device}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        device_map=device_map,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        **kwargs
    )

    # Gradient checkpointing bounds the activation memory.
    try:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            if hasattr(model.config, "gradient_checkpointing"):
                model.config.gradient_checkpointing = True
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
    except Exception:
        logger.exception("Failed to enable gradient checkpointing")

    return model


def get_glue_model(config):
    """
    Build a GLUE model with optional LoRA adapter.

    Called by FederatedScope's ``get_model`` for ``@glue`` data types.

    Args:
        config: Configuration object containing model and adapter settings

    Returns:
        GLUEAdapterModel with optional LoRA adapter configured
    """
    model_config = config.model
    model_name, model_hub = model_config.type.split('@')

    # Determine number of labels from config
    num_labels = getattr(model_config, 'out_channels', 2)

    if model_hub in ['huggingface_llm', 'huggingface']:
        model = get_model_from_huggingface_for_glue(
            model_name=model_name,
            config=config,
            num_labels=num_labels,
        )
    else:
        raise NotImplementedError(f"Model hub '{model_hub}' not supported for GLUE")

    if hasattr(model, 'config'):
        if num_labels == 1:
            model.config.problem_type = 'regression'
        else:
            model.config.problem_type = 'single_label_classification'

    # Get adapter configuration
    # Look for adapter config in either glue.adapter or llm.adapter
    adapter_cfg = None
    if hasattr(config, 'glue') and hasattr(config.glue, 'adapter'):
        adapter_cfg = config.glue.adapter
    elif hasattr(config, 'llm') and hasattr(config.llm, 'adapter'):
        adapter_cfg = config.llm.adapter

    if adapter_cfg is None or not getattr(adapter_cfg, 'use', False):
        # No adapter, return model wrapped
        return GLUEAdapterModel(model, use_adapter=False)

    # Configure adapter arguments
    base_args = adapter_cfg.args[0] if adapter_cfg.args else {}
    args = dict(base_args)

    # The adapter is built at rank max_rank
    max_rank = getattr(adapter_cfg, 'max_rank', None)
    if max_rank is not None:
        original_r = args.get('r', None)
        if original_r != max_rank:
            logger.warning(
                f"Overriding LoRA rank from {original_r} to {max_rank} "
                f"to match glue.adapter.max_rank"
            )
        args['r'] = max_rank

    # Build model with adapter
    model = GLUEAdapterModel(model, use_adapter=adapter_cfg.use, **args)

    return model
