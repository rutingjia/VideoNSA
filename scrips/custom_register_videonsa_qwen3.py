"""Register VideoNSA Qwen3 model_type for environments where swift lacks built-in mapping."""

import importlib.util
import sys
import types
from pathlib import Path

from swift.llm import TemplateType
from swift.llm.model.constant import MLLMModelType
from swift.llm.model.model.qwen2_vl import get_model_tokenizer_qwen2_vl
from swift.llm.model.model_arch import ModelArch
from swift.llm.model.register import Model, ModelGroup, ModelMeta, register_model


def get_model_tokenizer_videonsa_qwen3(*args, **kwargs):
    """Load local Qwen3-VL modeling and register VideoNSA Qwen3 tokenizer factory."""
    module_name = 'transformers.models.qwen3_vl.modeling_qwen3_vl'
    if module_name not in sys.modules:
        pkg_qwen3_name = 'transformers.models.qwen3_vl'
        if pkg_qwen3_name not in sys.modules:
            pkg = types.ModuleType(pkg_qwen3_name)
            pkg.__path__ = []
            sys.modules[pkg_qwen3_name] = pkg

        root = Path(__file__).resolve().parents[1]
        local_modeling = root / 'third_party' / 'transformers_qwen3_vl' / 'modeling_qwen3_vl.py'
        if not local_modeling.exists():
            raise RuntimeError(f'Local Qwen3-VL modeling file not found: {local_modeling}')

        spec = importlib.util.spec_from_file_location(module_name, local_modeling)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Failed to create module spec for: {local_modeling}')

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    kwargs['automodel_class'] = Qwen3VLForConditionalGeneration
    return get_model_tokenizer_qwen2_vl(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.videonsa_qwen3,
        [
            ModelGroup(
                [
                    Model('Qwen/Qwen3-VL-2B-Instruct'),
                    Model('Qwen/Qwen3-VL-8B-Instruct'),
                ]
            ),
        ],
        TemplateType.qwen2_5_vl,
        get_model_tokenizer_videonsa_qwen3,
        model_arch=ModelArch.videonsa,
        architectures=['Qwen3VLForConditionalGeneration'],
        requires=['transformers>=4.55', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video'],
    )
)
