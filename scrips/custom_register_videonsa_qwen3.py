"""Register VideoNSA Qwen3 model_type for environments where swift lacks built-in mapping."""

from swift.llm import TemplateType
from swift.llm.model.constant import MLLMModelType
from swift.llm.model.model_arch import ModelArch
from swift.llm.model.register import Model, ModelGroup, ModelMeta, register_model

try:
    from swift.llm.model.model.videonsa import get_model_tokenizer_videonsa_qwen3
except Exception as exc:
    raise RuntimeError('Failed to import get_model_tokenizer_videonsa_qwen3 from swift.') from exc


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
