"""Qwen3.5 9B as DashAI GGUF text-generation models.

One DashAI component is exposed per GGUF quantization published in
``unsloth/Qwen3.5-9B-GGUF``:
Q3_K_S, Q3_K_M, IQ4_XS, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0, BF16.
Each component downloads only its own file, following the convention DashAI
uses for multi-quantization repos (see ``mixtral_model.py``).

All of them reuse DashAI's ``GGUFTextGenerationModel`` (download via
HuggingFace, loading via llama.cpp) and override ``generate`` so that the
model answers WITHOUT its reasoning block: the GGUF chat template is rendered
with ``enable_thinking=False``. Any residual ``<think>...</think>`` text is
stripped from the output as a safeguard.

Every component adds a ``gpu_layers`` parameter for partial GPU offload, so a
quantization that does not fit entirely in GPU memory can still use the GPU.
"""

import logging
import re
from typing import List, Tuple

from DashAI.back.core.schema_fields import int_field, schema_field
from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.hugging_face.gguf_text_generation_base import (
    GGUFTextGenerationModel,
    GGUFTextGenerationSchema,
)
from DashAI.back.models.utils import LLAMA_DEVICE_TO_IDX

try:  # DashAI develop: classified load errors; older DashAI: plain load
    from DashAI.back.models.hugging_face.llama_utils import load_llama_or_raise
except ImportError:  # pragma: no cover
    load_llama_or_raise = None

log = logging.getLogger(__name__)

_REPO_ID = "unsloth/Qwen3.5-9B-GGUF"
_REPO_URL = f"https://huggingface.co/{_REPO_ID}"
_MODEL_URL = "https://huggingface.co/Qwen/Qwen3.5-9B"
_FILE_PREFIX = "Qwen3.5-9B"
_MODEL_NAME = "Qwen3.5 9B"

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", flags=re.DOTALL)
_STOP = ["<|im_end|>", "<|endoftext|>"]
#: ``{% generation %}`` is a transformers-only Jinja tag (assistant-token
#: masks for training); plain Jinja rejects it, so it is turned into a no-op.
_GENERATION_TAG_RE = re.compile(r"\{%(-?)\s*(end)?generation\s*(-?)%\}")

#: Short description of the model, per language, used in every component.
_ABOUT = {
    "en": "Official Qwen3.5 9B instruct model by the Qwen team",
    "es": "Modelo oficial Qwen3.5 9B del equipo Qwen",
    "pt": "Modelo oficial Qwen3.5 9B da equipe Qwen",
    "de": "Offizielles Qwen3.5-9B-Instruct-Modell des Qwen-Teams",
    "zh": "Qwen 团队官方 Qwen3.5 9B 指令模型",
}

#: Short, per-language note describing the quality/size trade-off of a tier.
_TIER_NOTES = {
    "q3": {
        "en": "small file, with some loss of quality",
        "es": "archivo pequeño, con algo de pérdida de calidad",
        "pt": "arquivo pequeno, com alguma perda de qualidade",
        "de": "kleine Datei, mit etwas Qualitätsverlust",
        "zh": "体积较小，有一定质量损失",
    },
    "q4": {
        "en": "balance between quality and size",
        "es": "equilibrio entre calidad y tamaño",
        "pt": "equilíbrio entre qualidade e tamanho",
        "de": "Gleichgewicht zwischen Qualität und Größe",
        "zh": "质量与体积的平衡",
    },
    "q5": {
        "en": "high quality, close to the original model",
        "es": "alta calidad, cercana al modelo original",
        "pt": "alta qualidade, próxima do modelo original",
        "de": "hohe Qualität, nahe am Originalmodell",
        "zh": "高质量，接近原始模型",
    },
    "q6": {
        "en": "very high quality, nearly lossless",
        "es": "calidad muy alta, casi sin pérdida",
        "pt": "qualidade muito alta, quase sem perda",
        "de": "sehr hohe Qualität, nahezu verlustfrei",
        "zh": "质量很高，几乎无损",
    },
    "q8": {
        "en": "8-bit, practically indistinguishable from the original model",
        "es": "8 bits, prácticamente indistinguible del modelo original",
        "pt": "8 bits, praticamente indistinguível do modelo original",
        "de": "8 Bit, praktisch nicht vom Originalmodell zu unterscheiden",
        "zh": "8 位量化，与原始模型几乎无差别",
    },
    "bf16": {
        "en": "unquantized 16-bit weights, maximum quality and highest memory use",
        "es": "pesos de 16 bits sin cuantizar, máxima calidad y mayor uso de memoria",
        "pt": "pesos de 16 bits sem quantização, qualidade máxima e maior uso de "
        "memória",
        "de": "nicht quantisierte 16-Bit-Gewichte, maximale Qualität und höchster "
        "Speicherbedarf",
        "zh": "未量化的 16 位权重，质量最高，内存占用最大",
    },
}


def _gguf_file(quant: str) -> str:
    """Return the exact GGUF file name of a quantization in the repo."""
    return f"{_FILE_PREFIX}-{quant}.gguf"


def _texts(
    quant: str, size_bytes: int, tier: str
) -> Tuple[MultilingualString, MultilingualString]:
    """Build the display name and description of one quantization.

    Parameters
    ----------
    quant : str
        Quantization label as it appears in the file name (e.g. ``"Q8_0"``).
    size_bytes : int
        Size of the GGUF file in bytes.
    tier : str
        Key of ``_TIER_NOTES`` describing the quality/size trade-off.

    Returns
    -------
    tuple of MultilingualString
        ``(DISPLAY_NAME, DESCRIPTION)``.
    """
    gb_en = f"{size_bytes / 1e9:.1f}"
    gb_comma = gb_en.replace(".", ",")
    note = _TIER_NOTES[tier]
    label = quant.upper()
    name = f"{_MODEL_NAME} ({label})"
    display_name = MultilingualString(en=name, es=name, pt=name, de=name, zh=name)
    description = MultilingualString(
        en=(
            f"{_ABOUT['en']}, loaded as a {label} GGUF (~{gb_en} GB; "
            f"{note['en']}) for inference with llama.cpp. Thinking mode is "
            f"disabled so answers are direct (suited for RAG). {_MODEL_URL}"
        ),
        es=(
            f"{_ABOUT['es']}, cargado como GGUF {label} (~{gb_comma} GB; "
            f"{note['es']}) para inferencia con llama.cpp. El modo de "
            "razonamiento está desactivado para que las respuestas sean "
            "directas (adecuado para RAG)."
        ),
        pt=(
            f"{_ABOUT['pt']}, carregado como GGUF {label} (~{gb_comma} GB; "
            f"{note['pt']}) para inferência com llama.cpp. O modo de "
            "raciocínio está desativado (adequado para RAG)."
        ),
        de=(
            f"{_ABOUT['de']}, als {label}-GGUF (~{gb_comma} GB; {note['de']}) "
            "für die Inferenz mit llama.cpp geladen. Der Denkmodus ist "
            "deaktiviert (geeignet für RAG)."
        ),
        zh=(
            f"{_ABOUT['zh']}，以 {label} GGUF（约 {gb_en} GB；{note['zh']}）"
            "格式通过 llama.cpp 推理。已关闭思考模式，适用于 RAG。"
        ),
    )
    return display_name, description


def _strip_think(text: str) -> str:
    """Remove any ``<think>...</think>`` block (or an unterminated one)."""
    return _THINK_RE.sub("", text).strip()


def _chatml_fallback(messages: list) -> str:
    """Plain ChatML rendering with thinking disabled (used if Jinja fails)."""
    parts = [
        f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n"
        for m in messages
    ]
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def _plain_jinja(template: str) -> str:
    """Make a HuggingFace chat template renderable by plain Jinja."""
    return _GENERATION_TAG_RE.sub(
        lambda m: (
            "{%"
            + m.group(1)
            + (" endif " if m.group(2) else " if true ")
            + m.group(3)
            + "%}"
        ),
        template,
    )


class Qwen35_9BSchema(GGUFTextGenerationSchema):
    """GGUF schema plus ``gpu_layers`` for partial GPU offload."""

    gpu_layers: schema_field(
        int_field(ge=-1, le=200),
        placeholder=-1,
        description=MultilingualString(
            en=(
                "Number of transformer layers offloaded to the GPU when a GPU "
                "device is selected. -1 offloads all layers. Use a smaller value "
                "(e.g. 30 of 33) when the model does not fit in GPU memory; the "
                "remaining layers run on CPU. Ignored when device is CPU."
            ),
            es=(
                "Número de capas que se cargan en la GPU cuando se elige un "
                "dispositivo GPU. -1 carga todas. Usa un valor menor (p. ej. 30 de "
                "33) si el modelo no cabe en la memoria de la GPU; el resto corre "
                "en CPU. Se ignora si el dispositivo es CPU."
            ),
            pt=(
                "Número de camadas carregadas na GPU quando um dispositivo GPU é "
                "selecionado. -1 carrega todas. Ignorado se o dispositivo for CPU."
            ),
            de=(
                "Anzahl der auf die GPU ausgelagerten Schichten, wenn ein "
                "GPU-Gerät gewählt ist. -1 lagert alle aus. Bei CPU ignoriert."
            ),
            zh="选择 GPU 设备时卸载到 GPU 的层数。-1 表示全部。设备为 CPU 时忽略。",
        ),
        alias=MultilingualString(
            en="GPU layers",
            es="Capas en GPU",
            pt="Camadas na GPU",
            de="GPU-Schichten",
            zh="GPU 层数",
        ),
    ) = -1  # type: ignore  # default keeps sessions created without it valid


class _Qwen35_9BCheckpoint(GGUFTextGenerationModel):
    """Shared behaviour of every Qwen3.5 9B quantization.

    Not registered as a component: only its concrete subclasses are exposed
    through the ``dashai.plugins`` entry points.

    Its name must NOT contain "Base": DashAI's ``ComponentRegistry`` treats
    every ancestor whose name contains "Base" and that has a ``TYPE`` as a
    DashAI base class, and rejects components with more than one of them.
    """

    REPO_ID = _REPO_ID
    SCHEMA = Qwen35_9BSchema
    COLOR: str = "#615ced"

    def __init__(self, **kwargs):
        """Load the GGUF like the base class, honouring ``gpu_layers``."""
        from llama_cpp import Llama

        kwargs = self.validate_and_transform(kwargs)
        self.max_tokens = kwargs.pop("max_tokens", 100)
        self.temperature = kwargs.pop("temperature", 0.7)
        self.frequency_penalty = kwargs.pop("frequency_penalty", 0.1)
        self.n_ctx = kwargs.pop("context_window", 512)
        gpu_layers = kwargs.pop("gpu_layers", -1)
        device_val = kwargs.get("device")
        use_gpu = LLAMA_DEVICE_TO_IDX.get(device_val, -1) >= 0
        main_gpu = LLAMA_DEVICE_TO_IDX.get(device_val, 0) if use_gpu else 0
        n_gpu_layers = (-1 if gpu_layers is None else gpu_layers) if use_gpu else 0
        model_path = str(self._local_gguf_path())

        def loader():
            return Llama(
                model_path=model_path,
                verbose=True,
                n_ctx=self.n_ctx,
                n_gpu_layers=n_gpu_layers,
                main_gpu=main_gpu,
            )

        if load_llama_or_raise is None:
            self.model = loader()
        else:
            self.model = load_llama_or_raise(
                loader,
                model_label=type(self).__name__,
                model_path=model_path,
                device=device_val,
                context_window=self.n_ctx,
            )
        log.info(
            "Loaded %s with n_gpu_layers=%s on %s",
            type(self).__name__,
            n_gpu_layers,
            device_val,
        )

    def _render_prompt(self, messages: list) -> str:
        """Render the GGUF chat template with ``enable_thinking=False``."""
        template = (getattr(self.model, "metadata", None) or {}).get(
            "tokenizer.chat_template"
        )
        if not template:
            return _chatml_fallback(messages)
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            def _raise(msg):
                raise ValueError(msg)

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = _raise
            return env.from_string(_plain_jinja(template)).render(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=False,
                bos_token="",
                eos_token="<|im_end|>",
            )
        except Exception:  # noqa: BLE001
            log.exception("Chat template rendering failed; using ChatML fallback")
            return _chatml_fallback(messages)

    def generate(self, prompt: list) -> List[str]:
        """Generate a reply (no reasoning block) for an OpenAI-style chat prompt."""
        text = self._render_prompt(prompt)
        output = self.model.create_completion(
            prompt=text,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            frequency_penalty=self.frequency_penalty,
            stop=_STOP,
        )
        return [_strip_think(output["choices"][0]["text"])]


# --------------------------------------------------------------------------- #
# Concrete checkpoints, smallest to largest. Sizes are the exact byte sizes
# reported by the HuggingFace API for each file.
# --------------------------------------------------------------------------- #


class Qwen35_9B_Q3_K_S(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (Q3_K_S quantization)."""

    GGUF_PATTERN = _gguf_file("Q3_K_S")
    DOWNLOAD_SIZE_BYTES = 4316865760
    DISPLAY_NAME, DESCRIPTION = _texts("Q3_K_S", DOWNLOAD_SIZE_BYTES, "q3")


class Qwen35_9B_Q3_K_M(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (Q3_K_M quantization)."""

    GGUF_PATTERN = _gguf_file("Q3_K_M")
    DOWNLOAD_SIZE_BYTES = 4673643744
    DISPLAY_NAME, DESCRIPTION = _texts("Q3_K_M", DOWNLOAD_SIZE_BYTES, "q3")


class Qwen35_9B_IQ4_XS(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (IQ4_XS quantization)."""

    GGUF_PATTERN = _gguf_file("IQ4_XS")
    DOWNLOAD_SIZE_BYTES = 5168653536
    DISPLAY_NAME, DESCRIPTION = _texts("IQ4_XS", DOWNLOAD_SIZE_BYTES, "q4")


class Qwen35_9B_Q4_K_S(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (Q4_K_S quantization)."""

    GGUF_PATTERN = _gguf_file("Q4_K_S")
    DOWNLOAD_SIZE_BYTES = 5394097376
    DISPLAY_NAME, DESCRIPTION = _texts("Q4_K_S", DOWNLOAD_SIZE_BYTES, "q4")


class Qwen35_9B_Q4_K_M(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (Q4_K_M quantization)."""

    GGUF_PATTERN = _gguf_file("Q4_K_M")
    DOWNLOAD_SIZE_BYTES = 5680522464
    DISPLAY_NAME, DESCRIPTION = _texts("Q4_K_M", DOWNLOAD_SIZE_BYTES, "q4")


class Qwen35_9B_Q5_K_S(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (Q5_K_S quantization)."""

    GGUF_PATTERN = _gguf_file("Q5_K_S")
    DOWNLOAD_SIZE_BYTES = 6361146592
    DISPLAY_NAME, DESCRIPTION = _texts("Q5_K_S", DOWNLOAD_SIZE_BYTES, "q5")


class Qwen35_9B_Q5_K_M(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (Q5_K_M quantization)."""

    GGUF_PATTERN = _gguf_file("Q5_K_M")
    DOWNLOAD_SIZE_BYTES = 6577841376
    DISPLAY_NAME, DESCRIPTION = _texts("Q5_K_M", DOWNLOAD_SIZE_BYTES, "q5")


class Qwen35_9B_Q6_K(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (Q6_K quantization)."""

    GGUF_PATTERN = _gguf_file("Q6_K")
    DOWNLOAD_SIZE_BYTES = 7458301152
    DISPLAY_NAME, DESCRIPTION = _texts("Q6_K", DOWNLOAD_SIZE_BYTES, "q6")


class Qwen35_9B_Q8_0(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (Q8_0 quantization)."""

    GGUF_PATTERN = _gguf_file("Q8_0")
    DOWNLOAD_SIZE_BYTES = 9527502048
    DISPLAY_NAME, DESCRIPTION = _texts("Q8_0", DOWNLOAD_SIZE_BYTES, "q8")


class Qwen35_9B_BF16(_Qwen35_9BCheckpoint):  # noqa: N801
    """Qwen3.5 9B GGUF checkpoint (BF16, unquantized)."""

    GGUF_PATTERN = _gguf_file("BF16")
    DOWNLOAD_SIZE_BYTES = 17920697312
    DISPLAY_NAME, DESCRIPTION = _texts("BF16", DOWNLOAD_SIZE_BYTES, "bf16")
