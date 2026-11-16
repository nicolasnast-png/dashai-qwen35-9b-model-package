"""Tests for the Qwen3.5 9B plugin (all GGUF quantizations).

Run from an environment where DashAI and this plugin are installed::

    pytest tests/
"""

import pathlib
from importlib.metadata import entry_points

import pytest
from DashAI.back.dependencies.downloads.downloadable import HFDownloadableMixin
from DashAI.back.dependencies.registry import ComponentRegistry
from DashAI.back.initial_components import get_initial_components
from dashai_qwen35_9b import model as plugin
from kink import di

#: Every GGUF offered by the plugin, with its exact size in bytes.
EXPECTED_FILES = {
    "Qwen3.5-9B-Q3_K_S.gguf": 4316865760,
    "Qwen3.5-9B-Q3_K_M.gguf": 4673643744,
    "Qwen3.5-9B-IQ4_XS.gguf": 5168653536,
    "Qwen3.5-9B-Q4_K_S.gguf": 5394097376,
    "Qwen3.5-9B-Q4_K_M.gguf": 5680522464,
    "Qwen3.5-9B-Q5_K_S.gguf": 6361146592,
    "Qwen3.5-9B-Q5_K_M.gguf": 6577841376,
    "Qwen3.5-9B-Q6_K.gguf": 7458301152,
    "Qwen3.5-9B-Q8_0.gguf": 9527502048,
    "Qwen3.5-9B-BF16.gguf": 17920697312,
}
#: The model's own chat template (from the original HuggingFace repo).
REAL_TEMPLATE = (pathlib.Path(__file__).parent / "chat_template.jinja").read_text(
    encoding="utf-8"
)
REAL_TEMPLATE_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _plugin_entry_points():
    return [
        ep
        for ep in entry_points(group="dashai.plugins")
        if ep.value.startswith("dashai_qwen35_9b.")
    ]


ENTRY_POINTS = _plugin_entry_points()
CLASSES = [ep.load() for ep in ENTRY_POINTS]


@pytest.fixture
def component_root(tmp_path):
    sentinel = object()
    old = di["config"] if "config" in di else sentinel  # noqa: SIM401
    di["config"] = {"COMPONENT_PATH": str(tmp_path)}
    yield pathlib.Path(tmp_path)
    if old is sentinel:
        del di["config"]
    else:
        di["config"] = old


def test_one_entry_point_per_gguf_file():
    assert len(CLASSES) == len(EXPECTED_FILES) == 10
    assert len(set(CLASSES)) == len(CLASSES)
    for ep, cls in zip(ENTRY_POINTS, CLASSES, strict=True):
        assert ep.name == cls.__name__
    assert {cls.GGUF_PATTERN for cls in CLASSES} == set(EXPECTED_FILES)


def test_eight_bit_quantization_is_included():
    assert plugin.Qwen35_9B_Q8_0 in CLASSES
    assert "8 bits" in str(plugin.Qwen35_9B_Q8_0.DESCRIPTION.es)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_checkpoint_metadata(cls):
    assert issubclass(cls, plugin._Qwen35_9BCheckpoint)
    assert issubclass(cls, HFDownloadableMixin)
    assert cls.REPO_ID == "unsloth/Qwen3.5-9B-GGUF"
    assert EXPECTED_FILES[cls.GGUF_PATTERN] == cls.DOWNLOAD_SIZE_BYTES
    assert cls.hf_repos() == [(cls.REPO_ID, "model", [cls.GGUF_PATTERN])]
    meta = cls.get_metadata()
    assert meta["requires_download"] is True
    assert meta["download_size_bytes"] == cls.DOWNLOAD_SIZE_BYTES
    quant = cls.GGUF_PATTERN.rsplit("-", 1)[1].removesuffix(".gguf").upper()
    assert cls.__name__.endswith(quant)
    for lang in ("en", "es", "pt", "de", "zh"):
        assert quant in str(getattr(cls.DISPLAY_NAME, lang))
        assert quant in str(getattr(cls.DESCRIPTION, lang))


def test_all_components_register_alongside_dashai_builtins():
    registry = ComponentRegistry(initial_components=get_initial_components() + CLASSES)
    names = set(registry.registry["GenerativeModel"])
    for cls in CLASSES:
        assert cls.__name__ in names
        assert registry[cls.__name__]["class"] is cls
    assert "_Qwen35_9BCheckpoint" not in names
    display_names = {str(registry[c.__name__]["display_name"]) for c in CLASSES}
    assert len(display_names) == len(CLASSES)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_each_component_resolves_only_its_own_file(component_root, cls):
    assert cls.is_downloaded() is False
    repo_dir = component_root / cls.__name__ / cls.REPO_ID.split("/")[-1]
    repo_dir.mkdir(parents=True)
    # Even if every quantization sits in the same folder, the exact file name
    # must be the one picked up (e.g. Q4_K_S must not match Q4_K_M).
    for name in EXPECTED_FILES:
        (repo_dir / name).write_text("weights")
    assert cls.is_downloaded() is True
    assert cls._local_gguf_path().name == cls.GGUF_PATTERN


def test_partial_download_is_not_downloaded(component_root):
    cls = plugin.Qwen35_9B_Q8_0
    repo_dir = component_root / cls.__name__ / cls.REPO_ID.split("/")[-1]
    repo_dir.mkdir(parents=True)
    (repo_dir / (cls.GGUF_PATTERN + ".incomplete")).write_text("partial")
    assert cls.is_downloaded() is False


class _FakeLlama:
    def __init__(self, template, reply):
        self.metadata = {"tokenizer.chat_template": template} if template else {}
        self.reply = reply
        self.calls = []

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"text": self.reply}]}


def _instance(cls, fake):
    obj = object.__new__(cls)
    obj.model = fake
    obj.max_tokens = 50
    obj.temperature = 0.2
    obj.frequency_penalty = 0.1
    return obj


@pytest.mark.parametrize(
    ("template", "suffix"),
    [
        (None, "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        (REAL_TEMPLATE, REAL_TEMPLATE_SUFFIX),
    ],
    ids=["chatml-fallback", "model-template"],
)
@pytest.mark.parametrize(
    "cls",
    [plugin.Qwen35_9B_Q8_0, plugin.Qwen35_9B_Q4_K_M],
    ids=lambda c: c.__name__,
)
def test_generate_disables_and_strips_thinking(cls, template, suffix):
    fake = _FakeLlama(template, "<think>razonando...</think>\n\nRespuesta final.")
    messages = [
        {"role": "system", "content": "Responde solo con el contexto."},
        {"role": "user", "content": "Hola"},
    ]
    out = _instance(cls, fake).generate(messages)
    assert out == ["Respuesta final."]
    call = fake.calls[0]
    assert call["prompt"].endswith(suffix)
    assert "Responde solo con el contexto." in call["prompt"]
    assert call["stop"] == ["<|im_end|>", "<|endoftext|>"]
    assert call["max_tokens"] == 50


def test_model_template_renders_without_fallback(caplog):
    fake = _FakeLlama(REAL_TEMPLATE, "ok")
    cls = plugin.Qwen35_9B_Q4_K_M
    history = [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Hola, ¿en qué ayudo?"},
        {"role": "user", "content": "Resume esto."},
    ]
    with caplog.at_level("ERROR"):
        _instance(cls, fake).generate(history)
    assert "ChatML fallback" not in caplog.text


def test_unterminated_think_block_is_stripped():
    fake = _FakeLlama(None, "<think>se cortó por max_tokens")
    cls = plugin.Qwen35_9B_Q6_K
    assert _instance(cls, fake).generate([{"role": "user", "content": "x"}]) == [""]


# --------------------------------------------------------------------------- #
# gpu_layers: partial GPU offload
# --------------------------------------------------------------------------- #


def test_schema_exposes_gpu_layers_with_backward_compatible_default():
    fields = plugin.Qwen35_9BSchema.model_fields
    assert "gpu_layers" in fields
    assert fields["gpu_layers"].default == -1
    for cls in CLASSES:
        assert cls.SCHEMA is plugin.Qwen35_9BSchema


def _load_with(monkeypatch, device, **params):
    import llama_cpp

    captured = {}

    class _RecordingLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    gpu_name = "GPU 0: Fake GPU - Compute Capability 8.6"
    monkeypatch.setitem(plugin.LLAMA_DEVICE_TO_IDX, gpu_name, 0)
    monkeypatch.setattr(llama_cpp, "Llama", _RecordingLlama)
    monkeypatch.setattr(plugin, "load_llama_or_raise", None)
    cls = plugin.Qwen35_9B_IQ4_XS
    monkeypatch.setattr(cls, "validate_and_transform", lambda self, kw: dict(kw))
    monkeypatch.setattr(cls, "_local_gguf_path", lambda self: pathlib.Path("m.gguf"))
    dev = gpu_name if device == "GPU" else "CPU"
    cls(device=dev, context_window=1024, **params)
    return captured


def test_gpu_layers_is_passed_to_llama_on_gpu(monkeypatch):
    assert _load_with(monkeypatch, "GPU", gpu_layers=30)["n_gpu_layers"] == 30


def test_gpu_layers_default_offloads_everything_on_gpu(monkeypatch):
    assert _load_with(monkeypatch, "GPU")["n_gpu_layers"] == -1


def test_gpu_layers_is_ignored_on_cpu(monkeypatch):
    assert _load_with(monkeypatch, "CPU", gpu_layers=30)["n_gpu_layers"] == 0
