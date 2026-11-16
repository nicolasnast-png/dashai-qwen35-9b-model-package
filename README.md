# dashai-qwen35-9b-model-package

A [DashAI](https://github.com/DashAISoftware/DashAI) plugin that adds **Qwen3.5 9B** as a text generation model: the official Qwen3.5 9B instruct model from the Qwen team (Alibaba), in the GGUF quantizations published by Unsloth. It runs locally with llama.cpp and works in DashAI's generative chat and **RAG** pipelines.

Qwen reports for this model AA-LCR 63.0, LongBench v2 55.2 and MMLongBench-Doc 57.7 (long-context and document understanding), and IFEval 91.5.

This is a community plugin. It is not affiliated with the original model authors.

## What the plugin adds

- **One component per quantization.** Each component is a `GGUFTextGenerationModel` that downloads only its own GGUF file, like the built-in GGUF models.

| Component | Quantization | Download |
| --- | --- | --- |
| `Qwen35_9B_Q3_K_S` | Q3_K_S | 4.32 GB |
| `Qwen35_9B_Q3_K_M` | Q3_K_M | 4.67 GB |
| `Qwen35_9B_IQ4_XS` | IQ4_XS | 5.17 GB |
| `Qwen35_9B_Q4_K_S` | Q4_K_S | 5.39 GB |
| `Qwen35_9B_Q4_K_M` | Q4_K_M | 5.68 GB |
| `Qwen35_9B_Q5_K_S` | Q5_K_S | 6.36 GB |
| `Qwen35_9B_Q5_K_M` | Q5_K_M | 6.58 GB |
| `Qwen35_9B_Q6_K` | Q6_K | 7.46 GB |
| `Qwen35_9B_Q8_0` | Q8_0 (8-bit) | 9.53 GB |
| `Qwen35_9B_BF16` | BF16 (unquantized) | 17.92 GB |

- **Standard GGUF parameters:** `max_tokens`, `temperature`, `frequency_penalty`, `context_window` and `device`. The model supports up to 262,144 tokens of context.
- **Partial GPU offload (`gpu_layers`):** when `device` is a GPU, `gpu_layers` sets how many layers go to the GPU. The default, `-1`, offloads all of them; the rest run on CPU. Use it when the model does not fit in GPU memory: on a 6 GB GPU, a ~5 GB quantization fits with about 30 layers. The setting is ignored when `device` is `CPU`. GPU use requires a CUDA build of `llama-cpp-python`.
- **Thinking mode disabled:** this is a reasoning model, and by default it writes a `<think>…</think>` block before answering. The plugin renders the model's own chat template with `enable_thinking=False`, which makes the model answer directly. It also strips any leftover `<think>` block.

## Requirements

- **DashAI:** a version that provides `GGUFTextGenerationModel` (0.11.0 or later; tested on the `develop` branch).
- **llama.cpp:** must support the `qwen35` architecture. `llama-cpp-python` 0.3.34, the version shipped with DashAI `develop`, supports it.
- **Memory:** disk space equal to the download size of the chosen quantization. For CPU inference, free RAM of at least that size plus about 2 GB, more for long contexts.

## Install

**From DashAI:** open the plugin store, search for `dashai-qwen35-9b-model-package` and install it.

**Manually:** install into DashAI's plugin directory, then restart DashAI:

```bash
pip install --no-deps --target ~/.DashAI/plugins/py3.12 dashai-qwen35-9b-model-package
```

Then download the weights of the quantization you want from its model card in DashAI, or with the API:

```bash
curl -X POST http://localhost:8000/api/v1/component/Qwen35_9B_IQ4_XS/download
```

## Use in a RAG session

Set the generation model of a RAG session to one of the components above. Example API payload:

```json
{"generation_model": {"component": "Qwen35_9B_IQ4_XS",
  "params": {"max_tokens": 600, "temperature": 0.0, "frequency_penalty": 0.1,
             "context_window": 10000, "device": "CPU"}}}
```

## Español

Plugin comunitario para DashAI que agrega **Qwen3.5 9B**: el modelo oficial Qwen3.5 9B del equipo Qwen (Alibaba), en las cuantizaciones GGUF publicadas por Unsloth.

- **Cuantizaciones:** cada cuantización aparece como un modelo aparte en DashAI y descarga solo su archivo.
- **Sin razonamiento:** el modo de razonamiento está desactivado, así que el modelo responde directo. Está pensado para RAG.
- **Uso parcial de la GPU:** el parámetro `gpu_layers` carga solo parte de las capas en la GPU cuando el modelo no cabe completo.

Qwen reporta para este modelo AA-LCR 63,0, LongBench v2 55,2 y MMLongBench-Doc 57,7 (contexto largo y comprensión de documentos), e IFEval 91,5.

1. Instálalo desde la tienda de plugins de DashAI, o con `pip install --no-deps --target ~/.DashAI/plugins/py3.12 dashai-qwen35-9b-model-package`.
2. Reinicia DashAI.
3. Descarga desde DashAI la cuantización que quieras, de 4,3 GB (Q3_K_S) a 17,9 GB (BF16).

## License

- **Plugin code:** Apache-2.0 (see [LICENSE](LICENSE)).
- **Model weights:** Apache-2.0, under the terms of the [original model card](https://huggingface.co/Qwen/Qwen3.5-9B).
- **Licencia de los pesos:** Apache-2.0, según la [ficha original del modelo](https://huggingface.co/Qwen/Qwen3.5-9B).
