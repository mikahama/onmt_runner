# onmt-runner

`onmt-runner` provides inference for legacy OpenNMT models on modern Python,
without installing OpenNMT-py itself. It supports:

- OpenNMT-py 1.x-style recurrent text checkpoints (`.pt`);
- OpenNMT-tf SavedModel directories;
- greedy or beam-search translation, n-best hypotheses, and prediction scores.

## Installation

For legacy OpenNMT-py checkpoints:

```bash
python -m pip install -r requirements.txt
```

For OpenNMT-tf SavedModels, install the optional TensorFlow extra as well:

```bash
python -m pip install 'onmt-runner[tensorflow]'
# Or, from a source checkout:
python -m pip install -r requirements-tensorflow.txt
```

## Translating

Load either format with the same entry point. Inputs must already be tokenized
as they were during training.

```python
from onmt_runner import Translator

translator = Translator.load("murre_norm_default.pt")
print(translator.translate("h e i"))
# h e i d ä n
```

For a character model, supply space-separated characters. Pass a detokenizer
to return ordinary text instead of tokenized text:

```python
translator.translate("h e i", detokenize="".join)
# heidän
```

Batch translation is also available:

```python
translator.translate_many(["h e i", "m i t ä"])
```

### OpenNMT-tf SavedModels

```python
translator = Translator.load("flags_dist")
print(translator.translate("Kainuu h e i"))
# h e i
```

Some older OpenNMT-tf exports use the retired TensorFlow Addons `GatherTree`
operation. `onmt-runner` compiles a small CPU compatibility operation on first
load. A C++17 compiler such as `g++` must therefore be available for these
exports.

## N-best translations and scores

`n_best=1` (the default) returns a string. When `n_best` is greater than one,
`translate` returns a ranked `list[str]`.

```python
translator.translate("h e i", beam_size=5, n_best=3)
```

Set `return_scores=True` to receive `Prediction` objects instead. Each has:

- `text`: the detokenized/tokenized translation;
- `score`: the score used to rank hypotheses;
- `raw_score`: cumulative model log-probability before normalization.

```python
prediction = translator.translate(
    "h e i",
    return_scores=True,
    length_penalty="wu",
    alpha=0.6,
)
print(prediction.text, prediction.score, prediction.raw_score)

predictions = translator.translate(
    "h e i", beam_size=5, n_best=3, return_scores=True
)
```

For OpenNMT-py models, `onmt-runner` runs beam search natively and supports
the GNMT-compatible length penalties `none`, `wu`, and `avg` (`alpha` controls
the penalty strength). For OpenNMT-tf SavedModels, `score` comes from the
exported `log_probs` output. Their beam width is fixed when exported, so
`n_best` cannot exceed the hypotheses returned by the SavedModel. The supplied
`flags_dist` export contains one hypothesis.

## Supported checkpoints

OpenNMT-py support currently covers legacy text LSTM/RNN checkpoints with
bidirectional encoders, input-feeding decoders, and dot/general global
attention. Transformer, copy-attention, coverage-attention, and non-text
checkpoints raise `UnsupportedCheckpointError`.

Legacy `.pt` files use Python pickle. Only load checkpoints you trust.

## Building a wheel

```bash
python setup.py bdist_wheel
```

The resulting wheel is written to `dist/`. A standards-based alternative is:

```bash
python -m pip install build
python -m build
```
