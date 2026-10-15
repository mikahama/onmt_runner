# onmt-runner

`onmt-runner` loads legacy (OpenNMT-py 1.x-style) recurrent `.pt` checkpoints
without installing OpenNMT-py or its obsolete torchtext dependency.

```python
from onmt_runner import Translator

translator = Translator.load("murre_norm_default.pt")
print(translator.translate("h e i"))
print(translator.translate_many(["h e i", "m i t ä"]))
```

OpenNMT-tf SavedModel directories use the same loading entry point after
installing the optional TensorFlow dependency:

```bash
python -m pip install 'onmt-runner[tensorflow]'
```

```python
from onmt_runner import Translator

translator = Translator.load("flags_dist")
print(translator.translate("Kainuu h e i"))
```

Some older OpenNMT-tf exports use the retired TensorFlow Addons `GatherTree`
operation. `onmt_runner` transparently compiles a small CPU compatibility op
on first use; a C++17 compiler such as `g++` must be available for those
exports.

Input must already be tokenized exactly as it was during training. For a
character model, that normally means characters separated by spaces.  Returned
tokens are joined with spaces by default; pass `detokenize=lambda tokens:
"".join(tokens)` to `translate` or `translate_many` for character output.

## Supported checkpoints

The initial release supports the common OpenNMT-py 1.x text checkpoint format
whose `opt.decoder_type` is `rnn`, including bidirectional LSTM encoders,
input-feeding decoders, and dot/general global attention.  It deliberately
does not execute serialized OpenNMT code: only tensor weights and vocabulary
metadata are read. As with `torch.load(..., weights_only=False)`, only load
checkpoints you trust: historic OpenNMT checkpoints use Python pickle.

Transformer, copy-attention, and non-text checkpoints raise a precise
`UnsupportedCheckpointError` rather than producing unreliable output.

## Building a wheel

```bash
python -m pip install -r requirements.txt
python setup.py bdist_wheel
```

The resulting wheel is written to `dist/`. A standards-based alternative is
`python -m pip install build && python -m build`.
