"""Native PyTorch inference for OpenNMT-py 1.x recurrent checkpoints."""

from __future__ import annotations

import sys
import types
import hashlib
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class UnsupportedCheckpointError(ValueError):
    """The checkpoint is valid, but uses an architecture not yet supported."""


@dataclass(frozen=True)
class Prediction:
    """A translated hypothesis with normalized and raw log-probability."""
    text: str
    score: float
    raw_score: float


class _PicklePlaceholder:
    """Receives state for obsolete OpenNMT/torchtext metadata classes."""

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


def _legacy_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]

    def get_attribute(attribute: str) -> Any:
        # Pickle needs a real callable for this one metadata-only helper.
        if attribute == "_split_tokenizer":
            return str.split
        placeholder = type(attribute, (_PicklePlaceholder,), {"__module__": name})
        setattr(module, attribute, placeholder)
        return placeholder

    module.__getattr__ = get_attribute  # type: ignore[attr-defined]
    return module


@contextmanager
def _obsolete_class_shims() -> Any:
    """Temporarily provide the names referenced by an old checkpoint pickle."""
    names = (
        "onmt",
        "onmt.inputters",
        "onmt.inputters.text_dataset",
        "torchtext",
        "torchtext.vocab",
        "torchtext.data",
        "torchtext.data.field",
        "torchtext.data.utils",
    )
    old = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            if name not in sys.modules:
                sys.modules[name] = _legacy_module(name)
        yield
    finally:
        for name, prior in old.items():
            if prior is None:
                sys.modules.pop(name, None)


def _vocabulary(checkpoint: dict[str, Any], side: str) -> tuple[list[str], dict[str, int], str]:
    try:
        field = checkpoint["vocab"][side].fields[0][1]
        vocab = field.vocab
        tokens = list(vocab.itos)
        lookup = dict(vocab.stoi)
        pad = field.pad_token
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise UnsupportedCheckpointError("checkpoint does not contain a legacy text vocabulary") from error
    return tokens, lookup, pad


@dataclass
class _DecoderState:
    hidden: list[tuple[Tensor, Tensor]]
    input_feed: Tensor


class _RecurrentModel(nn.Module):
    def __init__(self, checkpoint: dict[str, Any], src_pad: int) -> None:
        super().__init__()
        state = checkpoint["model"]
        opt = checkpoint["opt"]
        if getattr(opt, "decoder_type", None) != "rnn" or getattr(opt, "model_type", "text") != "text":
            raise UnsupportedCheckpointError("only legacy text RNN checkpoints are supported")
        if getattr(opt, "copy_attn", False) or getattr(opt, "coverage_attn", False):
            raise UnsupportedCheckpointError("copy/coverage attention checkpoints are not supported")
        if getattr(opt, "rnn_type", "LSTM") != "LSTM":
            raise UnsupportedCheckpointError("only LSTM checkpoints are currently supported")
        if "decoder.attn.linear_in.weight" not in state:
            raise UnsupportedCheckpointError("checkpoint has an unsupported decoder attention module")

        src_embedding = state["encoder.embeddings.make_embedding.emb_luts.0.weight"]
        tgt_embedding = state["decoder.embeddings.make_embedding.emb_luts.0.weight"]
        self.src_embeddings = nn.Embedding.from_pretrained(src_embedding, freeze=True, padding_idx=src_pad)
        self.tgt_embeddings = nn.Embedding.from_pretrained(tgt_embedding, freeze=True)
        self.encoder = nn.LSTM(
            src_embedding.shape[1],
            state["encoder.rnn.weight_hh_l0"].shape[1],
            num_layers=getattr(opt, "enc_layers"),
            bidirectional=getattr(opt, "brnn", False),
        )
        self.decoder_cells = nn.ModuleList()
        layers = getattr(opt, "dec_layers")
        for layer in range(layers):
            input_size = state[f"decoder.rnn.layers.{layer}.weight_ih"].shape[1]
            hidden_size = state[f"decoder.rnn.layers.{layer}.weight_hh"].shape[1]
            self.decoder_cells.append(nn.LSTMCell(input_size, hidden_size))
        hidden_size = state["decoder.rnn.layers.0.weight_hh"].shape[1]
        self.attn_in = nn.Linear(hidden_size, state["decoder.attn.linear_in.weight"].shape[0], bias=False)
        self.attn_out = nn.Linear(state["decoder.attn.linear_out.weight"].shape[1], hidden_size, bias=False)
        generator_state = checkpoint["generator"]
        self.generator = nn.Linear(generator_state["0.weight"].shape[1], generator_state["0.weight"].shape[0])

        remapped: dict[str, Tensor] = {}
        for key, value in state.items():
            if key.startswith("encoder."):
                if key.startswith("encoder.rnn."):
                    remapped[key.replace("encoder.rnn", "encoder", 1)] = value
            elif key.startswith("decoder.rnn.layers."):
                remapped[key.replace("decoder.rnn.layers", "decoder_cells", 1)] = value
            elif key == "decoder.attn.linear_in.weight":
                remapped["attn_in.weight"] = value
            elif key == "decoder.attn.linear_out.weight":
                remapped["attn_out.weight"] = value
        remapped["generator.weight"] = generator_state["0.weight"]
        remapped["generator.bias"] = generator_state["0.bias"]
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        material_missing = [key for key in missing if key not in {"src_embeddings.weight", "tgt_embeddings.weight"}]
        if material_missing or unexpected:
            raise UnsupportedCheckpointError(f"could not map checkpoint weights (missing={material_missing}, unexpected={unexpected})")

    def encode(self, source: Tensor, lengths: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        embedded = self.src_embeddings(source)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), enforce_sorted=False)
        memory, hidden = self.encoder(packed)
        memory, _ = nn.utils.rnn.pad_packed_sequence(memory)
        return memory, hidden

    def initial_state(self, encoder_hidden: tuple[Tensor, Tensor]) -> _DecoderState:
        h, c = encoder_hidden
        layers = len(self.decoder_cells)
        directions = h.shape[0] // layers
        if directions == 2:
            h = torch.cat((h[0::2], h[1::2]), dim=2)
            c = torch.cat((c[0::2], c[1::2]), dim=2)
        hidden = [(h[layer], c[layer]) for layer in range(layers)]
        batch = h.shape[1]
        return _DecoderState(hidden, h.new_zeros(batch, self.generator.in_features))

    def step(self, token: Tensor, state: _DecoderState, memory: Tensor, source_mask: Tensor) -> tuple[Tensor, _DecoderState]:
        output = torch.cat((self.tgt_embeddings(token), state.input_feed), dim=1)
        next_hidden: list[tuple[Tensor, Tensor]] = []
        for cell, (h, c) in zip(self.decoder_cells, state.hidden):
            h, c = cell(output, (h, c))
            next_hidden.append((h, c))
            output = h
        # memory: source length x batch x hidden.  Attention works batch-first.
        scores = torch.bmm(memory.transpose(0, 1), self.attn_in(output).unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~source_mask, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=1)
        context = torch.bmm(attention.unsqueeze(1), memory.transpose(0, 1)).squeeze(1)
        attn_output = torch.tanh(self.attn_out(torch.cat((context, output), dim=1)))
        return F.log_softmax(self.generator(attn_output), dim=1), _DecoderState(next_hidden, attn_output)


class Translator:
    """Loads and translates with a legacy OpenNMT-py recurrent checkpoint."""

    def __init__(self, checkpoint: dict[str, Any], device: str | torch.device = "cpu") -> None:
        self.src_tokens, self.src_stoi, src_pad = _vocabulary(checkpoint, "src")
        self.tgt_tokens, self.tgt_stoi, _ = _vocabulary(checkpoint, "tgt")
        self.device = torch.device(device)
        self.model = _RecurrentModel(checkpoint, self.src_stoi[src_pad]).to(self.device).eval()
        self.unk_id = self.src_stoi.get("<unk>", 0)
        self.bos_id = self.tgt_stoi["<s>"]
        self.eos_id = self.tgt_stoi["</s>"]

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "Translator | TensorFlowTranslator":
        """Load a legacy checkpoint or an OpenNMT-tf SavedModel directory."""
        path = Path(path)
        if path.is_dir() and (path / "saved_model.pb").is_file():
            return TensorFlowTranslator.load(path)
        with _obsolete_class_shims():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not {"model", "generator", "vocab", "opt"} <= checkpoint.keys():
            raise UnsupportedCheckpointError("not a recognized legacy OpenNMT-py checkpoint")
        return cls(checkpoint, device)

    @torch.inference_mode()
    def translate_many(
        self,
        texts: Sequence[str],
        *,
        max_length: int = 100,
        detokenize: Callable[[list[str]], str] | None = None,
        beam_size: int | None = None,
        n_best: int = 1,
        return_scores: bool = False,
        length_penalty: str = "none",
        alpha: float = 0.0,
    ) -> list[str] | list[list[str]]:
        """Translate pre-tokenized strings, optionally returning ranked hypotheses."""
        beam_size = max(1, n_best) if beam_size is None else beam_size
        if n_best < 1 or beam_size < 1:
            raise ValueError("n_best and beam_size must be positive")
        if n_best > beam_size:
            raise ValueError("n_best cannot exceed beam_size")
        if length_penalty not in {"none", "wu", "avg"}:
            raise ValueError("length_penalty must be 'none', 'wu', or 'avg'")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if not texts:
            return []
        words = [text.split() for text in texts]
        if any(not sentence for sentence in words):
            raise ValueError("each source sentence must contain at least one token")
        if beam_size > 1 or return_scores:
            join = detokenize or (lambda tokens: " ".join(tokens))
            translations = [self._beam_translate(sentence, max_length, beam_size, n_best, join, return_scores, length_penalty, alpha) for sentence in words]
            return translations if n_best > 1 else [translation[0] for translation in translations]
        lengths = torch.tensor([len(sentence) for sentence in words], device=self.device)
        max_source = int(lengths.max())
        source = torch.full((max_source, len(words)), self.src_stoi["<blank>"], dtype=torch.long, device=self.device)
        for batch, sentence in enumerate(words):
            source[: len(sentence), batch] = torch.tensor([self.src_stoi.get(token, self.unk_id) for token in sentence], device=self.device)
        memory, encoder_hidden = self.model.encode(source, lengths)
        source_mask = torch.arange(max_source, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        state = self.model.initial_state(encoder_hidden)
        token = torch.full((len(words),), self.bos_id, dtype=torch.long, device=self.device)
        results: list[list[str]] = [[] for _ in words]
        finished = torch.zeros(len(words), dtype=torch.bool, device=self.device)
        for _ in range(max_length):
            log_probs, state = self.model.step(token, state, memory, source_mask)
            token = log_probs.argmax(dim=1)
            for batch, token_id in enumerate(token.tolist()):
                if not finished[batch] and token_id != self.eos_id:
                    results[batch].append(self.tgt_tokens[token_id])
                if token_id == self.eos_id:
                    finished[batch] = True
            if bool(finished.all()):
                break
        join = detokenize or (lambda tokens: " ".join(tokens))
        texts = [join(tokens) for tokens in results]
        return [Prediction(text, 0.0, 0.0) for text in texts] if return_scores else texts

    def _beam_translate(
        self,
        sentence: list[str],
        max_length: int,
        beam_size: int,
        n_best: int,
        join: Callable[[list[str]], str],
        return_scores: bool,
        length_penalty: str,
        alpha: float,
    ) -> list[Any]:
        """Run independent beam search for one sentence.

        Keeping each hypothesis' decoder state separate avoids depending on
        the old OpenNMT beam-search implementation and its removed classes.
        """
        source = torch.tensor(
            [[self.src_stoi.get(token, self.unk_id)] for token in sentence], dtype=torch.long, device=self.device
        )
        lengths = torch.tensor([len(sentence)], device=self.device)
        memory, encoder_hidden = self.model.encode(source, lengths)
        source_mask = torch.ones((1, len(sentence)), dtype=torch.bool, device=self.device)
        initial = self.model.initial_state(encoder_hidden)
        # (last input token, generated tokens, cumulative log probability, decoder state)
        active: list[tuple[int, list[int], float, _DecoderState]] = [(self.bos_id, [], 0.0, initial)]
        finished: list[tuple[list[int], float]] = []
        for _ in range(max_length):
            candidates: list[tuple[int, list[int], float, _DecoderState]] = []
            for last_token, tokens, score, state in active:
                token = torch.tensor([last_token], dtype=torch.long, device=self.device)
                log_probs, next_state = self.model.step(token, state, memory, source_mask)
                values, ids = torch.topk(log_probs[0], beam_size)
                for value, token_id in zip(values.tolist(), ids.tolist()):
                    next_tokens = tokens + [token_id]
                    next_score = score + value
                    if token_id == self.eos_id:
                        finished.append((tokens, next_score))
                    else:
                        candidates.append((token_id, next_tokens, next_score, next_state))
            candidates.sort(key=lambda candidate: self._score(candidate[2], len(candidate[1]), length_penalty, alpha), reverse=True)
            active = candidates[:beam_size]
            if not active:
                break
        ranked = finished + [(tokens, score) for _, tokens, score, _ in active]
        ranked.sort(key=lambda candidate: self._score(candidate[1], len(candidate[0]), length_penalty, alpha), reverse=True)
        if not ranked:
            ranked = [([], float("-inf"))]
        return [Prediction(join([self.tgt_tokens[token_id] for token_id in tokens]), self._score(raw, len(tokens), length_penalty, alpha), raw) if return_scores else join([self.tgt_tokens[token_id] for token_id in tokens]) for tokens, raw in ranked[:n_best]]

    @staticmethod
    def _score(raw_score: float, length: int, penalty: str, alpha: float) -> float:
        if penalty == "wu":
            return raw_score / (((5.0 + length) / 6.0) ** alpha)
        if penalty == "avg":
            return raw_score / (max(length, 1) ** alpha)
        return raw_score

    def translate(self, text: str, **kwargs: Any) -> str | list[str]:
        """Translate one pre-tokenized sentence; return a list when ``n_best > 1``."""
        return self.translate_many([text], **kwargs)[0]


class TensorFlowTranslator:
    """Inference wrapper for an OpenNMT-tf SavedModel export.

    TensorFlow is deliberately imported only while loading this backend, so a
    PyTorch-only installation does not need the large TensorFlow dependency.
    """

    def __init__(self, signature: Any) -> None:
        self._signature = signature
        _, keyword_inputs = signature.structured_input_signature
        missing = {"tokens", "length"} - set(keyword_inputs)
        if missing:
            raise UnsupportedCheckpointError(
                "SavedModel does not expose the OpenNMT-tf `tokens` and `length` inputs "
                f"(missing {sorted(missing)})"
            )
        self._input_specs = keyword_inputs

    @classmethod
    def load(cls, path: str | Path) -> "TensorFlowTranslator":
        """Load an OpenNMT-tf SavedModel directory.

        Install the optional dependency first: ``pip install -r
        requirements-tensorflow.txt``.
        """
        try:
            import tensorflow as tf
        except ImportError as error:
            raise ImportError(
                "TensorFlow is required for OpenNMT-tf SavedModels. "
                "Install it with `pip install -r requirements-tensorflow.txt`."
            ) from error
        _ensure_gather_tree_op(tf)
        model = tf.saved_model.load(str(path))
        try:
            signature = model.signatures["serving_default"]
        except KeyError as error:
            available = ", ".join(model.signatures) or "none"
            raise UnsupportedCheckpointError(
                f"SavedModel has no serving_default signature (available: {available})"
            ) from error
        return cls(signature)

    def _tensorflow(self) -> Any:
        try:
            import tensorflow as tf
        except ImportError as error:  # pragma: no cover - load() gives the useful error.
            raise ImportError("TensorFlow is required for OpenNMT-tf SavedModels") from error
        return tf

    def translate_many(
        self,
        texts: Sequence[str],
        *,
        detokenize: Callable[[list[str]], str] | None = None,
        n_best: int = 1,
        return_scores: bool = False,
        **_: Any,
    ) -> list[str] | list[list[str]]:
        """Translate pre-tokenized strings using the exported serving graph.

        SavedModel exports choose their beam width at export time. ``n_best``
        selects from those returned, already-ranked hypotheses.
        """
        if n_best < 1:
            raise ValueError("n_best must be positive")
        if not texts:
            return []
        tokenized = [text.split() for text in texts]
        if any(not sentence for sentence in tokenized):
            raise ValueError("each source sentence must contain at least one token")
        tf = self._tensorflow()
        width = max(map(len, tokenized))
        padded = [sentence + [""] * (width - len(sentence)) for sentence in tokenized]
        inputs: dict[str, Any] = {
            "tokens": tf.constant(padded, dtype=tf.string),
            "length": tf.constant([len(sentence) for sentence in tokenized], dtype=self._input_specs["length"].dtype),
        }
        # Future OpenNMT-tf exports can add optional scalar inputs. Required
        # token/length inputs are intentionally all this public API supplies.
        outputs = self._signature(**inputs)
        if "tokens" not in outputs:
            raise UnsupportedCheckpointError("OpenNMT-tf serving signature did not return `tokens`")
        values = outputs["tokens"].numpy()
        model_scores = outputs.get("log_probs")
        score_values = model_scores.numpy().tolist() if model_scores is not None else None
        lengths = outputs.get("length")
        if lengths is None:
            output_lengths = [len(row) for row in values]
        else:
            length_values = lengths.numpy()
            output_lengths = length_values.tolist() if hasattr(length_values, "tolist") else list(length_values)
        join = detokenize or (lambda tokens: " ".join(tokens))
        results: list[list[Any]] = []
        for batch_index, (row, length) in enumerate(zip(values, output_lengths)):
            # OpenNMT-tf exports [batch, n_best, time], while exports with a
            # single hypothesis may omit the middle dimension.
            hypotheses = row if len(row) and not isinstance(row[0], (bytes, str)) else [row]
            lengths_for_row = length if isinstance(length, (list, tuple)) else [length]
            if n_best > len(hypotheses):
                raise ValueError(f"SavedModel returned only {len(hypotheses)} hypotheses; n_best={n_best} was requested")
            translations: list[str] = []
            for hypothesis, hypothesis_length in zip(hypotheses[:n_best], lengths_for_row[:n_best]):
                tokens = [token.decode("utf-8") if isinstance(token, bytes) else str(token) for token in hypothesis[: int(hypothesis_length)]]
                if tokens and tokens[-1] == "</s>":
                    tokens.pop()
                text = join(tokens)
                raw_score = float(score_values[batch_index][len(translations)]) if score_values is not None else 0.0
                translations.append(Prediction(text, raw_score, raw_score) if return_scores else text)
            results.append(translations)
        return results if n_best > 1 else [result[0] for result in results]

    def translate(self, text: str, **kwargs: Any) -> str | list[str]:
        """Translate one pre-tokenized sentence; return a list when ``n_best > 1``."""
        return self.translate_many([text], **kwargs)[0]


def _ensure_gather_tree_op(tf: Any) -> None:
    """Register the decoding op removed with TensorFlow Addons.

    Old OpenNMT-tf exports use ``Addons>GatherTree``. Modern Python versions
    have no compatible tensorflow-addons wheel, so compile this small CPU-only
    compatibility op once into the system temporary directory.
    """
    try:
        tf.compat.v1.get_default_graph().op_def_for_type("Addons>GatherTree")
        return
    except (KeyError, RuntimeError):
        pass
    source = Path(__file__).with_name("gather_tree_op.cc")
    digest = hashlib.sha256(source.read_bytes() + tf.__version__.encode()).hexdigest()[:16]
    library = Path(tempfile.gettempdir()) / f"onmt_runner_gather_tree_{digest}.so"
    if not library.exists():
        compile_flags = tf.sysconfig.get_compile_flags()
        link_flags = tf.sysconfig.get_link_flags()
        command = ["g++", "-std=c++17", "-shared", "-fPIC", "-O2", str(source), "-o", str(library), *compile_flags, *link_flags]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise UnsupportedCheckpointError("this SavedModel needs a C++ compiler to build its GatherTree compatibility op") from error
        except subprocess.CalledProcessError as error:
            raise UnsupportedCheckpointError(
                "could not build the GatherTree compatibility op:\n" + error.stderr[-2000:]
            ) from error
    tf.load_op_library(str(library))
