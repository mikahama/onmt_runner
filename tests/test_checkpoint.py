from importlib.util import find_spec
from pathlib import Path
import unittest

from onmt_runner import TensorFlowTranslator, Translator


MODEL = Path(__file__).parents[1] / "murre_norm_default.pt"


@unittest.skipUnless(MODEL.exists(), "example checkpoint is not present")
class ExampleCheckpointTests(unittest.TestCase):
    def test_loads_example_character_checkpoint(self) -> None:
        translator = Translator.load(MODEL)
        result = translator.translate("h e i", max_length=30)
        self.assertIsInstance(result, str)
        self.assertTrue(result)
        self.assertEqual(translator.translate("h e i", max_length=30, detokenize="".join), result.replace(" ", ""))


    def test_batch_and_single_translation_agree(self) -> None:
        translator = Translator.load(MODEL)
        self.assertEqual(translator.translate_many(["h e i", "h e i"], max_length=30), [
            translator.translate("h e i", max_length=30),
            translator.translate("h e i", max_length=30),
        ])


class TensorFlowBackendTests(unittest.TestCase):
    def test_uses_opennmt_tf_serving_token_and_length_contract(self) -> None:
        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def numpy(self):
                return self.value

        class FakeSignature:
            structured_input_signature = ((), {"tokens": object(), "length": type("Spec", (), {"dtype": "int32"})()})

            def __call__(self, *, tokens, length):
                self.tokens, self.length = tokens, length
                return {"tokens": FakeTensor([[[b"hello", b"there", b"</s>"]]]), "length": FakeTensor([[3]])}

        class FakeTF:
            string = "string"

            @staticmethod
            def constant(value, dtype=None):
                return value

        translator = TensorFlowTranslator(FakeSignature())
        translator._tensorflow = lambda: FakeTF()  # type: ignore[method-assign]
        self.assertEqual(translator.translate("Kainuu h e i"), "hello there")

    @unittest.skipUnless(find_spec("tensorflow") and Path("flags_dist/saved_model.pb").exists(), "TensorFlow or flags_dist is not present")
    def test_loads_provided_opennmt_tf_saved_model(self) -> None:
        translator = Translator.load("flags_dist")
        result = translator.translate("Kainuu h e i")
        self.assertEqual(result, "h e i")
