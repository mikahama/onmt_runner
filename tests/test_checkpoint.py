from importlib.util import find_spec
from pathlib import Path
import unittest

from onmt_runner import Prediction, TensorFlowTranslator, Translator


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

    def test_recurrent_n_best_returns_ranked_hypotheses(self) -> None:
        translator = Translator.load(MODEL)
        hypotheses = translator.translate("h e i", max_length=30, beam_size=3, n_best=2)
        self.assertIsInstance(hypotheses, list)
        self.assertEqual(len(hypotheses), 2)
        self.assertTrue(all(isinstance(hypothesis, str) for hypothesis in hypotheses))

    def test_recurrent_returns_prediction_score(self) -> None:
        prediction = Translator.load(MODEL).translate("h e i", return_scores=True)
        self.assertIsInstance(prediction, Prediction)
        self.assertLess(prediction.score, 0)


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
                return {
                    "tokens": FakeTensor([
                        [[b"hello", b"there", b"</s>"], [b"hi", b"</s>", b"</s>"]]
                    ]),
                    "length": FakeTensor([[3, 2]]),
                }

        class FakeTF:
            string = "string"

            @staticmethod
            def constant(value, dtype=None):
                return value

        translator = TensorFlowTranslator(FakeSignature())
        translator._tensorflow = lambda: FakeTF()  # type: ignore[method-assign]
        self.assertEqual(translator.translate("Kainuu h e i"), "hello there")
        self.assertEqual(translator.translate("Kainuu h e i", n_best=1), "hello there")
        self.assertEqual(translator.translate("Kainuu h e i", n_best=2), ["hello there", "hi"])
        self.assertIsInstance(translator.translate("Kainuu h e i", return_scores=True), Prediction)

    @unittest.skipUnless(find_spec("tensorflow") and Path("flags_dist/saved_model.pb").exists(), "TensorFlow or flags_dist is not present")
    def test_loads_provided_opennmt_tf_saved_model(self) -> None:
        translator = Translator.load("flags_dist")
        result = translator.translate("Kainuu h e i")
        self.assertEqual(result, "h e i")
