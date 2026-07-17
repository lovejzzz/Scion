from scion.constants import MAX_LITE_BROWSER_ARTIFACT_BYTES, MAX_SCION_ARTIFACT_BYTES
from scion.model_registry import MODEL_PINS, student_pin


def test_all_model_pins_are_immutable_and_open() -> None:
    for pin in MODEL_PINS.values():
        assert len(pin.revision) == 40
        assert pin.license == "Apache-2.0"
    assert MODEL_PINS["optional-teacher"].required is False
    assert MODEL_PINS["teacher"].model_id == "mlx-community/Qwen3.6-27B-8bit"
    assert MODEL_PINS["critic"].model_id == "mlx-community/gemma-4-31b-it-4bit"


def test_students_and_artifact_caps_match_delivery_contract() -> None:
    assert student_pin("lite").model_id == "google/gemma-4-E2B-it-qat-q4_0-unquantized"
    assert student_pin("pro").model_id == "google/gemma-4-12B-it-qat-q4_0-unquantized"
    assert MAX_LITE_BROWSER_ARTIFACT_BYTES == 64 * 1024 * 1024
    assert MAX_SCION_ARTIFACT_BYTES == 1_000_000_000
