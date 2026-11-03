"""The registry resolves a method's factory by name, with no hard-coded method branch."""
import pytest

from federatedscope.contrib.decentralized import registry


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(registry._FACTORIES)
    registry.clear_registry()
    yield
    registry._FACTORIES.clear()
    registry._FACTORIES.update(saved)


def test_name_normalisation_folds_case_and_hyphens():
    assert registry.normalise_method_name("Dec-LoRA") == "dec_lora"
    assert registry.normalise_method_name("  dec_lora  ") == "dec_lora"
    assert registry.normalise_method_name(None) == ""


def test_reserved_names_need_no_plugin_import():
    """Recognising a reserved name needs no plugin: asking registers nothing."""
    assert registry.is_reserved_name("dec_lora")
    assert registry.is_reserved_name("DEC-LORA")
    assert not registry.is_reserved_name("FedAvg")
    assert not registry.is_reserved_name("hetlora")
    # nothing was registered by asking
    assert registry.known_methods() == ()


def test_register_and_resolve():
    sentinel = object()
    registry.register_decentralized_method("toy_method", sentinel)

    assert registry.is_decentralized_method("toy_method")
    assert registry.is_decentralized_method("Toy-Method")
    assert registry.get_factory("toy_method") is sentinel
    assert registry.known_methods() == ("toy_method",)


def test_unknown_method_raises_with_a_useful_message():
    registry.register_decentralized_method("toy_method", object())
    with pytest.raises(KeyError) as excinfo:
        registry.get_factory("nope")
    assert "toy_method" in str(excinfo.value)


def test_empty_name_rejected():
    with pytest.raises(ValueError):
        registry.register_decentralized_method("   ", object())
