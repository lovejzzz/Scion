from scion.schemas import contract_response_schema, response_format, response_schema


def test_all_coursemapper_kinds_have_strict_json_schemas() -> None:
    for kind in ("lesson", "mc-item", "key-term", "source-bundle"):
        value = response_format(kind)
        assert value["type"] == "json_schema"
        assert value["json_schema"]["strict"] is True
        assert response_schema(kind)["additionalProperties"] is False


def test_locked_contract_schemas_are_closed_objects() -> None:
    for contract in (
        "prerequisite-json-v1",
        "schedule-json-v1",
        "degree-audit-json-v1",
        "uncertainty-json-v1",
        "tutor-json-v1",
        "tool-call-json-v1",
        "safety-json-v1",
        "coursemapper-kernel-json-v1",
    ):
        schema = contract_response_schema(contract)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
