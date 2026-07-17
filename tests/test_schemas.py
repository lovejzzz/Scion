from scion.schemas import response_format, response_schema


def test_all_coursemapper_kinds_have_strict_json_schemas() -> None:
    for kind in ("lesson", "mc-item", "key-term", "source-bundle"):
        value = response_format(kind)
        assert value["type"] == "json_schema"
        assert value["json_schema"]["strict"] is True
        assert response_schema(kind)["additionalProperties"] is False
