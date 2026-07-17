from scion.evaluate import parse_sse_chat, validate_browser_preflight


def test_browser_preflight_accepts_echoed_coursemapper_origin() -> None:
    assert (
        validate_browser_preflight(
            200,
            {
                "Access-Control-Allow-Origin": "http://localhost:5173",
                "Access-Control-Allow-Methods": "GET, POST",
                "Access-Control-Allow-Headers": "*",
            },
            origin="http://localhost:5173",
        )
        == []
    )


def test_browser_preflight_reports_missing_permissions() -> None:
    assert validate_browser_preflight(
        403,
        {"Access-Control-Allow-Origin": "https://example.invalid"},
        origin="http://localhost:5173",
    ) == ["cors-status:403", "cors-origin", "cors-post-method", "cors-content-type-header"]


def test_parse_sse_chat_collects_coursemapper_deltas() -> None:
    content, saw_done = parse_sse_chat(
        [
            b": keep-alive\n",
            b'data: {"choices":[{"delta":{"content":"{\\"q\\":"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"\\"Test?\\"}"}}]}\n',
            b"data: [DONE]\n",
        ]
    )
    assert content == '{"q":"Test?"}'
    assert saw_done is True
