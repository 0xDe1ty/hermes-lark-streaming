from hermes_lark_streaming.hooks import _request_effort


def test_request_effort_uses_wire_value():
    assert _request_effort({"request": {"body": {"reasoning": {"effort": "high"}}}}) == "high"
    assert _request_effort({"request": {"body": {"extra_body": {"reasoning": {"effort": "max"}}}}}) == "max"
    assert _request_effort({"request": {"body": {"reasoning_effort": "low"}}}) == "low"
    assert _request_effort({"request": {"body": {"reasoning": {"enabled": False}}}}) == "none"
    assert _request_effort({"request": {"_truncated": True}}) is None


def test_matching_request_ids_keeps_concurrent_turns_separate(monkeypatch):
    from hermes_lark_streaming import hooks

    observed = []
    monkeypatch.setattr(
        hooks, '_request_key', lambda payload: ('profile', payload['source'], payload['api_request_id'])
    )
    monkeypatch.setattr(hooks, '_dispatch', lambda kind, payload: observed.append(payload))
    hooks._requests.clear()
    hooks.observe_request(source='a', api_request_id='1', request={'body': {'reasoning': {'effort': 'high'}}})
    hooks.observe_request(source='b', api_request_id='2', request={'body': {'reasoning': {'effort': 'max'}}})
    hooks.observe_usage(source='b', api_request_id='2')
    hooks.observe_usage(source='a', api_request_id='1')
    assert [item['reasoning_effort'] for item in observed] == ['max', 'high']
