from types import SimpleNamespace

from gateway.zenos_runtime import apply_host_token_budget, restore_host_token_budget


def test_host_budget_is_bounded_for_one_turn_and_then_restored():
    agent = SimpleNamespace(max_iterations=90, max_tokens=None)
    state = apply_host_token_budget(
        agent,
        {
            "budgetId": "budget-test",
            "reservationId": "host-test",
            "maxCalls": 3,
            "maxOutputTokens": 1200,
        },
    )

    assert state["applied"] is True
    assert agent.max_iterations == 3
    assert agent.max_tokens == 1200

    restore_host_token_budget(agent, state)
    assert agent.max_iterations == 90
    assert agent.max_tokens is None


def test_host_budget_never_expands_an_existing_stricter_cap():
    agent = SimpleNamespace(max_iterations=2, max_tokens=700)
    state = apply_host_token_budget(agent, {"maxCalls": 8, "maxOutputTokens": 4000})

    assert agent.max_iterations == 2
    assert agent.max_tokens == 700

    restore_host_token_budget(agent, state)
    assert agent.max_iterations == 2
    assert agent.max_tokens == 700
