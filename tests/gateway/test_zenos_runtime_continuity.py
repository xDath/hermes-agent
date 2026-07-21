from gateway.zenos_runtime import build_continuity_packet, compute_continuity_packet_hash


def _history(extra_tail=False):
    messages = [
        {
            "id": "goal-0",
            "role": "user",
            "content": (
                "Upgrade Zenos Runtime and preserve this original goal: one user command "
                "must continue through compaction and restart without asking the user to type lanjut."
            ),
            "created_at": "2026-07-21T00:00:00Z",
        },
        {
            "id": "constraint-1",
            "role": "user",
            "content": "Constraint: do not bypass approval for destructive or deploy mutations.",
            "created_at": "2026-07-21T00:00:01Z",
        },
    ]
    for index in range(2, 302):
        messages.append(
            {
                "id": f"noise-{index}",
                "role": "assistant" if index % 2 else "user",
                "content": f"Routine discussion item {index} without a state transition. " + ("x" * 480),
                "created_at": f"2026-07-21T00:{(index // 60) % 60:02d}:{index % 60:02d}Z",
            }
        )
    messages.insert(
        170,
        {
            "id": "decision-170",
            "role": "assistant",
            "content": "Decision: Runtime ContinuityCoordinator is the only checkpoint authority.",
            "created_at": "2026-07-21T01:00:00Z",
        },
    )
    messages.insert(
        260,
        {
            "id": "patch-260",
            "role": "tool",
            "name": "edit",
            "content": "Patch applied to app/lib/gateway-continuity.ts and app/lib/continuity-coordinator.ts.",
            "created_at": "2026-07-21T01:10:00Z",
        },
    )
    messages.insert(
        285,
        {
            "id": "validation-285",
            "role": "tool",
            "name": "test",
            "content": "Validation passed: npm test completed with 114 tests and zero failures.",
            "created_at": "2026-07-21T01:20:00Z",
        },
    )
    messages.append(
        {
            "id": "tail-final",
            "role": "user",
            "content": "Final instruction: continue the remaining validation and deliver one terminal answer.",
            "created_at": "2026-07-21T01:30:00Z",
        }
    )
    if extra_tail:
        messages.append(
            {
                "id": "tail-extra",
                "role": "tool",
                "name": "test",
                "content": "Validation passed: cross-language packet hash fixture now matches.",
                "created_at": "2026-07-21T01:31:00Z",
            }
        )
    return messages


def _serialized(packet):
    sections = [
        packet["head"],
        packet["milestones"],
        packet["activeToolState"],
        packet["openWork"],
        packet["recentTail"],
    ]
    return repr(sections)


def test_continuity_packet_hash_matches_the_cross_language_contract_fixture():
    packet = {
        "version": "continuity-v2",
        "sessionId": "fixture-session",
        "turnId": "fixture-turn",
        "sourceCursor": "msg:3:fixture",
        "estimatedTokens": 123456,
        "head": [
            {
                "role": "user",
                "content": "Preserve tujuan utama.",
                "message_id": "m0",
            }
        ],
        "milestones": [
            {
                "kind": "decision",
                "text": "Runtime owns checkpoints.",
                "sourceMessageIds": ["m1"],
                "sourceHash": "a" * 64,
                "occurredAt": "2026-07-21T00:00:00.000Z",
            }
        ],
        "recentTail": [
            {
                "role": "user",
                "content": "Continue validation.",
                "message_id": "m2",
            }
        ],
        "activeToolState": [],
        "openWork": [],
        "previousCheckpointId": "checkpoint-0",
    }

    assert compute_continuity_packet_hash(packet) == (
        "fa768c2c48eb15230c08088c924926001c71670c650c95f11f8bc533a1ec67d9"
    )
    assert compute_continuity_packet_hash({**packet, "contentHash": "tampered"}) == (
        "fa768c2c48eb15230c08088c924926001c71670c650c95f11f8bc533a1ec67d9"
    )


def test_continuity_packet_is_retry_deterministic_and_preserves_full_history_signals():
    history = _history()
    first = build_continuity_packet(
        history,
        session_id="session-continuity",
        turn_id="turn-continuity",
        estimated_tokens=190_000,
        max_chars=80_000,
        max_messages=300,
        previous_checkpoint_id="checkpoint-previous",
    )
    retry = build_continuity_packet(
        history,
        session_id="session-continuity",
        turn_id="turn-continuity",
        estimated_tokens=190_000,
        max_chars=80_000,
        max_messages=300,
        previous_checkpoint_id="checkpoint-previous",
    )

    assert first is not None
    assert retry == first
    assert first["version"] == "continuity-v2"
    assert first["previousCheckpointId"] == "checkpoint-previous"
    assert first["sourceCursor"].startswith("msg:")
    assert len(first["contentHash"]) == 64

    serialized = _serialized(first)
    assert "one user command" in serialized
    assert "only checkpoint authority" in serialized
    assert "app/lib/gateway-continuity.ts" in serialized
    assert "114 tests and zero failures" in serialized
    assert "deliver one terminal answer" in serialized


def test_continuity_packet_cursor_and_hash_change_only_after_new_source_evidence():
    first = build_continuity_packet(
        _history(),
        session_id="session-continuity",
        turn_id="turn-continuity",
        estimated_tokens=190_000,
        max_chars=80_000,
    )
    changed = build_continuity_packet(
        _history(extra_tail=True),
        session_id="session-continuity",
        turn_id="turn-continuity",
        estimated_tokens=190_200,
        max_chars=80_000,
    )

    assert first is not None
    assert changed is not None
    assert changed["sourceCursor"] != first["sourceCursor"]
    assert changed["contentHash"] != first["contentHash"]
    assert "cross-language packet hash fixture now matches" in _serialized(changed)


def test_continuity_packet_remains_bounded_with_large_tool_payloads():
    history = _history()
    history.insert(
        250,
        {
            "id": "large-tool",
            "role": "tool",
            "name": "terminal",
            "content": "Large test payload " + ("z" * 80_000),
            "created_at": "2026-07-21T01:05:00Z",
        },
    )
    packet = build_continuity_packet(
        history,
        session_id="session-continuity",
        turn_id="turn-continuity",
        estimated_tokens=240_000,
        max_chars=40_000,
        max_messages=120,
    )

    assert packet is not None
    assert len(repr(packet["head"])) <= 8_000
    assert len(repr(packet["milestones"])) <= 16_000
    assert len(repr(packet["activeToolState"])) <= 10_000
    assert len(repr(packet["recentTail"])) <= 20_000
