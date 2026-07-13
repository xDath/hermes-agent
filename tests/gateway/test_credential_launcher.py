from pathlib import Path

from scripts.run_gateway_with_credential import credential_environment


def test_credential_launcher_parses_values_as_data_without_shell_execution(tmp_path: Path):
    marker = tmp_path / "must-not-exist"
    credential = tmp_path / "zenos-runtime.env"
    credential.write_text(
        "\n".join(
            [
                "SIMPLE=value",
                'SPACED="Bearer NFT example token"',
                f'DATA_ONLY="$(touch {marker})"',
                "1INVALID=ignored",
            ]
        ),
        encoding="utf-8",
    )

    parsed = credential_environment(credential)

    assert parsed["SIMPLE"] == "value"
    assert parsed["SPACED"] == "Bearer NFT example token"
    assert parsed["DATA_ONLY"] == f"$(touch {marker})"
    assert "1INVALID" not in parsed
    assert not marker.exists()
