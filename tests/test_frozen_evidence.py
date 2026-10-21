from scripts.verify_evidence import verify_frozen_files


def test_historical_evidence_and_inputs_unchanged():
    assert verify_frozen_files() == 17
