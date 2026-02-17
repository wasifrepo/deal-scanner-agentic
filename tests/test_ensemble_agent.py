from agents.ensemble_agent import EnsembleAgent


class _DummyPreprocessor:
    model_name = "dummy-preprocessor"

    def __init__(self):
        self.inputs = []

    def preprocess(self, text: str) -> str:
        self.inputs.append(text)
        return f"rewritten::{text}"


class _DummySpecialist:
    def __init__(self):
        self.inputs = []

    def price(self, text: str) -> float:
        self.inputs.append(text)
        return 100.0


class _DummyFrontier:
    def __init__(self, _collection):
        self.inputs = []

    def price(self, text: str) -> float:
        self.inputs.append(text)
        return 200.0


def test_ensemble_uses_frontier_and_specialist_only(monkeypatch):
    monkeypatch.setattr("agents.ensemble_agent.Preprocessor", _DummyPreprocessor)
    monkeypatch.setattr("agents.ensemble_agent.SpecialistAgent", _DummySpecialist)
    monkeypatch.setattr("agents.ensemble_agent.FrontierAgent", _DummyFrontier)

    agent = EnsembleAgent(collection=None)
    result = agent.price("sample description")

    assert result == 190.0
    assert agent.specialist.inputs == ["rewritten::sample description"]
    assert agent.frontier.inputs == ["rewritten::sample description"]
