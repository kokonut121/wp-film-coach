"""Tests for pipeline/agent.py."""

from unittest.mock import MagicMock, patch

from pipeline.agent import summarize_events, generate_report, stream_chat


SAMPLE_EVENTS_JSON = {
    "meta": {
        "duration_s": 600,
        "fps": 30,
        "team_a_colour": "team_a",
        "team_b_colour": "team_b",
    },
    "positions": [
        {"t_seconds": 0, "player_id": 0, "team": "team_a", "x_metres": 10.0, "y_metres": 6.5},
        {"t_seconds": 1, "player_id": 1, "team": "team_b", "x_metres": 15.0, "y_metres": 6.5},
    ],
    "events": [
        {"t_seconds": 120.5, "type": "turnover", "detail": "team_a→team_b", "location": None},
        {"t_seconds": 245.0, "type": "man_up", "detail": "team_a man-up (6v5)", "location": None},
        {"t_seconds": 300.0, "type": "counter_attack", "detail": "team_b counter-attack (4 players advanced)", "location": None},
    ],
    "formations": [
        {"t_seconds": 0, "team": "team_a", "formation": "3-3", "confidence": 0.85},
        {"t_seconds": 60, "team": "team_a", "formation": "4-2", "confidence": 0.72},
    ],
    "metrics": {
        "heatmaps": {"0": [[1, 0], [2, 3]], "1": [[0, 1], [1, 0]]},
        "possession": {"period_1": {"team_a": 0.55, "team_b": 0.45}},
        "hull_area": [
            {"t_seconds": 0, "team_a_area": 30.0, "team_b_area": 25.0},
            {"t_seconds": 10, "team_a_area": 28.0, "team_b_area": 32.0},
        ],
        "centroid_spread": [
            {"t_seconds": 0, "team_a_spread": 4.5, "team_b_spread": 5.0},
        ],
    },
}


class TestSummarizeEvents:
    def test_produces_nonempty_string(self):
        summary = summarize_events(SAMPLE_EVENTS_JSON)
        assert isinstance(summary, str)
        assert len(summary) > 100

    def test_contains_event_log(self):
        summary = summarize_events(SAMPLE_EVENTS_JSON)
        assert "TURNOVER" in summary
        assert "MAN_UP" in summary
        assert "COUNTER_ATTACK" in summary

    def test_contains_formations(self):
        summary = summarize_events(SAMPLE_EVENTS_JSON)
        assert "3-3" in summary
        assert "4-2" in summary

    def test_contains_possession(self):
        summary = summarize_events(SAMPLE_EVENTS_JSON)
        assert "period_1" in summary

    def test_contains_timestamps(self):
        summary = summarize_events(SAMPLE_EVENTS_JSON)
        assert "02:00" in summary  # 120.5s
        assert "04:05" in summary  # 245s
        assert "05:00" in summary  # 300s


class TestGenerateReport:
    @patch("pipeline.agent.anthropic.Anthropic")
    def test_returns_report_with_sections(self, mock_anthropic_class, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = (
            "## Summary\nGame overview.\n\n"
            "## Key Moments\n- T=02:00 Turnover\n\n"
            "## Tactical Patterns\nPatterns here.\n\n"
            "## Individual Notes\nPlayer notes.\n\n"
            "## Recommendations\n1. Do this."
        )
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        summary = summarize_events(SAMPLE_EVENTS_JSON)
        report = generate_report(summary)

        assert "## Summary" in report
        assert "## Key Moments" in report
        assert "## Tactical Patterns" in report
        assert "## Individual Notes" in report
        assert "## Recommendations" in report

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"


class TestStreamChat:
    @patch("pipeline.agent.anthropic.Anthropic")
    def test_yields_text_chunks(self, mock_anthropic_class, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock the streaming context manager
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Hello", " there", "!"])

        mock_client.messages.stream.return_value = mock_stream

        summary = summarize_events(SAMPLE_EVENTS_JSON)
        messages = [{"role": "user", "content": "What happened in the first quarter?"}]

        chunks = list(stream_chat(summary, messages))
        assert len(chunks) == 3
        assert chunks[0] == "Hello"
        assert "".join(chunks) == "Hello there!"
