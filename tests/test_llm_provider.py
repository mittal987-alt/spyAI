"""
Unit tests for unified multi-provider LLM client (Ollama, OpenAI, Heuristic).
"""

from unittest.mock import MagicMock, patch
import pytest

from app.models.llm_provider import (
    check_ollama_status,
    get_available_providers,
    unified_generate,
)


def test_heuristic_generation():
    """Verify deterministic fallback generation produces expected structure without keys or Ollama."""
    result = unified_generate(
        prompt="Explain this drift anomaly in feature X",
        provider="heuristic",
    )
    assert result["provider"] == "heuristic"
    assert result["model"] == "deterministic-heuristic-v1"
    assert "Explain this drift" in result["text"]


def test_get_available_providers_structure():
    """Verify provider discovery format."""
    providers_info = get_available_providers()
    assert "providers" in providers_info
    assert "recommended_provider" in providers_info
    provider_ids = [p["id"] for p in providers_info["providers"]]
    assert "heuristic" in provider_ids
    assert "ollama" in provider_ids
    assert "openai" in provider_ids


def test_ollama_mock_generation():
    """Test Ollama generation handling when Ollama responds with mock JSON."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "LLaMA 3.2 analyzed: Covariate shift detected in feature X.",
        "eval_count": 42,
        "total_duration": 500_000_000,
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = unified_generate(
            prompt="Analyze drift in feature X",
            provider="ollama",
            model="llama3.2",
        )
        assert res["provider"] == "ollama"
        assert res["model"] == "llama3.2"
        assert "LLaMA 3.2" in res["text"]
        assert res["tokens_evaluated"] == 42


def test_auto_provider_resolution_without_network():
    """When Ollama is offline and no OpenAI key is set, auto resolves to heuristic."""
    with patch("app.models.llm_provider.check_ollama_status", return_value={"available": False, "installed_models": []}):
        res = unified_generate(
            prompt="Test prompt",
            provider="auto",
            api_key=None,
        )
        assert res["provider"] == "heuristic"
