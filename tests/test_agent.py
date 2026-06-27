import os
import json
import csv
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agent.forensic_agent import ForensicRAGAgent

client = TestClient(app)

@pytest.fixture
def mock_agent_env(tmp_path, monkeypatch):
    # Setup mock reports and metrics files
    metrics_file = tmp_path / "metrics_summary.json"
    health_file = tmp_path / "dataset_health.md"
    metadata_file = tmp_path / "metadata.csv"

    # Write dummy metrics
    metrics_data = {
        "accuracy": 0.885,
        "precision": 0.892,
        "recall": 0.871,
        "f1_score": 0.881,
        "auc": 0.941
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f)

    # Write dummy health report
    with open(health_file, "w", encoding="utf-8") as f:
        f.write("# Dataset Health Report\n")
        f.write("Status: completed\n")
        f.write("Total video duplicates: 4\n")
        f.write("No corrupt files found.\n")

    # Write dummy metadata
    with open(metadata_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label"])
        writer.writerow(["real/001.mp4", "real"])
        writer.writerow(["real/002.mp4", "real"])
        writer.writerow(["fake/003.mp4", "fake"])

    # Monkeypatch paths in agent class
    monkeypatch.setattr(ForensicRAGAgent, "metrics_path", str(metrics_file))
    monkeypatch.setattr(ForensicRAGAgent, "health_report_path", str(health_file))
    monkeypatch.setattr(ForensicRAGAgent, "metadata_path", str(metadata_file))

    return {
        "metrics_file": str(metrics_file),
        "health_file": str(health_file),
        "metadata_file": str(metadata_file)
    }

def test_agent_context_retrieval(mock_agent_env):
    agent = ForensicRAGAgent()
    context = agent.retrieve_context("accuracy metrics")
    
    assert context["metrics"] is not None
    assert context["metrics"]["accuracy"] == 0.885
    assert context["metadata_summary"]["total_videos"] == 3
    assert context["metadata_summary"]["real_videos"] == 2
    assert context["metadata_summary"]["fake_videos"] == 1

def test_agent_answering_offline(mock_agent_env):
    agent = ForensicRAGAgent()
    
    # 1. Performance query
    ans_perf = agent.answer_query("What is the model accuracy?")
    assert "88.50%" in ans_perf
    assert "88.10%" in ans_perf  # F1 score

    # 2. Stats query
    ans_stats = agent.answer_query("How many real videos are there?")
    assert "3 total videos" in ans_stats
    assert "Real videos: 2" in ans_stats

    # 3. Health query
    ans_health = agent.answer_query("Are there duplicates?")
    assert "duplicates: 4" in ans_health

def test_agent_endpoint(mock_agent_env):
    # Test POST request to FastAPI endpoint
    response = client.post(
        "/agent/query",
        json={"query": "What is the accuracy of the model?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "88.50%" in data["answer"]
