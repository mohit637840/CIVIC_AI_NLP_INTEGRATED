from app.services.nlp_service import nlp_service


def test_nlp_adapter_returns_teammate_contract():
    result = nlp_service.analyze("There is a pothole on the road near the school")

    assert result["complaint"] == "There is a pothole on the road near the school"
    assert result["primary_category"]
    assert "severity" in result
    assert "priority" in result
    assert "department" in result
    assert "suggested_action" in result
