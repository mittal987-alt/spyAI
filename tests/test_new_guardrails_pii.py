from app.diagnostics.guardrails import scan_prompt_guardrails

def test_new_pii_patterns():
    # 1. Indian Aadhaar number
    res = scan_prompt_guardrails("My Aadhaar is 1234-5678-9012 for verification.")
    assert any(p["entity_type"] == "AADHAAR_NUMBER" for p in res["pii"])
    assert "[AADHAAR_REDACTED]" in res["sanitized_text"]

    # 2. Indian PAN card
    res = scan_prompt_guardrails("Income tax PAN is ABCDE1234F.")
    assert any(p["entity_type"] == "PAN_CARD" for p in res["pii"])
    assert "[PAN_REDACTED]" in res["sanitized_text"]

    # 3. Stripe Secret Key
    res = scan_prompt_guardrails("My test key is sk_test_51MzXYZabc12345678901234567890.")
    assert any(p["entity_type"] == "STRIPE_KEY" for p in res["pii"])
    assert "[STRIPE_KEY_REDACTED]" in res["sanitized_text"]

    # 4. Google API Key
    res = scan_prompt_guardrails("API key: AIzaSyD1234567890abcdefghijklmnopqrstuv")
    assert any(p["entity_type"] == "GOOGLE_API_KEY" for p in res["pii"])
    assert "[GOOGLE_KEY_REDACTED]" in res["sanitized_text"]

    # 5. IBAN
    res = scan_prompt_guardrails("Transfer to DE89370400440532013000 please.")
    assert any(p["entity_type"] == "IBAN" for p in res["pii"])
    assert "[IBAN_REDACTED]" in res["sanitized_text"]
