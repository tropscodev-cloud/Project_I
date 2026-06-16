import requests

def test_endpoints():
    print("Testing Crowd Density endpoint...")
    try:
        res = requests.get("http://127.0.0.1:8000/api/v1/usecases/crowd/cam1", timeout=5)
        print("Crowd density status code:", res.status_code)
        if res.status_code == 200:
            payload = res.json().get("chatbot_payload")
            print("Default Chatbot Payload:\n", payload)
            assert "*" not in payload, "Asterisk found in payload!"
            assert "🛡️" not in payload and "❤️" not in payload, "Emoji found in payload!"
            assert "Mathematical proof" in payload, "Mathematical proof not found in payload!"
    except Exception as e:
        print("Crowd density default test failed:", e)

    print("\nTesting Crowd Density custom query ('highest density')...")
    try:
        res = requests.get("http://127.0.0.1:8000/api/v1/usecases/crowd/cam1?q=highest+density", timeout=5)
        print("Crowd density custom query status code:", res.status_code)
        if res.status_code == 200:
            payload = res.json().get("chatbot_payload")
            print("Custom Chatbot Payload:\n", payload)
            assert "*" not in payload, "Asterisk found in payload!"
            assert "🛡️" not in payload and "❤️" not in payload, "Emoji found in payload!"
            assert "Mathematical proof of peak density" in payload, "Mathematical proof not found in payload!"
            print("Crowd Density custom query passed!")
    except Exception as e:
        print("Crowd density custom query test failed:", e)

    print("\nTesting Hidden Associates endpoint...")
    try:
        res = requests.get("http://127.0.0.1:8000/api/v1/usecases/security/associates/42", timeout=5)
        print("Hidden Associates status code:", res.status_code)
        if res.status_code == 200:
            payload = res.json().get("chatbot_payload")
            print("Default Chatbot Payload:\n", payload)
            assert "*" not in payload, "Asterisk found in payload!"
            assert "🛡️" not in payload and "❤️" not in payload, "Emoji found in payload!"
            assert "Mathematical proof" in payload, "Mathematical proof not found in payload!"
    except Exception as e:
        print("Hidden Associates default test failed:", e)

    print("\nTesting Hidden Associates custom query ('who is the most connected person')...")
    try:
        res = requests.get("http://127.0.0.1:8000/api/v1/usecases/security/associates/42?q=who+is+the+most+connected+person", timeout=5)
        print("Hidden Associates custom query status code:", res.status_code)
        if res.status_code == 200:
            payload = res.json().get("chatbot_payload")
            print("Custom Chatbot Payload:\n", payload)
            assert "*" not in payload, "Asterisk found in payload!"
            assert "🛡️" not in payload and "❤️" not in payload, "Emoji found in payload!"
            assert "Mathematical proof of connectivity" in payload, "Mathematical proof not found in payload!"
            print("Hidden Associates custom query passed!")
    except Exception as e:
        print("Hidden Associates custom query test failed:", e)

if __name__ == "__main__":
    test_endpoints()
