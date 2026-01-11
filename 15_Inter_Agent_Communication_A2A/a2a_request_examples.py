import json

# Synchronous Request (JSON-RPC)
sync_request = {
  "jsonrpc": "2.0",
  "id": "1",
  "method": "sendTask",
  "params": {
    "id": "task-001",
    "sessionId": "session-001",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "What is the exchange rate from USD to EUR?"
        }
      ]
    },
    "acceptedOutputModes": ["text/plain"],
    "historyLength": 5
  }
}

# Streaming Request (JSON-RPC)
streaming_request = {
  "jsonrpc": "2.0",
  "id": "2",
  "method": "sendTaskSubscribe",
  "params": {
    "id": "task-002",
    "sessionId": "session-001",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "What's the exchange rate for JPY to GBP today?"
        }
      ]
    },
    "acceptedOutputModes": ["text/plain"],
    "historyLength": 5
  }
}

def save_examples():
    with open("a2a_sync_request.json", "w") as f:
        json.dump(sync_request, f, indent=2)
    with open("a2a_streaming_request.json", "w") as f:
        json.dump(streaming_request, f, indent=2)
    print("A2A JSON-RPC request examples saved to .json files.")

if __name__ == "__main__":
    save_examples()
