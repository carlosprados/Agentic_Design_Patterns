import time
from typing import Callable, Any

def evaluate_response_accuracy(agent_output: str, expected_output: str) -> float:
    """
    Calculates a simple binary accuracy score.
    """
    is_correct = agent_output.strip().lower() == expected_output.strip().lower()
    return 1.0 if is_correct else 0.0

def timed_agent_action(agent_function: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measures execution time of an agent action in milliseconds.
    """
    start_time = time.perf_counter()
    result = agent_function(*args, **kwargs)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    return result, latency_ms

class LLMInteractionMonitor:
    """
    Conceptual monitor for tracking token usage across interactions.
    """
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def record_interaction(self, prompt: str, response: str):
        # Placeholder splitting; use tiktoken or similar for real projects
        input_tokens = len(prompt.split())
        output_tokens = len(response.split())
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        print(f"Metrics: In={input_tokens}, Out={output_tokens}")

    def get_summary(self):
        return {
            "total_input": self.total_input_tokens,
            "total_output": self.total_output_tokens
        }

if __name__ == "__main__":
    print("--- Basic Agent Evaluation & Monitoring Demo ---")
    
    # Accuracy
    score = evaluate_response_accuracy("Paris", "Paris")
    print(f"Accuracy Score: {score}")
    
    # Latency
    def dummy_tool(q): time.sleep(0.1); return f"Result for {q}"
    res, latency = timed_agent_action(dummy_tool, "test")
    print(f"Action Latency: {latency:.2f} ms")
    
    # Tokens
    monitor = LLMInteractionMonitor()
    monitor.record_interaction("Hello", "Hi there!")
    print(f"Monitoring Summary: {monitor.get_summary()}")
