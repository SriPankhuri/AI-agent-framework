class MockLLM:
    """
    Fake LLM so framework can run without real AI model
    """

    def generate(self, prompt: str):
        return f"[MOCK LLM RESPONSE] {prompt}"

    def synthesize(self, original_input, results):
        """
        Combine all task outputs into final response
        """
        summary = "\n".join(
            f"{task}: {data.get('data')}"
            for task, data in results.items()
        )

        return f"""
===== FINAL AGENT REPORT =====
User Request: {original_input}

Workflow Results:
{summary}

===== END REPORT =====
"""
