from abc import ABC, abstractmethod
from typing import Optional, Any, Generator
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    MOCK = "mock"


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMRequest:
    """Request to an LLM."""
    messages: list[Message]
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False
    stop_sequences: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    provider: str
    model: str
    tokens_used: int
    finish_reason: str  # 'stop', 'length', 'error'
    metadata: dict = field(default_factory=dict)
    latency_ms: float = 0.0


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to the LLM."""
        pass
    
    @abstractmethod
    def stream_complete(self, request: LLMRequest) -> Generator[str, None, None]:
        """Stream a completion response from the LLM."""
        pass
    
    @abstractmethod
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        pass


class MockLLMClient(LLMClient):
    """
    Mock LLM client that simulates responses.
    Useful for testing and development.
    """
    
    def __init__(self, model: str = "mock-gpt-4", delay_ms: int = 100):
        self.model = model
        self.delay_ms = delay_ms
        self.call_count = 0
        self.response_templates = {
            'task_planning': "To accomplish this task, I'll break it down into steps:\n1. First step\n2. Second step\n3. Third step",
            'analysis': "Based on the information provided, here's my analysis:\n- Key point 1\n- Key point 2\n- Conclusion",
            'default': "I understand your request. Here's my response to help with that."
        }
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Simulate an LLM completion."""
        start_time = time.time()
        self.call_count += 1
        
        # Simulate processing delay
        time.sleep(self.delay_ms / 1000.0)
        
        # Generate mock response based on request content
        last_message = request.messages[-1].content.lower()
        
        if 'plan' in last_message or 'steps' in last_message:
            response_text = self.response_templates['task_planning']
        elif 'analyz' in last_message or 'explain' in last_message:
            response_text = self.response_templates['analysis']
        else:
            response_text = self.response_templates['default']
        
        # Add request-specific context
        response_text += f"\n\n[Mock response #{self.call_count} for: {last_message[:50]}...]"
        
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response_text,
            provider=self.get_provider().value,
            model=self.model,
            tokens_used=len(response_text.split()),
            finish_reason='stop',
            latency_ms=latency,
            metadata={'mock': True, 'call_count': self.call_count}
        )
    
    def stream_complete(self, request: LLMRequest) -> Generator[str, None, None]:
        """Simulate streaming response."""
        response = self.complete(request)
        
        # Simulate streaming by yielding word by word
        words = response.content.split()
        for word in words:
            time.sleep(self.delay_ms / 10000.0)  # Faster than full delay
            yield word + " "
    
    def get_provider(self) -> LLMProvider:
        return LLMProvider.MOCK


class PlaceholderLLMClient(LLMClient):
    """
    Placeholder client for real LLM providers.
    Replace the implementation when integrating real APIs.
    """
    
    def __init__(self, provider: LLMProvider, model: str, api_key_placeholder: str = "YOUR_API_KEY_HERE"):
        self.provider = provider
        self.model = model
        self.api_key_placeholder = api_key_placeholder
        
        # Configuration placeholders
        self.config = {
            'base_url': self._get_base_url(),
            'timeout': 30,
            'max_retries': 3
        }
    
    def _get_base_url(self) -> str:
        """Get base URL for provider."""
        urls = {
            LLMProvider.OPENAI: "https://api.openai.com/v1",
            LLMProvider.ANTHROPIC: "https://api.anthropic.com/v1",
            LLMProvider.LOCAL: "http://localhost:8000"
        }
        return urls.get(self.provider, "")
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Placeholder for real API call.
        
        In production, replace with:
        - Actual HTTP requests to provider API
        - Proper authentication headers
        - Error handling and retries
        - Response parsing
        """
        raise NotImplementedError(
            f"Real {self.provider.value} integration not implemented. "
            f"This is a placeholder. To use real LLMs:\n"
            f"1. Add SDK: pip install openai anthropic\n"
            f"2. Set API key in environment\n"
            f"3. Implement actual API calls\n"
            f"For now, use MockLLMClient for testing."
        )
    
    def stream_complete(self, request: LLMRequest) -> Generator[str, None, None]:
        """Placeholder for streaming."""
        raise NotImplementedError("See complete() method for integration steps.")
    
    def get_provider(self) -> LLMProvider:
        return self.provider


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    @staticmethod
    def create(provider: LLMProvider, model: str = None, 
               api_key: str = None, **kwargs) -> LLMClient:
        """
        Create an LLM client.
        
        Args:
            provider: The LLM provider
            model: Model name/ID
            api_key: API key (if needed)
            **kwargs: Additional provider-specific config
            
        Returns:
            LLMClient instance
        """
        if provider == LLMProvider.MOCK:
            return MockLLMClient(
                model=model or "mock-gpt-4",
                delay_ms=kwargs.get('delay_ms', 100)
            )
        
        # For real providers, return placeholder
        return PlaceholderLLMClient(
            provider=provider,
            model=model or "default-model",
            api_key_placeholder=api_key or "YOUR_API_KEY_HERE"
        )


class ConversationManager:
    """Manages conversation history and LLM interactions."""
    
    def __init__(self, client: LLMClient, system_prompt: str = None):
        self.client = client
        self.messages: list[Message] = []
        
        if system_prompt:
            self.add_system_message(system_prompt)
    
    def add_system_message(self, content: str):
        """Add a system message."""
        self.messages.append(Message(role="system", content=content))
    
    def add_user_message(self, content: str, **metadata):
        """Add a user message."""
        self.messages.append(Message(role="user", content=content, metadata=metadata))
    
    def add_assistant_message(self, content: str, **metadata):
        """Add an assistant message."""
        self.messages.append(Message(role="assistant", content=content, metadata=metadata))
    
    def generate_response(self, user_input: str, max_tokens: int = 1000,
                         temperature: float = 0.7) -> LLMResponse:
        """
        Generate a response for user input.
        
        Args:
            user_input: The user's message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            LLMResponse from the LLM
        """
        # Add user message to history
        self.add_user_message(user_input)
        
        # Create request
        request = LLMRequest(
            messages=self.messages.copy(),
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Get response
        response = self.client.complete(request)
        
        # Add assistant response to history
        self.add_assistant_message(response.content)
        
        return response
    
    def stream_response(self, user_input: str, max_tokens: int = 1000,
                       temperature: float = 0.7) -> Generator[str, None, None]:
        """Stream a response for user input."""
        self.add_user_message(user_input)
        
        request = LLMRequest(
            messages=self.messages.copy(),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        # Collect streamed response
        full_response = []
        for chunk in self.client.stream_complete(request):
            full_response.append(chunk)
            yield chunk
        
        # Add complete response to history
        self.add_assistant_message(''.join(full_response))
    
    def clear_history(self, keep_system: bool = True):
        """Clear conversation history."""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages.clear()
    
    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self.messages.copy()
    
    def export_history(self) -> str:
        """Export history as JSON."""
        return json.dumps([{
            'role': m.role,
            'content': m.content,
            'metadata': m.metadata
        } for m in self.messages], indent=2)


# Example usage
if __name__ == "__main__":
    print("=== LLM Client Interface Demo ===\n")
    
    # Create a mock client for testing
    client = LLMClientFactory.create(
        provider=LLMProvider.MOCK,
        model="mock-gpt-4",
        delay_ms=50
    )
    
    # Create conversation manager
    conversation = ConversationManager(
        client=client,
        system_prompt="You are a helpful AI assistant for task planning."
    )
    
    # Test regular completion
    print("--- Regular Completion ---")
    response = conversation.generate_response(
        "Help me plan a project to build a web scraper"
    )
    print(f"Response: {response.content}")
    print(f"Tokens: {response.tokens_used}, Latency: {response.latency_ms:.2f}ms\n")
    
    # Test streaming
    print("--- Streaming Completion ---")
    print("Response: ", end="", flush=True)
    for chunk in conversation.stream_response("What are the key steps?"):
        print(chunk, end="", flush=True)
    print("\n")
    
    # Show conversation history
    print("--- Conversation History ---")
    history = conversation.get_history()
    for i, msg in enumerate(history):
        print(f"{i+1}. [{msg.role}]: {msg.content[:60]}...")
    
    # Export history
    print("\n--- Export History (JSON) ---")
    print(conversation.export_history()[:200] + "...")
    
    # Demonstrate placeholder for real providers
    print("\n--- Real Provider Placeholder ---")
    try:
        real_client = LLMClientFactory.create(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="sk-placeholder"
        )
        real_client.complete(LLMRequest(messages=[Message("user", "test")]))
    except NotImplementedError as e:
        print(f"Expected error: {str(e)[:100]}...")
