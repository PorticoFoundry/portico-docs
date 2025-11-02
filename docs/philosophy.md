# Philosophy

Portico is built on the principles of hexagonal architecture (also known as [ports](ports/index.md) and [adapters](adapters/index.md)). This isn't just academic theory - it's a practical approach to building applications that stay maintainable as they grow from prototype to production.

## Core Principles

### Separation of Concerns

Your business logic should be independent of implementation details. Whether you're using OpenAI or Anthropic, PostgreSQL or SQLite, Redis or in-memory caching - these are deployment decisions, not architecture decisions. Your core application logic shouldn't change when you swap out infrastructure.

### Dependency Inversion

[Kits](kits/index.md) (your business logic) depend on ports (interfaces), never on adapters (implementations). This inversion means you can test your logic without external dependencies, swap implementations without code changes, and keep your architecture clean as complexity grows.

### Composition Root

There's exactly one place in your codebase where concrete implementations are chosen: the `compose` module. This forces architectural discipline and makes it obvious where dependencies are wired together. If a kit imports an adapter directly, the build fails. Architecture is enforced by tooling, not just documentation.

## Why This Approach Wins

### 1. Swap implementations instantly

Need to switch from OpenAI to Anthropic? Change one line in your config. Want to test with SQLite locally but use PostgreSQL in production? Same code, different adapter.

```python
# Development
config = {
    "llm": compose.llm(provider="openai", api_key="..."),
    "database": compose.database(adapter="sqlite", path="./dev.db")
}

# Production
config = {
    "llm": compose.llm(provider="anthropic", api_key="..."),
    "database": compose.database(adapter="postgres", url="...")
}
```

### 2. Test your logic without external dependencies

Since kits only depend on ports, you can test your business logic with fake implementations:

```python
class FakeLLMProvider(LLMProvider):
    async def complete(self, prompt: str) -> str:
        return "Fake response for testing"

# Test your business logic without calling real APIs
kit = LLMKit(FakeLLMProvider())
result = await kit.generate("test prompt")
assert result == "Fake response for testing"
```

### 3. Your codebase stays clean

Without architecture boundaries, dependencies creep everywhere. Three months later, your auth code is importing database models, your API routes are calling external services directly, and changing anything breaks everything.

Portico enforces clean boundaries:

- **Kits** → can only import Ports
- **Ports** → can't import anything except standard library
- **[Adapters](adapters/index.md)** → implement Ports, can import external libraries
- **Compose** → the only place that wires Kits + Adapters together

These rules are checked on every build. Your architecture stays clean automatically.

### 4. Start simple, scale when ready

Begin with in-memory caching and SQLite. When you need to scale, swap in Redis and PostgreSQL. Same business logic, zero rewrites.

```python
# Week 1 - MVP
compose.cache(adapter="memory")  # Good enough

# Month 3 - Growth
compose.cache(adapter="redis", url="redis://...")  # Same interface, better performance
```

## What This Means in Practice

### For Prototyping

Start with the simplest possible implementations. In-memory caching, SQLite databases, fake LLM providers for testing. Focus on your business logic and user experience, not infrastructure decisions.

### For Testing

Test your business logic in isolation. No need to spin up databases, mock external APIs, or configure complex test fixtures. Your kits work with any implementation of their ports.

### For Production

When you're ready to scale, swap in production-grade implementations. Redis for caching, PostgreSQL for persistence, real LLM providers. Your business logic doesn't change - only your composition root configuration.

### For Maintenance

Six months from now, when you need to add a new feature or fix a bug, you'll find code that's organized by concern, not by technology. Auth logic lives in AuthKit, not scattered across route handlers, database models, and middleware.

## The Trade-offs

Hexagonal architecture isn't free. You write more files (ports, adapters, kits instead of just "code"). You have a level of indirection between your business logic and external services. You need to think about interfaces upfront.

But here's the thing: **if you're using Portico, we've already done this work for you**. The ports are defined. The adapters are implemented. The kits are built and tested. You get all the benefits of hexagonal architecture without having to design and build it yourself.

The payoff is huge:

- **Flexibility** - Change implementations without touching business logic
- **Testability** - Test without external dependencies
- **Clarity** - Clear boundaries between concerns
- **Maintainability** - Architecture that scales with complexity

Portico makes this trade-off for you: we invest in the upfront structure so you get long-term maintainability out of the box.

---

Ready to see this in action? Check out the [demos](demos.md) or dive into the [documentation](index.md).
