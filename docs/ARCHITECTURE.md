# Architecture

## Core Components

- `deal_agent_framework.py`
  - Bootstraps logging, Chroma collection, planner, and persistent memory (`memory.json`).
  - Exposes `run()` used by the UI loop.

- `agents/planning_agent.py`
  - Coordinates scanner, ensemble pricing, and messaging.
  - Applies `DEAL_THRESHOLD` to determine notification.

- `agents/scanner_agent.py`
  - Scrapes RSS feeds through `ScrapedDeal.fetch()`.
  - Uses structured output (`DealSelection`) to normalize top deals.

- `agents/ensemble_agent.py`
  - Preprocesses item description.
  - Blends two estimators:
    - Frontier (RAG + LLM)
    - Specialist (fine-tuned Modal model)

- `agents/frontier_agent.py`
  - Embeds query text with SentenceTransformer.
  - Retrieves similar products from Chroma.
  - Calls LLM with RAG context for price estimation.

- `agents/specialist_agent.py`
  - Calls Modal class service `pricer-service.Pricer` remotely.

- `agents/messaging_agent.py`
  - Sends notifications via Pushover API.
  - Can optionally use Claude through LiteLLM to craft richer copy.

- `price_is_right.py`
  - Gradio UI.
  - Live logs, deal history table, and 3D vector plot.
  - Periodic refresh loop for autonomous scanning.

## Data Models

- `agents/deals.py` defines:
  - `Deal`
  - `DealSelection`
  - `Opportunity`
  - `ScrapedDeal` (RSS + HTML parsing helper)

## Deployment Topology

- Local process:
  - Scanner, planner, UI, memory, Chroma retrieval
- Remote process (Modal):
  - Specialist fine-tuned model inference

## Model Blend

Active ensemble blend in `agents/ensemble_agent.py`:
- Frontier estimator: 0.9
- Specialist estimator: 0.1
