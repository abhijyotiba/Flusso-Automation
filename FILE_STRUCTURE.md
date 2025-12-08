# Flusso Workflow Project File Structure

This document describes the key files and folders required to run and understand the Flusso workflow implementation, including the ReAct agentic flow.

---

## Top-Level Files
- `main_react.py`         # Entry point for ReAct workflow (if present)
- `main.py`              # Main FastAPI app entry point
- `run_local_server.py`  # Local server runner
- `test_workflow_manual.py` # Manual workflow test script
- `requirements.txt`     # Python dependencies
- `Procfile`, `render.yaml`, `Dockerfile` # Deployment configs
- `README.md`            # General project documentation
- `react_readme.md`      # ReAct workflow documentation
- `PROJECT_CONTEXT.md`   # Project context and architecture

---

## app/
- `app/__init__.py`
- `app/main.py`                # FastAPI app or workflow orchestrator
- `app/graph/`                 # Workflow graph and agent logic
    - `graph_builder.py`           # Sequential workflow graph
    - `graph_builder_react.py`     # ReAct workflow graph
    - `state.py`                   # State model
    - `react_agent.py`             # ReAct agent logic
    - `react_agent_helpers.py`     # ReAct agent helpers
    - Other graph nodes
- `app/nodes/`                 # Workflow nodes
    - `fetch_ticket.py`             # Freshdesk fetch node
    - `freshdesk_update.py`         # Freshdesk update node
    - `draft_response.py`           # Final output formatting
    - Other nodes (customer_lookup, orchestration_agent, etc.)
    - `decisions/`                  # Decision logic nodes
    - `response/`                   # Response formatting nodes
- `app/tools/`                 # Tool definitions
    - `product_search.py`
    - `document_search.py`
    - `vision_search.py`
    - `past_tickets.py`
    - `attachment_analyzer.py`
    - `finish.py`
- `app/clients/`               # API/service clients
    - `freshdesk_client.py`
    - `gemini_client.py`
    - `llm_client.py`
    - `pinecone_client.py`
- `app/config/`                # Configuration
    - `constants.py`
    - `settings.py`
- `app/utils/`                 # Utilities
    - `detailed_logger.py`
    - `validation.py`
    - `retry.py`
    - `audit.py`
    - `attachment_processor.py`
- `app/services/`              # Service layer (if used)

---

## Reference & Logs
- `reference code for retrivals agents/` # Reference implementations (optional)
- `workflow_logs/`                      # Workflow logs (optional)
- `__pycache__/`                        # Python cache (ignore)

---

## Summary
- All main logic is in the `app/` folder and its subfolders.
- Entry points: `main_react.py`, `main.py`, `test_workflow_manual.py`.
- Tools, nodes, and agents are modularized for clarity and extensibility.
- Deployment and documentation files are at the root level.

---

> For a minimal runnable version, copy the entire `app/` folder, all entry points, `requirements.txt`, and relevant config files.
