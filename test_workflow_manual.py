"""
Manual Workflow Test
Run the workflow on a specific ticket ID to test end-to-end.

Usage:
    python test_workflow_manual.py 45                    # Run with ReACT agent (default)
    python test_workflow_manual.py 45 --mode react       # Explicitly use ReACT agent
    python test_workflow_manual.py 45 --mode sequential  # Use sequential workflow
    python test_workflow_manual.py --ticket-id 45
"""

import asyncio
import sys
import argparse
import logging
from datetime import datetime

# Import the cache initializer
from app.services.product_catalog_cache import init_product_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_react_iterations(state: dict):
    """Print detailed ReACT agent reasoning chain."""
    iterations = state.get("react_iterations", [])
    
    if not iterations:
        print("   No ReACT iterations recorded")
        return
    
    print(f"\n🧠 ReACT REASONING CHAIN ({len(iterations)} iterations):")
    print("-" * 60)
    
    for it in iterations:
        iter_num = it.get("iteration", "?")
        thought = it.get("thought", "")[:150]
        action = it.get("action", "unknown")
        observation = it.get("observation", "")[:200]
        duration = it.get("duration", 0)
        
        print(f"\n   ═══ Iteration {iter_num} ═══")
        print(f"   💭 Thought: {thought}...")
        print(f"   🔧 Action: {action}")
        print(f"   📤 Observation: {observation}...")
        print(f"   ⏱️  Duration: {duration:.2f}s")


def print_gathered_resources(state: dict):
    """Print what resources the ReACT agent gathered."""
    print("\n📚 GATHERED RESOURCES:")
    print("-" * 40)
    
    # Product info
    product = state.get("identified_product")
    if product:
        print(f"   ✅ Product: {product.get('model', 'N/A')} - {product.get('name', 'Unknown')}")
        print(f"      Category: {product.get('category', 'N/A')}")
        print(f"      Confidence: {product.get('confidence', 0):.0%}")
    else:
        print("   ❌ No product identified")
    
    # Documents
    docs = state.get("gathered_documents", [])
    print(f"\n   📄 Documents: {len(docs)} found")
    for i, doc in enumerate(docs[:3], 1):
        # Defensive: handle both dicts and strings
        if isinstance(doc, dict):
            print(f"      {i}. {doc.get('title', 'Unknown')}")
        elif isinstance(doc, str):
            print(f"      {i}. {doc}")
        else:
            print(f"      {i}. Unknown (invalid format)")
    
    # Images
    images = state.get("gathered_images", [])
    print(f"\n   🖼️  Images: {len(images)} found")
    
    # Past tickets
    tickets = state.get("gathered_past_tickets", [])
    print(f"\n   🎫 Past Tickets: {len(tickets)} similar")
    for i, ticket in enumerate(tickets[:3], 1):
        # Defensive: handle both dicts and strings
        if isinstance(ticket, dict):
            print(f"      {i}. #{ticket.get('ticket_id', 'N/A')}: {ticket.get('subject', 'No subject')[:40]}")
        elif isinstance(ticket, str):
            print(f"      {i}. {ticket}")
        else:
            print(f"      {i}. Unknown (invalid format)")


async def run_workflow_on_ticket(ticket_id: int, mode: str = "react"):
    """Run the full workflow on a specific ticket."""
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║       🌊 FLUSSO MANUAL WORKFLOW TEST                      ║
    ╠═══════════════════════════════════════════════════════════╣
    ║   Ticket ID: {str(ticket_id):<44}║
    ║   Mode:      {mode.upper():<44}║
    ║   Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<44}║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # 1. Initialize Product Cache (CRITICAL FOR CSV SEARCH)
    print("\n🚀 Initializing Service Cache...")
    try:
        init_product_cache()
        print("   ✅ Product Catalog Cache Initialized")
    except Exception as e:
        print(f"   ⚠️ Cache initialization warning: {e}")

    # Import based on mode
    if mode == "react":
        from app.graph.graph_builder_react import build_react_graph as build_graph
        mode_label = "ReACT Agent"
    else:
        from app.graph.graph_builder import build_graph
        mode_label = "Sequential"
    
    from app.graph.state import TicketState
    
    try:
        # Build the workflow graph
        print("\n📊 Building workflow graph...")
        workflow = build_graph()
        print(f"   ✅ {mode_label} graph built successfully")
        
        # Create initial state with ReACT fields
        initial_state: TicketState = {
            "ticket_id": str(ticket_id),
            "audit_events": [],
            # ReACT-specific fields
            "react_iterations": [],
            "react_total_iterations": 0,
            "react_status": "pending",
            "gathered_documents": [],
            "gathered_images": [],
            "gathered_past_tickets": [],
        }
        print(f"   ✅ Initial state created for ticket #{ticket_id}")
        
        # Run the workflow
        print("\n🚀 Executing workflow...\n")
        print("=" * 60)
        
        # Use sync invoke (graph.invoke) wrapped in thread for async compatibility
        import asyncio
        final_state = await asyncio.to_thread(workflow.invoke, initial_state)
        
        print("=" * 60)
        
        # Print basic ticket info
        print("\n📋 TICKET INFO:")
        print("-" * 40)
        print(f"   Ticket ID: {final_state.get('ticket_id', 'N/A')}")
        print(f"   Subject: {final_state.get('ticket_subject', 'N/A')}")
        print(f"   Category: {final_state.get('ticket_category', 'N/A')}")
        print(f"   Requester: {final_state.get('requester_email', 'N/A')}")
        
        # Print ReACT-specific info
        if mode == "react":
            react_status = final_state.get("react_status", "unknown")
            react_iterations = final_state.get("react_total_iterations", 0)
            
            print(f"\n🤖 ReACT AGENT STATUS:")
            print("-" * 40)
            print(f"   Status: {react_status}")
            print(f"   Iterations: {react_iterations}/15")
            
            # Print reasoning chain
            print_react_iterations(final_state)
            
            # Print gathered resources
            print_gathered_resources(final_state)
        
        # Print VIP info
        customer_type = final_state.get("customer_type", "regular")
        print(f"\n👤 Customer Type: {customer_type}")
        if customer_type == "vip":
            print("   ⭐ VIP customer - special handling applied")
        
        # Print confidence metrics
        print(f"\n📊 CONFIDENCE METRICS:")
        print("-" * 40)
        print(f"   Product Confidence: {final_state.get('product_match_confidence', 0):.0%}")
        print(f"   Hallucination Risk: {final_state.get('hallucination_risk', 0):.0%}")
        print(f"   Enough Information: {'Yes' if final_state.get('enough_information', False) else 'No'}")
        print(f"   VIP Compliant: {'Yes' if final_state.get('vip_compliant', True) else 'No'}")
            
        # Print draft response
        draft = final_state.get("draft_response") or final_state.get("generated_reply")
        if draft:
            print(f"\n📝 DRAFT RESPONSE:")
            print("-" * 40)
            # Truncate if too long
            if len(draft) > 500:
                draft = draft[:500] + "..."
            print(f"   {draft}")
        
        # Print resolution
        resolution = final_state.get("resolution_decision")
        if resolution:
            print(f"\n✅ RESOLUTION: {resolution}")
            reason = final_state.get("resolution_reason", "")
            if reason:
                print(f"   Reason: {reason}")
        
        # Print tags
        tags = final_state.get("suggested_tags", [])
        if tags:
            print(f"\n🏷️  TAGS: {', '.join(tags)}")
        
        # Print any errors
        if final_state.get("skip_workflow_applied"):
            print(f"\n⏭️  SKIPPED: {final_state.get('skip_reason', 'Unknown reason')}")
            
        print("\n" + "=" * 60)
        print(f"   Workflow completed at {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        print(f"\n❌ Workflow failed with error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run Flusso workflow on a specific Freshdesk ticket"
    )
    parser.add_argument(
        "ticket_id",
        nargs="?",
        type=int,
        help="Freshdesk ticket ID to process"
    )
    parser.add_argument(
        "--ticket-id", "-t",
        type=int,
        dest="ticket_id_flag",
        help="Freshdesk ticket ID to process"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["react", "sequential"],
        default="react",
        help="Workflow mode: 'react' (default) or 'sequential'"
    )
    
    args = parser.parse_args()
    
    # Get ticket ID from either positional or flag argument
    ticket_id = args.ticket_id or args.ticket_id_flag
    
    if not ticket_id:
        parser.print_help()
        print("\n❌ Error: Please provide a ticket ID")
        print("   Example: python test_workflow_manual.py 45")
        print("   Example: python test_workflow_manual.py 45 --mode react")
        print("   Example: python test_workflow_manual.py 45 --mode sequential")
        sys.exit(1)
    
    # Run the workflow
    try:
        asyncio.run(run_workflow_on_ticket(ticket_id, mode=args.mode))
    except KeyboardInterrupt:
        print("\n\n👋 Workflow cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()