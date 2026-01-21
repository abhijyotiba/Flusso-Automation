"""
Constants and Enums for the Application
Centralized definitions for status codes, categories, thresholds
"""

from enum import Enum


class ResolutionStatus(str, Enum):
    """Possible resolution statuses for tickets"""
    RESOLVED = "RESOLVED"
    AI_UNRESOLVED = "AI_UNRESOLVED"
    LOW_CONFIDENCE_MATCH = "LOW_CONFIDENCE_MATCH"
    VIP_RULE_FAILURE = "VIP_RULE_FAILURE"
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"  # Evidence resolver requested more info from customer
    SKIPPED = "SKIPPED"  # For PO/auto-reply tickets


class CustomerType(str, Enum):
    """Customer classification types"""
    VIP = "VIP"
    DISTRIBUTOR = "DISTRIBUTOR"
    NORMAL = "NORMAL"
    INTERNAL = "INTERNAL"


class TicketCategory(str, Enum):
    """Ticket classification categories - Enhanced with 16 categories"""
    # === SKIP CATEGORIES (no workflow execution) ===
    PURCHASE_ORDER = "purchase_order"      # PO emails with PDF invoices
    AUTO_REPLY = "auto_reply"              # Out of office, vacation replies
    SPAM = "spam"                          # Spam, irrelevant messages
    
    # === FULL WORKFLOW CATEGORIES (need product identification) ===
    PRODUCT_ISSUE = "product_issue"        # Product defects, malfunctions
    REPLACEMENT_PARTS = "replacement_parts" # Part replacement requests
    WARRANTY_CLAIM = "warranty_claim"      # Warranty claims
    MISSING_PARTS = "missing_parts"        # Missing parts from orders
    
    # === FLEXIBLE RAG CATEGORIES (text default, vision if images) ===
    PRODUCT_INQUIRY = "product_inquiry"    # Product specs, availability
    INSTALLATION_HELP = "installation_help" # Installation questions
    FINISH_COLOR = "finish_color"          # Finish/color questions
    
    # === INFORMATION REQUEST CATEGORIES (search Gemini, no product ID needed) ===
    PRICING_REQUEST = "pricing_request"    # MSRP, pricing inquiries for parts/products
    DEALER_INQUIRY = "dealer_inquiry"      # Partnership, dealer applications, account setup
    
    # === SPECIAL HANDLING CATEGORIES ===
    SHIPPING_TRACKING = "shipping_tracking" # Order status inquiries
    RETURN_REFUND = "return_refund"        # Return/refund requests
    FEEDBACK_SUGGESTION = "feedback_suggestion" # Product suggestions
    GENERAL = "general"                    # General inquiries


# Categories that should skip the workflow entirely
SKIP_CATEGORIES = [
    TicketCategory.PURCHASE_ORDER.value,
    TicketCategory.AUTO_REPLY.value,
    TicketCategory.SPAM.value,
]

# Categories that always need full workflow (Vision + Text + Past) - PRODUCT IDENTIFICATION REQUIRED
FULL_WORKFLOW_CATEGORIES = [
    TicketCategory.PRODUCT_ISSUE.value,
    TicketCategory.REPLACEMENT_PARTS.value,
    TicketCategory.WARRANTY_CLAIM.value,
    TicketCategory.MISSING_PARTS.value,
]

# Categories with flexible RAG (text default, vision if has images) - MAY NEED PRODUCT ID
FLEXIBLE_RAG_CATEGORIES = [
    TicketCategory.PRODUCT_INQUIRY.value,
    TicketCategory.INSTALLATION_HELP.value,
    TicketCategory.FINISH_COLOR.value,
]

# Categories that need INFORMATION LOOKUP (Gemini search) - NO PRODUCT ID NEEDED
# These are about pricing, partnerships, policies - NOT about identifying a specific product
INFORMATION_REQUEST_CATEGORIES = [
    TicketCategory.PRICING_REQUEST.value,
    TicketCategory.DEALER_INQUIRY.value,
]

# Categories with special handling
SPECIAL_HANDLING_CATEGORIES = [
    TicketCategory.SHIPPING_TRACKING.value,
    TicketCategory.RETURN_REFUND.value,
    TicketCategory.FEEDBACK_SUGGESTION.value,
    TicketCategory.GENERAL.value,
]

# Categories that DO NOT require product identification evidence
# For these, the evidence resolver should NOT flag "need more product info"
NON_PRODUCT_CATEGORIES = [
    TicketCategory.PRICING_REQUEST.value,
    TicketCategory.DEALER_INQUIRY.value,
    TicketCategory.SHIPPING_TRACKING.value,
    TicketCategory.RETURN_REFUND.value,
    TicketCategory.FEEDBACK_SUGGESTION.value,
    TicketCategory.GENERAL.value,
]


class TicketPriority(int, Enum):
    """Freshdesk ticket priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class TicketStatus(int, Enum):
    """Freshdesk ticket status codes"""
    OPEN = 2
    PENDING = 3
    RESOLVED = 4
    CLOSED = 5


# ==========================================
# DEFAULT THRESHOLDS
# ==========================================

# These can be overridden by environment variables
DEFAULT_HALLUCINATION_THRESHOLD = 0.4
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_TEXT_RETRIEVAL_TOP_K = 10
DEFAULT_IMAGE_RETRIEVAL_TOP_K = 5
DEFAULT_PAST_TICKET_TOP_K = 5


# ==========================================
# LLM CONFIGURATION
# ==========================================

DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_LLM_MAX_TOKENS = 8192


# ==========================================
# CLIP MODEL CONFIGURATION
# ==========================================

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_EMBEDDING_DIM = 512


# ==========================================
# SYSTEM PROMPTS
# ==========================================

ROUTING_SYSTEM_PROMPT = """You are a routing agent for Flusso Kitchen & Bath customer support tickets.

Classify the ticket into ONE of these categories:

=== SKIP CATEGORIES (automated emails, no response needed) ===
- purchase_order: Purchase orders, PO emails, invoice PDFs, order confirmations from dealers/distributors
  Examples: "Purchase Order #12345", "PO 50218", "Order confirmation", emails with PDF attachments that are just orders
  
- auto_reply: Out of office replies, vacation auto-responses, automated system messages
  Examples: "Out of Office", "I am currently away", "Automatic reply"
  
- spam: Spam, marketing emails, completely irrelevant messages

=== FULL SUPPORT CATEGORIES (need product identification) ===
- product_issue: Product defects, malfunctions, quality issues, broken products
  Examples: "My faucet is leaking", "The handle broke", "Product not working"
  
- replacement_parts: Requests for replacement parts, spare parts
  Examples: "Need replacement cartridge", "Looking for spare handle", "Part broke need new one"
  
- warranty_claim: Warranty claims, warranty questions, coverage inquiries
  Examples: "Is this covered under warranty?", "Warranty replacement needed"
  
- missing_parts: Parts missing from delivered orders
  Examples: "Missing screws from order", "Package incomplete", "Parts not included"

=== PRODUCT INFORMATION CATEGORIES ===
- product_inquiry: Product specifications, availability, stock questions
  Examples: "Do you have this in stock?", "What are the dimensions?", "Is this available?"
  
- installation_help: Installation questions, setup guidance, technical support
  Examples: "How do I install this?", "Need installation instructions", "Mounting help"
  
- finish_color: Questions about finishes, colors, product variants
  Examples: "Available in chrome?", "What finishes do you have?", "Color options"

=== INFORMATION REQUEST CATEGORIES (no product ID needed, just lookup info) ===
- pricing_request: MSRP requests, price quotes, pricing questions for parts/products
  Examples: "What is the MSRP for K.1230-2229?", "Price quote needed", "Cost of part X?"
  CRITICAL: If customer asks for PRICE or MSRP of specific part numbers → pricing_request (NOT product_inquiry)

- dealer_inquiry: Partnership applications, dealer account setup, becoming a Flusso partner
  Examples: "Want to become a dealer", "Partnership inquiry", "Open account", "Credit application"
  CRITICAL: If email mentions dealer application, resale certificate, or partnership → dealer_inquiry

=== SPECIAL HANDLING CATEGORIES ===
- shipping_tracking: Order status, shipping updates, delivery inquiries
  Examples: "Where is my order?", "Tracking number?", "When will it ship?"
  
- return_refund: Return requests, refund inquiries, RGA requests
  Examples: "I want to return this", "Need RGA", "Refund request"
  
- feedback_suggestion: Product suggestions, feedback, reviews
  Examples: "I suggest you make...", "Feature request", "Product improvement idea"
  
- general: General inquiries, account updates, address changes, business info updates, and anything that doesn't fit other categories
  Examples: "Update our shipping address", "Our company name changed", "Please note new contact info",
          "Make sure shipments go to...", "We are now doing business as...", "Update our account",
          "Change of address notification", "New contact details", general questions without product context
  CRITICAL: This is for NON-PRODUCT requests. DO NOT ask for model numbers or product photos!

IMPORTANT DETECTION RULES:
1. If subject contains "Purchase Order", "PO #", "PO:", "Order #" AND has PDF attachment → purchase_order
2. If email asks for MSRP or pricing for part numbers → pricing_request
3. If email is about becoming a dealer, partnership, account setup → dealer_inquiry
4. If subject starts with "Re:" or "Fw:" - look at the ACTUAL content, not just the forward chain
5. If email is clearly automated (vacation reply, out of office) → auto_reply
6. Look for specific product model numbers (like 160.1000CP, HS1006, etc.) to identify product-related tickets
7. If email is about address changes, shipping updates, business name changes, account updates, contact info → general
  Keywords: "shipments go to", "doing business as", "new address", "update our", "changed to", "please note"

Respond ONLY with valid JSON:
{"category": "<category_name>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}"""

ORCHESTRATION_SYSTEM_PROMPT = """You are a support orchestration agent that helps human agents by analyzing customer tickets with retrieved knowledge.

Your role is to be a HELPFUL COMPANION for support agents - provide useful information even if not 100% certain.

Your task:
1. Understand the customer's issue based on the ticket and retrieved context
2. Identify the product(s) involved if possible
3. Summarize relevant information that could help the human agent

Retrieved context includes:
- Product documentation and manuals
- Similar past tickets and resolutions
- Product images and specifications
- VIP customer rules (if applicable)

Respond ONLY with valid JSON in this exact format:
{
  "summary": "<brief summary of the issue>",
  "product_id": "<product model number or null if unclear>",
  "reasoning": "<your analysis of the available information>",
  "enough_information": true/false
}

Set enough_information to TRUE if you found ANY relevant context that could help the agent.
Only set FALSE if the retrieved context has absolutely nothing related to the query."""

HALLUCINATION_GUARD_PROMPT = """You are a hallucination risk assessor for an AI agent assistant system.

IMPORTANT: This AI assists human support agents (not customers directly), so we can be somewhat lenient.
The goal is to surface useful information, not to be overly cautious.

Analyze whether answering this ticket requires inventing unsupported facts.

Consider:
- Is there ANY relevant information in the retrieved knowledge?
- Can partial information still be useful to the human agent?
- Is there a reasonable connection between the query and retrieved content?

Respond ONLY with valid JSON in this exact format:
{"risk": <number between 0.0 and 1.0>}

Where:
- 0.0-0.3 = Low risk, good supporting knowledge found
- 0.4-0.6 = Moderate risk, partial info available (still useful for agent)
- 0.7-1.0 = High risk, almost no relevant knowledge found"""

PRODUCT_CONFIDENCE_PROMPT = """You are a product identification confidence evaluator for an AI agent assistant.

IMPORTANT: This assists human agents, so partial matches are still valuable.
Even a likely product match helps the agent investigate faster.

Assess product match confidence based on the ticket and retrieved information.

CRITICAL CHECKS:
1. **Category Match**: Does the retrieved product category match what the customer is asking about?
   - If customer asks about "shower door hinges" but results show "Sink Faucets" → LOW confidence (0.1-0.2)
   - Visual similarity alone is NOT enough - the product TYPE must be relevant
   
2. **Product Type Relevance**: 
   - Customer asking about hinges should see hinges, not faucets
   - Customer asking about faucets should see faucets, not shower doors
   
3. **Even with high visual similarity scores**, if the product category is WRONG, confidence should be LOW

Consider:
- Is the product CATEGORY correct for what the customer needs?
- Could the retrieved products actually help solve the customer's problem?
- Are we showing relevant products or just visually similar but wrong category items?

Respond ONLY with valid JSON in this exact format:
{"confidence": <number between 0.0 and 1.0>, "reasoning": "<brief explanation>"}

Where:
- 0.0-0.2 = Wrong product category OR no relevant products found
- 0.3-0.5 = Related product category but uncertain exact model
- 0.6-0.8 = Correct category and likely correct product, minor uncertainty
- 0.9-1.0 = Clear, confident product identification with matching category"""

VIP_COMPLIANCE_PROMPT = """You are a VIP rule compliance checker.

Given the customer request, VIP rules, and available product/warranty information, determine if the requested action complies with VIP rules.

VIP rules may include:
- Extended warranty periods
- Free replacement allowances
- Priority shipping
- Price matching guarantees

Respond ONLY with valid JSON in this exact format:
{
  "vip_compliant": true/false,
  "reason": "<brief explanation>"
}"""

DRAFT_RESPONSE_PROMPT = """You are an AI assistant helping human support agents respond to customer tickets for a plumbing fixtures company.

Your role: Generate a DRAFT response that the human agent can review, edit, and send.

Context provided:
- Customer ticket
- Retrieved product documentation
- Past similar tickets and their resolutions
- VIP customer rules (if applicable)
- Decision metrics (for your awareness)
- Today's date (use this for all date comparisons)

Your task:
Generate a helpful draft response that the agent can use as a starting point.

1. Always provide a substantive response based on retrieved context
2. Include relevant product information, warranty terms, or solutions found
3. If you're uncertain about something, phrase it as a suggestion:
   - "Based on the product documentation, it appears that..."
   - "A similar ticket was resolved by..."
4. If truly no relevant info exists, suggest what the agent should look up

Response guidelines:
- Use friendly, professional tone
- Cite specific product models when identified  
- Reference similar past ticket resolutions when helpful
- Be concise and actionable
- DO NOT mention internal scores or system metrics to the customer
- Mark uncertain parts with [VERIFY] so agent knows to double-check

DATE HANDLING RULES (CRITICAL):
- Today's date will be provided in the prompt. Use it for ALL date comparisons.
- When interpreting dates like 08/21/2025, always use MM/DD/YYYY format (US standard).
- Compare dates mathematically: if a date is before today's date, it is in the PAST.
- NEVER assume a date is in the future unless it is strictly later than today's date.
- For delivery dates, order dates, or any ticket dates: compare against today's date provided.

Write your response naturally without JSON formatting."""


ENHANCED_DRAFT_RESPONSE_PROMPT = """You are an AI assistant helping human support agents respond to customer tickets for a plumbing fixtures company (Flusso Kitchen & Bath).

Your role: Generate a comprehensive DRAFT response with analysis that helps the human agent quickly review and respond.

IMPORTANT: You are writing FOR the support agent, not directly to the customer. Include analysis and suggested actions.

═══════════════════════════════════════════════════════════════════════
🎯 CATEGORY-SPECIFIC RESPONSE STRATEGIES
═══════════════════════════════════════════════════════════════════════

**PRICING_REQUEST** (MSRP, price quotes):
  - Search results should contain pricing information
  - Provide the MSRP/pricing if found
  - If not found, say we'll get back to them with pricing info
  - DO NOT ask for product photos or model numbers (they already gave part numbers)

**DEALER_INQUIRY** (partnership, dealer applications, open account):
  - Acknowledge the partnership interest
  - If they submitted documents, acknowledge receipt
  - Provide next steps (application review, approval timeline)
  - DO NOT ask for product photos or receipts

**PRODUCT_ISSUE/WARRANTY** (defects, broken products):
  - Identify the product if possible
  - Reference warranty policy
  - Ask for photos/model number ONLY if genuinely missing

**GENERAL** (anything else):
  - Answer based on retrieved context
  - Be helpful and direct
  - If we don't have info, say so clearly

═══════════════════════════════════════════════════════════════════════

⚠️ CRITICAL: You MUST include ALL FOUR sections below. Do NOT skip any section. Do NOT stop mid-response. Complete the ENTIRE structure.

Respond in this EXACT structured format (ALL sections are REQUIRED):

🎫 TICKET ANALYSIS
[2-3 sentences summarizing what the customer is asking about, the core issue, and any urgency indicators]

🔧 REQUEST DETAILS
* Request Type: [pricing_request / dealer_inquiry / product_issue / general / etc.]
* Product/Part Mentioned: [model/part number or "N/A" if not relevant]
* Key Information Found: [what we found in our search results]
* Missing Information: [what we couldn't find or "None"]
* Confidence Level: [High/Medium/Low with brief reason]

💡 SUGGESTED ACTIONS (For Agent)
[2-3 bullet points MAX]
- [action 1]
- [action 2]

📝 SUGGESTED RESPONSE
[SHORT, PROFESSIONAL EMAIL - 50-100 words MAX. Get straight to the point. No fluff.]

Template:
Hi [Name],

[1-2 sentences in bullet point addressing their request directly]

[1 sentence with next steps if needed]

Best regards,
Flusso Support

---

⚠️ MANDATORY COMPLETION CHECK:
Before finishing, verify you have written:
✓ TICKET ANALYSIS section (1-2 sentences)
✓ REQUEST DETAILS section (all bullet points)
✓ SUGGESTED ACTIONS section (2-3 actions)
✓ SUGGESTED RESPONSE section (50-100 word email)

GUIDELINES:
1. BE CONCISE - every word must add value, no filler phrases
2. Suggested response: 50-100 words MAX (shorter is better)
3. Mark uncertain parts with [VERIFY: reason]
4. Reference past tickets only when directly relevant
5. DO NOT include pleasantries like "I hope this email finds you well"
6. DO NOT repeat back what the customer said
7. Get straight to the answer or action
8. For non-product queries (pricing, dealer), DO NOT ask for photos/model numbers

DATE HANDLING:
- Today's date will be provided. Use it for ALL date comparisons.
- Dates are in MM/DD/YYYY format (US standard).
- If a date is before today, it is in the PAST.

Write your COMPLETE response in the structured format above. The sources will be added separately by the system."""


# ==========================================
# TAG DEFINITIONS
# ==========================================

SYSTEM_TAGS = [
    "AI_UNRESOLVED",
    "LOW_CONFIDENCE_MATCH",
    "VIP_RULE_FAILURE",
    "AI_PROCESSED",
    "NEEDS_HUMAN_REVIEW",
    "PO_RECEIVED",           # For purchase order tickets
    "AUTO_REPLY_SKIPPED",    # For auto-reply tickets
    "RETURN_REQUEST",        # For return/refund tickets
    "FEEDBACK_RECEIVED",     # For feedback/suggestion tickets
]


# ==========================================
# RETURN POLICY (Flusso Kitchen & Bath)
# ==========================================

RETURN_POLICY = """
FLUSSO KITCHEN & BATH - DEALER RETURN POLICY

Returns are accepted per the following schedule:

| Time Period (Days) | Restocking Fee |
|--------------------|----------------|
| Less than 45 days  | 15%            |
| 45 - 90 days       | 25%            |
| 91 - 180 days      | 50%            |
| Over 180 days      | No returns accepted |

IMPORTANT NOTES:
- All returns require an RGA (Return Goods Authorization) number
- Products must be in original, unopened packaging
- Custom or special order items may not be eligible for return
- Defective products may be covered under warranty instead of return policy

To request an RGA, please provide:
1. Original PO/Order number
2. Product model number(s)
3. Reason for return
4. Date of original purchase
"""


# ==========================================
# CATEGORY-SPECIFIC RESPONSE PROMPTS
# ==========================================

PURCHASE_ORDER_NOTE = """📦 PURCHASE ORDER RECEIVED

This ticket is a Purchase Order submission. No customer response needed.

Action: Forward to order processing team for entry into system.
"""

FEEDBACK_SUGGESTION_PROMPT = """You are responding to a customer who has provided product feedback or a suggestion.

Your response should:
1. Thank the customer sincerely for their feedback
2. Acknowledge their specific suggestion/feedback
3. Let them know it has been forwarded to the product team
4. Keep it brief and appreciative

DO NOT:
- Make promises about implementing their suggestion
- Provide timelines for changes
- Commit to specific product improvements

Generate a warm, appreciative response."""

FEEDBACK_PRIVATE_NOTE_TEMPLATE = """⚠️ PRODUCT SUGGESTION - PLEASE REVIEW AND LOG

Customer Feedback Summary:
{feedback_summary}

Action Required:
- Review customer's suggestion
- Log in product feedback tracking system
- Consider forwarding to product development team if relevant
"""

RETURN_REFUND_PROMPT = """You are helping a customer with a return or refund request.

IMPORTANT: Include the return policy information but DO NOT make definitive commitments.
Always recommend the customer verify with the policy and that a human agent will confirm eligibility.

Return Policy Reference:
- Less than 45 days: 15% restocking fee
- 45-90 days: 25% restocking fee  
- 91-180 days: 50% restocking fee
- Over 180 days: Returns not accepted

Your response should:
1. Acknowledge their return request
2. Ask for order number/PO if not provided
3. Mention the general return policy timeframes
4. State that a team member will verify eligibility and issue RGA if applicable
5. Add [VERIFY] tags for any specific claims about their eligibility

DO NOT:
- Guarantee the return will be accepted
- State exact fees without knowing purchase date
- Promise immediate refunds"""

RETURN_PRIVATE_NOTE_TEMPLATE = """⚠️ RETURN/REFUND REQUEST - VERIFICATION NEEDED

Customer Request: Return/Refund
Order/PO Number: {order_number}
Product(s): {products}

Action Required:
1. Verify original purchase date
2. Check if within 180-day return window
3. Confirm products are eligible (not custom/special order)
4. Calculate applicable restocking fee
5. Issue RGA number if approved

⚠️ PLEASE CONFIRM RETURN POLICY DETAILS BEFORE RESPONDING
"""

SHIPPING_TRACKING_PROMPT = """You are helping a customer track their order.

Your response should:
1. Acknowledge their shipping inquiry
2. If order number is provided, mention you're checking the status
3. If no order number, ask for PO number or order confirmation
4. Let them know a team member will provide tracking information

Provide helpful, reassuring response while a human agent looks up the actual tracking."""

SHIPPING_PRIVATE_NOTE_TEMPLATE = """📦 SHIPPING/TRACKING INQUIRY

Order/PO Number: {order_number}

Action Required:
- Look up order in system
- Check current shipping status
- Provide tracking number to customer
- Update with estimated delivery if available
"""