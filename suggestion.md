Summary Framework for Solving Tickets:
No redundant asks: don’t ask for photo/finish/PO/address if ticket says present


Policy gate: if category touches warranty/missing parts → must cite policy record(s) and apply dates/windows


Parts gate: if part numbers present → must be recognized via tool; if image confidence low → ask clarifying instead of asserting


Completeness gate: if replacement suggested → must include address confirmation and proof-of-purchase logic


Case gate: if multiple tickets in same case → include them in context and avoid duplicative notes
Details:
Adding a structured “Ticket Intake Extractor” to deterministically extract whether the ticket already contains: model, finish, PO#, receipt, address, video, photos, part numbers.
Fixes: “asks for photo/address/finish/model when already present” + more consistent “always ask for X if missing”.
Create a ticket_facts record per ticket:
has_address, has_receipt, has_po, has_video, has_photos


product_model_candidates[]


part_numbers[] (regex-extracted)


finish_candidates[]


customer_name, requester_email (from Freshdesk metadata, not LLM guesses)

Building a “Required Fields Matrix” by ticket type + validator for each category, enforcing required info and only asking for missing items.
Fixes: 97840 template requirement, 97625/97493 address gaps, 97764 receipt gaps, and prevents unnecessary questions.
Example categories (based on your feedback):
Warranty claim (needs receipt/PO + address; may need video)


Missing parts (needs 45-day check + parts diagram + address)


Product question/compatibility (needs exact model/part numbers; if unknown ask receipt)


Replacement request (needs warranty check + address + proof of purchase)
Backend validator produces a missing_fields[] list from ticket_facts and forces the response to include only those asks.

Separating “Policy Engine” from the LLM (policy is not optional)
Applying warranty/missing-window rules as constraints.
Fixes: 97588 / 97854 / 97853 / 97988 policy misses and “assumed covered indefinitely”.
Store policy docs in a dedicated table (or KB index), with metadata:


policy_type (hose_warranty, missing_parts_window, lifetime_warranty, etc.)


effective_date, product_family, coverage_months, exceptions


Retrieval returns the specific policy snippet + ID.


Adding a validator:


If ticket involves hoses → must reference hose coverage months + compare to receipt date if present.


If missing parts → must mention the 45-day window.
