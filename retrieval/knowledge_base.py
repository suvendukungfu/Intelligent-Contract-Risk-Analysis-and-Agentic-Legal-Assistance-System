"""
retrieval/knowledge_base.py
----------------------------
A curated corpus of 35 legal best-practice entries covering the most
common risky contract clauses. This library is embedded into ChromaDB
and queried via RAG during the agent pipeline.
"""

LEGAL_KNOWLEDGE_BASE = [
    # ── INDEMNIFICATION ───────────────────────────────────────────────────
    {
        "id": "ind_001",
        "topic": "Indemnification",
        "content": (
            "An indemnification clause requires one party to compensate "
            "the other for losses arising from specified events. Broad indemnification "
            "provisions, such as those covering 'any and all claims,' can expose a party "
            "to unlimited financial liability. Best practice: negotiate caps, carve-outs "
            "for gross negligence, and mutual indemnification where possible."
        )
    },
    {
        "id": "ind_002",
        "topic": "Indemnification",
        "content": (
            "One-sided indemnification (only one party must indemnify) is a significant "
            "red flag. Under US contract law, courts increasingly scrutinize such clauses "
            "for unconscionability. Parties should seek mutual indemnification or, at minimum, "
            "a carve-out for the indemnifying party's own negligence."
        )
    },

    # ── LIABILITY LIMITATION ──────────────────────────────────────────────
    {
        "id": "liab_001",
        "topic": "Limitation of Liability",
        "content": (
            "Limitation of liability clauses cap the damages one party can recover from "
            "another. Typical caps range from 12 months of fees paid to a fixed dollar amount. "
            "Watch for: caps that are too low relative to potential damages, exclusions of "
            "consequential damages that eliminate meaningful recovery, and asymmetric caps "
            "that only protect one party."
        )
    },
    {
        "id": "liab_002",
        "topic": "Limitation of Liability",
        "content": (
            "Exclusions of consequential and indirect damages (lost profits, loss of data) "
            "are standard in technology contracts but can be devastating if a data breach "
            "or service failure causes massive downstream losses. Best practice: negotiate "
            "carve-outs for breaches of confidentiality and IP indemnification."
        )
    },

    # ── TERMINATION ───────────────────────────────────────────────────────
    {
        "id": "term_001",
        "topic": "Termination",
        "content": (
            "Termination for convenience clauses allow a party to end a contract without "
            "cause. This benefits the terminating party but can leave the other with sunk "
            "costs. Risk: if no notice period or wind-down compensation is included, "
            "the non-terminating party has little protection. Best practice: require "
            "30–90 days notice and compensation for work in progress."
        )
    },
    {
        "id": "term_002",
        "topic": "Termination",
        "content": (
            "Termination for cause clauses require a material breach as the trigger. "
            "Vague definitions of 'material breach' create disputes. Best practice: "
            "enumerate specific events that constitute a material breach and provide "
            "a cure period (typically 30 days) before termination is effective."
        )
    },

    # ── CONFIDENTIALITY / NDA ─────────────────────────────────────────────
    {
        "id": "nda_001",
        "topic": "Confidentiality",
        "content": (
            "Confidentiality clauses must define 'Confidential Information' precisely. "
            "Overly broad definitions (e.g., 'all information shared between parties') "
            "can be unenforceable. Best practice: distinguish between oral and written "
            "disclosures, require marking of confidential materials, and set a reasonable "
            "duration (3–5 years is standard; perpetual obligations are disfavored)."
        )
    },
    {
        "id": "nda_002",
        "topic": "Confidentiality",
        "content": (
            "Exceptions to confidentiality obligations are standard and typically include: "
            "information already publicly known, information independently developed, "
            "information received from third parties without restriction, and information "
            "required to be disclosed by law or court order. Missing these exceptions "
            "creates impractically broad obligations."
        )
    },

    # ── GOVERNING LAW & JURISDICTION ──────────────────────────────────────
    {
        "id": "gov_001",
        "topic": "Governing Law",
        "content": (
            "Governing law clauses determine which jurisdiction's laws apply to disputes. "
            "Delaware is favored in US commercial contracts due to its sophisticated "
            "Court of Chancery and well-developed corporate law. International contracts "
            "often specify New York law or English law. Risk: choosing a jurisdiction "
            "unfamiliar to one party creates hidden legal cost and disadvantages."
        )
    },
    {
        "id": "gov_002",
        "topic": "Dispute Resolution",
        "content": (
            "Arbitration clauses waive the right to a jury trial and class action "
            "participation. While arbitration is typically faster and cheaper for B2B "
            "disputes, mandatory arbitration with venue in the other party's city "
            "creates geographic and cost disparities. Best practice: specify AAA or "
            "JAMS rules, neutral venue, and each party bears its own costs."
        )
    },

    # ── INTELLECTUAL PROPERTY ─────────────────────────────────────────────
    {
        "id": "ip_001",
        "topic": "Intellectual Property",
        "content": (
            "Work-for-hire clauses transfer IP ownership to the engaging party. This is "
            "standard for commissioned software and creative work, but contractors must "
            "carve out pre-existing IP. Without a clear carve-out, contractors risk "
            "inadvertently transferring their background technology. Best practice: "
            "list pre-existing IP in an exhibit to the contract."
        )
    },
    {
        "id": "ip_002",
        "topic": "Intellectual Property",
        "content": (
            "License-back provisions allow the original IP owner to use IP they have "
            "transferred or assigned. These are common in software development agreements. "
            "Watch for: irrevocable, perpetual, royalty-free licenses that effectively "
            "nullify the value of the IP assignment."
        )
    },

    # ── FORCE MAJEURE ─────────────────────────────────────────────────────
    {
        "id": "fm_001",
        "topic": "Force Majeure",
        "content": (
            "Force majeure clauses excuse non-performance due to events beyond a party's "
            "control. Post-COVID, courts scrutinized whether pandemics and supply chain "
            "disruptions qualify. Best practice: explicitly list cyberattacks, pandemics, "
            "and government actions alongside traditional events (acts of God, natural "
            "disasters). Include a maximum suspension period after which either party "
            "can terminate."
        )
    },
    {
        "id": "fm_002",
        "topic": "Force Majeure",
        "content": (
            "A force majeure clause that does not include notice requirements can be "
            "abused. Best practice: require prompt written notice (within 48–72 hours) "
            "of the event invoking force majeure, ongoing updates, and mitigation efforts "
            "to minimize the impact on the other party."
        )
    },

    # ── PAYMENT TERMS ─────────────────────────────────────────────────────
    {
        "id": "pay_001",
        "topic": "Payment Terms",
        "content": (
            "Payment terms should specify: amount, due date, late payment interest (typically "
            "1.5% per month or the statutory rate), invoicing process, and accepted payment "
            "methods. Ambiguous payment terms are a leading cause of commercial disputes. "
            "Net-60 or longer terms without interest provisions significantly disadvantage "
            "vendors and small businesses."
        )
    },
    {
        "id": "pay_002",
        "topic": "Payment Terms",
        "content": (
            "Milestone-based payment structures tie payment to deliverable acceptance. "
            "Risk: acceptance criteria must be objective and measurable to avoid disputes. "
            "Vague criteria such as 'client satisfaction' give the paying party broad "
            "discretion to withhold payment. Best practice: define acceptance criteria "
            "as specific, measurable, and time-bound tests."
        )
    },

    # ── NON-COMPETE / NON-SOLICITATION ────────────────────────────────────
    {
        "id": "noncomp_001",
        "topic": "Non-Compete",
        "content": (
            "Non-compete clauses restrict an employee or contractor from working with "
            "competitors after the relationship ends. Enforceability varies dramatically "
            "by state: California, North Dakota, and Oklahoma ban them outright. "
            "Best practice: ensure the clause is limited in geographic scope, duration "
            "(12–24 months maximum), and role specificity. Overly broad non-competes "
            "are routinely struck down by courts."
        )
    },
    {
        "id": "noncomp_002",
        "topic": "Non-Solicitation",
        "content": (
            "Non-solicitation clauses prevent a departing employee from recruiting "
            "colleagues or soliciting clients. These are easier to enforce than "
            "non-compete clauses but must still be reasonable in scope and duration. "
            "Best practice: limit to 12 months and define 'solicitation' explicitly; "
            "passive recruitment (e.g., responding to LinkedIn messages) should be "
            "excluded."
        )
    },

    # ── WARRANTIES ────────────────────────────────────────────────────────
    {
        "id": "warr_001",
        "topic": "Warranties",
        "content": (
            "Express warranties are explicit promises about product or service performance. "
            "Disclaimer of implied warranties (merchantability, fitness for particular "
            "purpose) is standard in commercial contracts but must be conspicuous (e.g., "
            "in ALL CAPS per UCC requirements). Watch for: warranty periods shorter than "
            "the product's expected useful life, and carve-outs that undermine the value "
            "of the warranty."
        )
    },
    {
        "id": "warr_002",
        "topic": "Warranties",
        "content": (
            "Software warranties often cover uptime (SLA), bug fixes within a specified "
            "timeframe, and compliance with documentation. The remedy for breach of a "
            "software warranty is typically limited to 'commercially reasonable efforts "
            "to repair'—a low bar. Best practice: negotiate specific performance thresholds "
            "and, if SLA is breached, service credits or the right to terminate."
        )
    },

    # ── DATA PRIVACY ──────────────────────────────────────────────────────
    {
        "id": "priv_001",
        "topic": "Data Privacy",
        "content": (
            "Under GDPR (EU), CCPA (California), and emerging US state laws, Data Processing "
            "Agreements (DPAs) are mandatory when one party processes personal data on "
            "behalf of another. Risk: contracts without DPAs expose both parties to "
            "regulatory fines. Best practice: include a DPA addendum specifying data "
            "processing purposes, security measures, sub-processor restrictions, and "
            "data subject rights fulfillment."
        )
    },
    {
        "id": "priv_002",
        "topic": "Data Privacy",
        "content": (
            "Data breach notification obligations under GDPR require notification within "
            "72 hours to supervisory authorities and without undue delay to affected "
            "individuals. Contract provisions should align: specify the processor's "
            "obligation to notify the controller 'without undue delay' and provide "
            "assistance with its own notification obligations."
        )
    },

    # ── ASSIGNMENT ────────────────────────────────────────────────────────
    {
        "id": "asgn_001",
        "topic": "Assignment",
        "content": (
            "Anti-assignment clauses prevent parties from transferring contract rights "
            "without consent. Risk: overly broad clauses can prevent legitimate business "
            "restructuring (e.g., post-acquisition integration). Best practice: allow "
            "assignment to affiliates without consent, and include 'change of control' "
            "carve-outs or provisions giving the non-assigning party termination rights "
            "upon a change of control."
        )
    },

    # ── INJUNCTIVE RELIEF ─────────────────────────────────────────────────
    {
        "id": "inj_001",
        "topic": "Injunctive Relief",
        "content": (
            "Injunctive relief provisions allow a party to seek a court order to stop "
            "certain conduct immediately, without having to post bond. These are most "
            "common in NDA and IP contexts. Risk: they create significant leverage for "
            "the party holding the provision, as an injunction can halt business "
            "operations. Best practice: ensure the provision is reciprocal and limited "
            "to material IP or confidentiality breaches."
        )
    },

    # ── SEVERABILITY ──────────────────────────────────────────────────────
    {
        "id": "sev_001",
        "topic": "Severability",
        "content": (
            "Severability clauses ensure that if one provision is found unenforceable, "
            "the rest of the contract remains valid. Without a severability clause, "
            "an unenforceable provision can in some jurisdictions void the entire contract. "
            "Best practice: include a severability clause in all commercial contracts."
        )
    },

    # ── ENTIRE AGREEMENT ──────────────────────────────────────────────────
    {
        "id": "int_001",
        "topic": "Integration Clause",
        "content": (
            "An integration (entire agreement) clause states that the written contract "
            "is the complete agreement between the parties, superseding all prior "
            "negotiations and representations. This prevents parties from relying on "
            "oral promises or emails made before signing. Risk: parties should ensure "
            "ALL agreed terms are in the written contract before signing, as prior "
            "communications will be inadmissible to vary the agreement."
        )
    },

    # ── GENERAL BEST PRACTICE ─────────────────────────────────────────────
    {
        "id": "gen_001",
        "topic": "General Contract Principles",
        "content": (
            "Every commercial contract should clearly define: (1) the parties, "
            "(2) the consideration exchanged, (3) the scope of work or deliverables, "
            "(4) timelines and milestones, (5) payment terms, (6) risk allocation "
            "(liability, indemnification), (7) dispute resolution, and (8) termination rights. "
            "Missing any of these elements creates ambiguity that typically resolves "
            "against the drafter."
        )
    },
    {
        "id": "gen_002",
        "topic": "General Contract Principles",
        "content": (
            "The contra proferentem doctrine states that ambiguous terms in a contract "
            "are construed against the party who drafted them. This is a strong incentive "
            "to draft with precision and to negotiate clear definitions for any term that "
            "could carry more than one reasonable meaning."
        )
    },
    {
        "id": "gen_003",
        "topic": "Contract Review Checklist",
        "content": (
            "When reviewing any commercial contract, examine: "
            "1. Is the scope of work/services clearly defined? "
            "2. Are payment terms specific (amount, due date, late fees)? "
            "3. Is liability capped at a reasonable amount? "
            "4. Are IP ownership rights clearly assigned? "
            "5. Are termination rights and consequences defined? "
            "6. Is the governing law neutral or favorable? "
            "7. Are confidentiality obligations time-limited and reasonable? "
            "8. Are there data privacy DPA provisions if personal data is involved?"
        )
    },
]
