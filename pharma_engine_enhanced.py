"""
Enhanced Pharmaceutical Engine
Provides sophisticated mode detection, pharmaceutical domain logic, and analysis optimization
"""

from typing import Dict, List, Tuple, Optional
import re

# ============================================================================
# KEYWORDS FOR ANALYSIS MODE DETECTION
# ============================================================================

MODE_DETECTION_KEYWORDS = {
    "📚 In-depth Knowledge (Database)": {
        "primary_indicators": [
            'complete', 'entire', 'whole', 'comprehensive',
            'everything', 'detailed', 'full details', 'all requirements',
            'complete details', 'entire procedure', 'full specification',
            'all parameters', 'comprehensive list', 'everything about'
        ],
        "secondary_indicators": [
            'specification', 'requirement', 'procedure', 'how to',
            'step by step', 'detailed explanation', 'elaborate'
        ],
        "score_weight": 1.0
    },
    
    "📄 Summary (Database)": {
        "primary_indicators": [
            'summarize', 'summary', 'brief', 'concise', 'overview',
            'key points', 'main', 'essential', 'highlight', 'brief overview',
            'quick summary', 'in short', 'outline', 'synopsis'
        ],
        "secondary_indicators": [
            'important', 'critical', 'main findings', 'summary version',
            'condensed', 'abbreviated', 'shortened', 'bottom line'
        ],
        "score_weight": 1.0
    },
    
    "📋 SOP Style": {
        "primary_indicators": [
            'sop', 'standard operating procedure', 'procedure document',
            'create sop', 'write procedure', 'operating procedure',
            'formal procedure', 'procedural document', 'step procedure',
            'official procedure', 'standardized procedure'
        ],
        "secondary_indicators": [
            'format as sop', 'sop format', 'procedure style',
            'operational procedure', 'working procedure', 'documented procedure'
        ],
        "score_weight": 1.0
    },
    
    "✅ Regulatory Audit Checklist": {
        "primary_indicators": [
            'audit', 'checklist', 'compliance', 'gmp compliance',
            'regulatory checklist', 'audit checklist', 'compliance checklist',
            'verification', 'audit trail', 'gmp audit', 'regulatory audit',
            'inspection', 'audit readiness', 'compliance verification'
        ],
        "secondary_indicators": [
            'gap analysis', 'compliance assessment', 'regulatory readiness',
            'audit preparation', 'compliance check', 'verification checklist'
        ],
        "score_weight": 1.0
    },
    
    "📚 In-depth + Online": {
        "primary_indicators": [
            'current', 'latest', 'recent', 'comprehensive',
            'combined', 'both', 'internal and external', 'database and online',
            'current standards', 'latest requirements', 'current plus',
            'comprehensive research', 'with current'
        ],
        "secondary_indicators": [
            'compare with', 'versus current', 'aligned with', 'versus standards',
            'how does it compare', 'what about current'
        ],
        "score_weight": 0.9
    },
    
    "🌐 Online Only": {
        "primary_indicators": [
            'online', 'internet', 'web', 'research', 'current standards',
            'latest guidance', 'fda guidance', 'usup standards', 'bp standards',
            'regulatory guidance', 'official standards', 'what does',
            'current requirement', 'online research', 'web search'
        ],
        "secondary_indicators": [
            'from web', 'online source', 'internet source', 'published',
            'official document', 'guidance document', 'industry standard'
        ],
        "score_weight": 0.85
    },
    
    "📊 Summary + Online": {
        "primary_indicators": [
            'brief', 'quick', 'fast', 'summary', 'concise',
            'latest summary', 'current summary', 'brief current',
            'quick summary with current', 'brief and current'
        ],
        "secondary_indicators": [
            'condensed', 'abbreviated', 'short', 'quick reference',
            'quick update', 'brief update', 'executive summary with current'
        ],
        "score_weight": 0.85
    }
}

# ============================================================================
# PHARMACEUTICAL DOMAIN KEYWORDS
# ============================================================================

PHARMA_DOMAIN_KEYWORDS = {
    "analytical": [
        'hplc', 'gc', 'lc-ms', 'chromatography', 'analytical',
        'system suitability', 'method validation', 'precision', 'accuracy',
        'linearity', 'specificity', 'robustness', 'dissolution'
    ],
    
    "formulation": [
        'tablet', 'capsule', 'liquid', 'powder', 'suspension',
        'solution', 'ointment', 'gel', 'cream', 'formulation',
        'excipient', 'active ingredient', 'potency', 'strength'
    ],
    
    "manufacturing": [
        'gmp', 'manufacturing', 'production', 'batch', 'lot',
        'scale-up', 'validation', 'process validation', 'equipment',
        'cleaning', 'sterilization', 'aseptic'
    ],
    
    "quality": [
        'quality control', 'qc', 'qa', 'quality assurance',
        'specification', 'acceptance criteria', 'limits', 'usp',
        'bp', 'ph.eur.', 'ich', 'coa', 'certificate of analysis'
    ],
    
    "regulatory": [
        'regulatory', 'fda', 'ema', 'guidance', 'warning letter',
        'inspection', 'compliance', 'gxp', 'audit', '21 cfr',
        'warning letter', 'regulatory action', 'approval'
    ],
    
    "microbiology": [
        'microbiology', 'sterility', 'bioburden', 'endotoxin',
        'microbes', 'bacterial', 'fungal', 'antimicrobial',
        'preservative', 'microbial contamination'
    ],
    
    "stability": [
        'stability', 'shelf life', 'degradation', 'expiry',
        'storage condition', 'accelerated', 'long term',
        'intermediate', 'light testing', 'moisture'
    ],
    
    "clinical": [
        'clinical', 'trial', 'pharmacokinetics', 'bioavailability',
        'efficacy', 'safety', 'adverse event', 'patient'
    ]
}

# ============================================================================
# MODE DETECTION FUNCTION
# ============================================================================

def detect_analysis_mode(query: str) -> Tuple[str, float]:
    """
    Detect the most appropriate analysis mode based on query.
    
    Args:
        query: User query string
    
    Returns:
        Tuple of (detected_mode, confidence_score)
    """
    query_lower = query.lower()
    
    # Calculate scores for each mode
    mode_scores = {}
    
    for mode, keywords in MODE_DETECTION_KEYWORDS.items():
        score = 0.0
        
        # Check primary indicators (higher weight)
        primary_matches = sum(
            1 for keyword in keywords["primary_indicators"]
            if keyword in query_lower
        )
        score += primary_matches * 2.0
        
        # Check secondary indicators (lower weight)
        secondary_matches = sum(
            1 for keyword in keywords["secondary_indicators"]
            if keyword in query_lower
        )
        score += secondary_matches * 1.0
        
        # Apply mode weight
        score *= keywords.get("score_weight", 1.0)
        
        mode_scores[mode] = score
    
    # Find mode with highest score
    if mode_scores:
        best_mode = max(mode_scores, key=mode_scores.get)
        confidence = mode_scores[best_mode]
        
        # Normalize confidence to 0-1 range
        max_possible_score = 20  # Rough estimate
        normalized_confidence = min(confidence / max_possible_score, 1.0)
        
        return best_mode, normalized_confidence
    
    # Default to In-depth if no match
    return "📚 In-depth Knowledge (Database)", 0.5


def detect_pharma_domain(query: str) -> List[str]:
    """
    Detect pharmaceutical domain areas referenced in query.
    Useful for providing domain-specific context.
    
    Args:
        query: User query string
    
    Returns:
        List of detected domain areas
    """
    query_lower = query.lower()
    detected_domains = []
    
    for domain, keywords in PHARMA_DOMAIN_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_domains.append(domain)
    
    return detected_domains


def get_domain_specific_context(domains: List[str]) -> Dict[str, str]:
    """
    Provide domain-specific context and considerations.
    
    Args:
        domains: List of detected pharmaceutical domains
    
    Returns:
        Dictionary with domain-specific guidance
    """
    context_guidance = {
        "analytical": (
            "Focus on method validation requirements (ICH Q2R2), system suitability criteria, "
            "acceptance ranges (typically RSD ≤2% for HPLC), and documentation requirements."
        ),
        "formulation": (
            "Consider excipient compatibility, stability, bioavailability, and compliance with "
            "pharmacopeial standards (USP, BP, Ph.Eur.). Include specification requirements."
        ),
        "manufacturing": (
            "Reference GMP requirements, equipment validation, process validation, "
            "change control, and batch record completeness."
        ),
        "quality": (
            "Include specification limits, acceptance criteria per pharmacopeiae, "
            "test methods, and documentation requirements."
        ),
        "regulatory": (
            "Cross-reference FDA CFR sections, EMA guidelines, ICH documents, "
            "and regulatory expectations from recent guidance."
        ),
        "microbiology": (
            "Specify limits for sterility, bioburden, endotoxin per USP/BP/Ph.Eur., "
            "include preservation methods, and testing frequencies."
        ),
        "stability": (
            "Include storage conditions, testing intervals, degradation pathways, "
            "shelf-life determination, and labeling requirements."
        ),
        "clinical": (
            "Focus on efficacy, safety, pharmacokinetics, and clinical evaluation "
            "per ICH guidance documents."
        )
    }
    
    return {domain: context_guidance.get(domain, "") for domain in domains if domain in context_guidance}


# ============================================================================
# SEARCH DEPTH OPTIMIZATION
# ============================================================================

def optimize_search_depth(mode: str, query_length: int, estimated_complexity: float = 0.5) -> int:
    """
    Optimize vector search depth based on mode and query characteristics.
    
    Args:
        mode: Analysis mode
        query_length: Length of user query in characters
        estimated_complexity: Complexity score 0-1
    
    Returns:
        Recommended search depth (chunks)
    """
    # Base depth by mode
    base_depths = {
        "📚 In-depth Knowledge (Database)": 16,      # Comprehensive
        "📄 Summary (Database)": 10,                  # Focused
        "📋 SOP Style": 14,                          # Detailed procedure
        "✅ Regulatory Audit Checklist": 18,         # Thorough
        "📚 In-depth + Online": 12,                   # Balanced
        "🌐 Online Only": 0,                          # No database search
        "📊 Summary + Online": 8                      # Quick
    }
    
    base = base_depths.get(mode, 12)
    
    # Adjust based on query length
    if query_length > 300:  # Complex query
        base += 3
    elif query_length < 50:  # Simple query
        base -= 2
    
    # Adjust based on estimated complexity
    base = int(base * (0.7 + estimated_complexity * 0.6))
    
    # Keep within reasonable bounds (4-20)
    return max(4, min(base, 20))


# ============================================================================
# RESPONSE QUALITY ASSESSMENT
# ============================================================================

def assess_response_completeness(response_text: str, mode: str) -> Dict[str, float]:
    """
    Assess if response adequately addresses the analysis mode requirements.
    
    Args:
        response_text: Generated response
        mode: Analysis mode
    
    Returns:
        Dictionary with quality metrics
    """
    text_lower = response_text.lower()
    metrics = {
        "citations_present": 0.0,
        "structure_score": 0.0,
        "technical_depth": 0.0,
        "completeness": 0.0
    }
    
    # Count citations
    citation_count = text_lower.count("[") + text_lower.count("source:")
    metrics["citations_present"] = min(citation_count / 10, 1.0)  # Normalize
    
    # Check structure based on mode
    structure_indicators = {
        "📚 In-depth Knowledge (Database)": ["##", "###", "**", "table", "[source"],
        "📄 Summary (Database)": ["##", "bullet", "key", "-"],
        "📋 SOP Style": ["purpose", "scope", "responsibility", "procedure", "8."],
        "✅ Regulatory Audit Checklist": ["✅", "❌", "⚠️", "checklist", "status"],
        "📚 In-depth + Online": ["database", "online", "agreement", "discrepanc"],
        "🌐 Online Only": ["fda", "ema", "usup", "bp", "guideline"],
        "📊 Summary + Online": ["key", "finding", "takeaway", "recommend"]
    }
    
    indicators = structure_indicators.get(mode, [])
    structure_count = sum(1 for ind in indicators if ind in text_lower)
    metrics["structure_score"] = (structure_count / len(indicators)) if indicators else 0.5
    
    # Assess technical depth
    technical_terms = [
        'specification', 'acceptance', 'validation', 'procedure',
        'requirement', 'standard', 'limit', 'parameter'
    ]
    tech_count = sum(1 for term in technical_terms if term in text_lower)
    metrics["technical_depth"] = min(tech_count / 8, 1.0)
    
    # Overall completeness
    metrics["completeness"] = (
        metrics["citations_present"] * 0.3 +
        metrics["structure_score"] * 0.3 +
        metrics["technical_depth"] * 0.4
    )
    
    return metrics


# ============================================================================
# PROMPT REFINEMENT SUGGESTIONS
# ============================================================================

def get_refinement_suggestions(query: str, detected_mode: str, 
                              confidence: float) -> List[str]:
    """
    Suggest query refinements if confidence is low.
    
    Args:
        query: Original user query
        detected_mode: Detected analysis mode
        confidence: Confidence score (0-1)
    
    Returns:
        List of refinement suggestions
    """
    suggestions = []
    
    if confidence < 0.6:
        if detected_mode == "📄 Summary (Database)":
            suggestions.append(
                "💡 For a better summary, you could add keywords like 'brief', 'key points', or 'overview'"
            )
        elif detected_mode == "📋 SOP Style":
            suggestions.append(
                "💡 For SOP format, include terms like 'create an SOP', 'write procedure', or 'format as SOP'"
            )
        elif detected_mode == "✅ Regulatory Audit Checklist":
            suggestions.append(
                "💡 For audit checklist, mention 'audit', 'compliance', 'gmp compliance', or 'regulatory verification'"
            )
    
    # Query clarity suggestions
    if len(query) < 30:
        suggestions.append(
            "💡 Consider providing more detail in your query for more accurate analysis"
        )
    
    if not any(char in query for char in ['?', '.']):
        suggestions.append(
            "💡 Structured queries (with clear questions) often produce better results"
        )
    
    return suggestions


# ============================================================================
# ANALYSIS CONFIGURATION GENERATOR
# ============================================================================

def generate_analysis_config(query: str) -> Dict:
    """
    Generate complete analysis configuration based on query.
    
    Args:
        query: User query
    
    Returns:
        Configuration dictionary for analysis execution
    """
    # Detect mode
    detected_mode, confidence = detect_analysis_mode(query)
    
    # Detect pharmaceutical domains
    domains = detect_pharma_domain(query)
    domain_context = get_domain_specific_context(domains)
    
    # Optimize search depth
    search_depth = optimize_search_depth(detected_mode, len(query))
    
    # Get refinement suggestions
    suggestions = get_refinement_suggestions(query, detected_mode, confidence)
    
    return {
        "query": query,
        "detected_mode": detected_mode,
        "mode_confidence": confidence,
        "recommended_search_depth": search_depth,
        "pharma_domains": domains,
        "domain_context": domain_context,
        "refinement_suggestions": suggestions,
        "requires_database": confidence > 0.3,  # Only if some confidence in database analysis
        "should_enable_streaming": detected_mode not in ["✅ Regulatory Audit Checklist"],  # Checklists better without stream
        "estimated_response_time": {
            "📚 In-depth Knowledge (Database)": "3-5 seconds",
            "📄 Summary (Database)": "2-3 seconds",
            "📋 SOP Style": "5-8 seconds",
            "✅ Regulatory Audit Checklist": "6-10 seconds",
            "📚 In-depth + Online": "8-15 seconds",
            "🌐 Online Only": "4-8 seconds",
            "📊 Summary + Online": "4-7 seconds"
        }.get(detected_mode, "3-5 seconds")
    }


# ============================================================================
# EXPORT & VALIDATION
# ============================================================================

def validate_analysis_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate generated analysis configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not config.get("query"):
        errors.append("Query cannot be empty")
    
    if config.get("recommended_search_depth", 0) < 4:
        errors.append("Search depth too low (minimum 4)")
    
    if config.get("recommended_search_depth", 0) > 20:
        errors.append("Search depth too high (maximum 20)")
    
    if not config.get("detected_mode"):
        errors.append("Analysis mode not detected")
    
    if config.get("mode_confidence", 0) < 0:
        errors.append("Confidence score invalid")
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # Example usage
    test_queries = [
        "Provide complete details on HPLC system suitability",
        "Summarize the main quality control requirements",
        "Create an SOP for analytical balance calibration",
        "Generate audit checklist for GMP compliance",
        "What are current FDA guidance on analytical methods?"
    ]
    
    for query in test_queries:
        config = generate_analysis_config(query)
        is_valid, errors = validate_analysis_config(config)
        
        print(f"\nQuery: {query}")
        print(f"Detected Mode: {config['detected_mode']}")
        print(f"Confidence: {config['mode_confidence']:.2%}")
        print(f"Search Depth: {config['recommended_search_depth']}")
        print(f"Valid: {is_valid}")
        if errors:
            print(f"Errors: {errors}")
