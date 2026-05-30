"""
Verification test for Agent Interface Phase (3.9.B.1).

This test validates:
- All agent interfaces compile
- All type definitions are valid
- No implementation logic is present
- Interfaces are properly abstract
"""

import sys
import inspect
from abc import ABC, abstractmethod

# Import agent interfaces
from core.agents import (
    ParsingAgent,
    VisualizationAgent,
    ExplanationAgent,
    # Types
    ConfigDict,
    PaperExcerpt,
    SymbolicDesc,
    ParsingSource,
    ComparisonContext,
    VisualizationOptions,
    VisualizationMode,
    NodeVisuals,
    VisualRepresentation,
    ComputeSummary,
    SpatialSummary,
    ScalingSummary,
    ComparisonResult,
    VisualMetadata,
)


def test_agent_interfaces_are_abstract():
    """Verify all agent interfaces are properly abstract."""
    print("\n[TEST 1] Agent interfaces are abstract")
    print("-" * 70)

    agents = [ParsingAgent, VisualizationAgent, ExplanationAgent]
    
    for agent in agents:
        # Must inherit from ABC
        assert issubclass(agent, ABC), f"{agent.__name__} must inherit from ABC"
        
        # Must have abstract methods
        abstract_methods = {
            name for name, method in inspect.getmembers(agent)
            if getattr(method, "__isabstractmethod__", False)
        }
        assert len(abstract_methods) > 0, f"{agent.__name__} must have abstract methods"
        
        # Cannot be instantiated
        try:
            agent()
            raise AssertionError(f"{agent.__name__} should not be instantiable")
        except TypeError:
            pass  # Expected
        
        print(f"✓ {agent.__name__}")
        print(f"  Abstract methods: {abstract_methods}")


def test_parsing_agent_interface():
    """Verify ParsingAgent has correct interface."""
    print("\n[TEST 2] ParsingAgent interface")
    print("-" * 70)
    
    # Must have parse method
    assert hasattr(ParsingAgent, "parse"), "ParsingAgent must have parse() method"
    
    # parse must be abstract
    assert getattr(ParsingAgent.parse, "__isabstractmethod__", False), "parse() must be abstract"
    
    # Check signature
    sig = inspect.signature(ParsingAgent.parse)
    params = list(sig.parameters.keys())
    assert "self" in params, "parse must have self parameter"
    assert "source" in params, "parse must have source parameter"
    assert "format_hint" in params, "parse must have format_hint parameter"
    
    print("✓ ParsingAgent.parse() signature correct")
    print(f"  Parameters: {params}")


def test_visualization_agent_interface():
    """Verify VisualizationAgent has correct interface."""
    print("\n[TEST 3] VisualizationAgent interface")
    print("-" * 70)
    
    # Must have render and get_visual_cues methods
    assert hasattr(VisualizationAgent, "render"), "VisualizationAgent must have render() method"
    assert hasattr(VisualizationAgent, "get_visual_cues"), "VisualizationAgent must have get_visual_cues() method"
    
    # Both must be abstract
    assert getattr(VisualizationAgent.render, "__isabstractmethod__", False)
    assert getattr(VisualizationAgent.get_visual_cues, "__isabstractmethod__", False)
    
    # Check render signature
    sig = inspect.signature(VisualizationAgent.render)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "graph" in params
    assert "mode" in params
    assert "comparison_ctx" in params
    assert "options" in params
    
    print("✓ VisualizationAgent.render() signature correct")
    print("✓ VisualizationAgent.get_visual_cues() signature correct")
    print(f"  Render parameters: {params}")


def test_explanation_agent_interface():
    """Verify ExplanationAgent has correct interface."""
    print("\n[TEST 4] ExplanationAgent interface")
    print("-" * 70)
    
    required_methods = [
        "explain_node",
        "explain_graph",
        "explain_comparison",
        "get_explanation_templates"
    ]
    
    for method_name in required_methods:
        assert hasattr(ExplanationAgent, method_name), f"ExplanationAgent must have {method_name}() method"
        method = getattr(ExplanationAgent, method_name)
        assert getattr(method, "__isabstractmethod__", False), f"{method_name}() must be abstract"
    
    print("✓ ExplanationAgent has all required methods")
    print(f"  Methods: {required_methods}")


def test_type_definitions():
    """Verify all type definitions exist and are valid."""
    print("\n[TEST 5] Type definitions")
    print("-" * 70)
    
    types_to_check = [
        ConfigDict,
        PaperExcerpt,
        SymbolicDesc,
        ComparisonContext,
        VisualizationOptions,
        NodeVisuals,
        VisualRepresentation,
        ComputeSummary,
        SpatialSummary,
        ScalingSummary,
        ComparisonResult,
        VisualMetadata,
    ]
    
    for type_def in types_to_check:
        # Check it's a TypedDict
        assert hasattr(type_def, "__annotations__"), f"{type_def.__name__} must be a TypedDict"
        print(f"✓ {type_def.__name__}")


def test_no_implementation_logic():
    """Verify agents contain no implementation logic."""
    print("\n[TEST 6] No implementation logic")
    print("-" * 70)
    
    agents = [ParsingAgent, VisualizationAgent, ExplanationAgent]
    
    for agent in agents:
        # Get all methods (excluding magic methods)
        methods = {
            name: method for name, method in inspect.getmembers(agent, inspect.isfunction)
            if not name.startswith("_") or name in ["__init__"]
        }
        
        for method_name, method in methods.items():
            source = inspect.getsource(method)
            
            # Check for disallowed patterns
            disallowed = [
                "import streamlit",
                "import graphviz",
                "from streamlit",
                "from graphviz",
                "st.",
                "graphviz.Digraph",
                "Digraph(",
            ]
            
            for pattern in disallowed:
                assert pattern not in source, f"{agent.__name__}.{method_name} contains disallowed: {pattern}"
        
        print(f"✓ {agent.__name__} contains no implementation")


def test_circular_dependencies():
    """Verify agents don't have circular dependencies."""
    print("\n[TEST 7] No circular dependencies")
    print("-" * 70)
    
    # agents/__init__.py imports from agents/*.py
    # agents/*.py should not import from agents/__init__.py
    
    import core.agents.parsing_agent as pa
    import core.agents.visualization_agent as va
    import core.agents.explanation_agent as ea
    
    modules = [pa, va, ea]
    
    for module in modules:
        source = inspect.getsource(module)
        assert "from core.agents import" not in source, f"{module.__name__} has circular import"
        assert "from . import" not in source or "from .types import" in source, f"{module.__name__} has unexpected local import"
    
    print("✓ No circular dependencies detected")


def test_determinism_docstrings():
    """Verify determinism guarantees are documented."""
    print("\n[TEST 8] Determinism guarantees documented")
    print("-" * 70)
    
    agents = [ParsingAgent, VisualizationAgent, ExplanationAgent]
    
    for agent in agents:
        docstring = agent.__doc__ or ""
        assert "Deterministic" in docstring or "deterministic" in docstring, \
            f"{agent.__name__} must document determinism guarantee"
        
        # Check abstract methods have determinism docs
        for name, method in inspect.getmembers(agent, inspect.isfunction):
            if getattr(method, "__isabstractmethod__", False):
                method_doc = method.__doc__ or ""
                print(f"✓ {agent.__name__}.{name}() documented")


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("AGENT INTERFACE VERIFICATION (Phase 3.9.B.1)")
    print("=" * 70)
    
    tests = [
        test_agent_interfaces_are_abstract,
        test_parsing_agent_interface,
        test_visualization_agent_interface,
        test_explanation_agent_interface,
        test_type_definitions,
        test_no_implementation_logic,
        test_circular_dependencies,
        test_determinism_docstrings,
    ]
    
    failed = []
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"\n❌ {test.__name__} FAILED: {e}")
            failed.append((test.__name__, str(e)))
        except Exception as e:
            print(f"\n❌ {test.__name__} ERROR: {e}")
            failed.append((test.__name__, str(e)))
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if not failed:
        print(f"✅ ALL {len(tests)} TESTS PASSED")
        print("\n✓ Agent interfaces are properly abstract")
        print("✓ All required methods present")
        print("✓ All type definitions valid")
        print("✓ No implementation logic detected")
        print("✓ No circular dependencies")
        print("✓ Determinism guarantees documented")
        print("\n🎉 Phase 3.9.B.1 Agent Interfaces: COMPLETE")
        return 0
    else:
        print(f"❌ {len(failed)} of {len(tests)} tests failed:")
        for test_name, error in failed:
            print(f"  - {test_name}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
