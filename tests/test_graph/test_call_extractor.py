"""Tests for graph.call_extractor.CallExtractor."""

from __future__ import annotations

from graph.call_extractor import CallExtractor, CallSite, ReferenceSite


def test_python_bare_function_call() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("python", "def foo():\n    bar()\n")
    assert sites == [CallSite(callee_name="bar", is_self_qualified=False, qualifier_name=None)]


def test_python_self_qualified_call() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("python", "def foo(self):\n    self.bar()\n")
    assert sites == [CallSite(callee_name="bar", is_self_qualified=True, qualifier_name=None)]


def test_python_cls_qualified_call() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("python", "def foo(cls):\n    cls.bar()\n")
    assert sites == [CallSite(callee_name="bar", is_self_qualified=True, qualifier_name=None)]


def test_python_object_qualified_call() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("python", "def foo():\n    obj.bar()\n")
    assert sites == [CallSite(callee_name="bar", is_self_qualified=False, qualifier_name="obj")]


def test_python_class_qualified_call() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("python", "def foo():\n    Helper.bar()\n")
    assert sites == [CallSite(callee_name="bar", is_self_qualified=False, qualifier_name="Helper")]


def test_python_recursive_call() -> None:
    extractor = CallExtractor()
    sites = extractor.extract(
        "python", "def fib(n):\n    return fib(n - 1) + fib(n - 2)\n"
    )
    assert sites == [
        CallSite(callee_name="fib", is_self_qualified=False, qualifier_name=None),
        CallSite(callee_name="fib", is_self_qualified=False, qualifier_name=None),
    ]


def test_python_multiple_calls_in_body() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("python", "def foo():\n    a()\n    self.b()\n    obj.c()\n")
    assert [s.callee_name for s in sites] == ["a", "b", "c"]


def test_python_chained_attribute_call_has_no_qualifier() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("python", "def foo():\n    a.b.c()\n")
    assert sites == [CallSite(callee_name="c", is_self_qualified=False, qualifier_name=None)]


def test_python_dynamic_dispatch_is_skipped() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("python", "def foo():\n    funcs[0]()\n")
    assert sites == []


def test_javascript_bare_function_call() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("javascript", "function foo() {\n  bar();\n}\n")
    assert sites == [CallSite(callee_name="bar", is_self_qualified=False, qualifier_name=None)]


def test_javascript_this_qualified_call() -> None:
    extractor = CallExtractor()
    sites = extractor.extract(
        "javascript", "class A {\n  foo() {\n    this.bar();\n  }\n}\n"
    )
    assert sites == [CallSite(callee_name="bar", is_self_qualified=True, qualifier_name=None)]


def test_javascript_object_qualified_call() -> None:
    extractor = CallExtractor()
    sites = extractor.extract("javascript", "function foo() {\n  obj.bar();\n}\n")
    assert sites == [CallSite(callee_name="bar", is_self_qualified=False, qualifier_name="obj")]


def test_no_calls_returns_empty_list() -> None:
    extractor = CallExtractor()
    assert extractor.extract("python", "x = 1\n") == []


def test_unsupported_language_returns_empty_list() -> None:
    extractor = CallExtractor()
    assert extractor.extract("ruby", "def foo; bar; end") == []


def test_module_level_reference_argument_is_captured() -> None:
    extractor = CallExtractor()
    sites = extractor.extract_module_level(
        "javascript", 'router.post("/register", register);\n'
    )
    assert sites.reference_sites == [ReferenceSite(name="register")]


def test_module_level_multiple_reference_arguments() -> None:
    extractor = CallExtractor()
    sites = extractor.extract_module_level(
        "javascript", 'router.get("/me", protect, me);\n'
    )
    assert sites.reference_sites == [ReferenceSite(name="protect"), ReferenceSite(name="me")]


def test_module_level_call_site_is_also_captured() -> None:
    extractor = CallExtractor()
    sites = extractor.extract_module_level("javascript", 'router.post("/x", handler);\n')
    assert sites.call_sites == [CallSite(callee_name="post", is_self_qualified=False, qualifier_name="router")]


def test_module_level_does_not_descend_into_function_bodies() -> None:
    code = (
        "function setup() {\n"
        "  helperInsideFunction();\n"
        "}\n"
        "router.post('/x', topLevelHandler);\n"
    )
    extractor = CallExtractor()
    sites = extractor.extract_module_level("javascript", code)
    assert sites.reference_sites == [ReferenceSite(name="topLevelHandler")]
    assert [c.callee_name for c in sites.call_sites] == ["post"]


def test_module_level_string_and_number_arguments_are_not_references() -> None:
    extractor = CallExtractor()
    sites = extractor.extract_module_level("javascript", 'app.listen(3000, "localhost");\n')
    assert sites.reference_sites == []


def test_module_level_unsupported_language_returns_empty() -> None:
    extractor = CallExtractor()
    sites = extractor.extract_module_level("python", "foo(bar)\n")
    assert sites.call_sites == []
    assert sites.reference_sites == []
