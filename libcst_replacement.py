import csv
import json
import ast
import argparse
import collections
import re
import textwrap
from typing import Optional, List, Dict, Any, Tuple, Set
import warnings
import copy
import builtins
import keyword

# LibCST imports for format-preserving transformations
try:
    import libcst as cst
    from libcst import matchers as m
    from libcst import metadata as cst_metadata

    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False
    print("Warning: LibCST not available. Format preservation will be limited.")


def get_outputs(file_path):
    """Read outputs from JSON file, similar to original get_outputs function"""
    outputs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if "prompt" in data and "output" in data and "solution" in data:
                        outputs.append(
                            {
                                "prompt": data["prompt"],
                                "output": data["output"],
                                "solution": data["solution"],
                            }
                        )
                    else:
                        print(f"Warning: Missing keys in line: {line}")
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line} - {e}")
    return outputs


def _strip_code_fences(s: str) -> str:
    """Extract code from markdown code fences"""
    m = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    return m[0] if m else s


def _sanitize_whitespace(s: str) -> str:
    """Normalize whitespace and remove BOM"""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.lstrip("\ufeff")
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = s.expandtabs(4)
    s = textwrap.dedent(s)
    s = s.lstrip("\n").rstrip()
    return s


def _fix_py2_print(s: str) -> str:
    """Convert Python 2 print statements to Python 3"""
    lines = []
    pat = re.compile(r"^\s*print(?!\s*\()(?P<rest>.+)$")
    for line in s.split("\n"):
        m = pat.match(line)
        if m:
            lines.append(
                re.sub(pat, lambda mm: "print(" + mm.group("rest").rstrip() + ")", line)
            )
        else:
            lines.append(line)
    return "\n".join(lines)


def normalize_solution(text: str) -> str:
    """Normalize text to parseable Python source code"""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = _strip_code_fences(s)
    s = _sanitize_whitespace(s)
    s = _fix_py2_print(s)
    return s


def _wrap_in_func(s: str) -> str:
    """Wrap code in function to handle indentation issues"""
    indented = "\n".join(("    " + ln if ln.strip() else ln) for ln in s.split("\n"))
    return "def __snippet__():\n" + indented + "\n"


def is_valid_ast(code: str) -> bool:
    """Check if code can be parsed as valid AST"""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            ast.parse(code)
        return True
    except SyntaxError:
        # Try wrapping in function
        try:
            wrapped = _wrap_in_func(code)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                ast.parse(wrapped)
            return True
        except SyntaxError:
            return False


def is_valid_libcst(code: str) -> bool:
    """Check if code can be parsed by LibCST"""
    if not LIBCST_AVAILABLE:
        return False
    try:
        cst.parse_expression(code)
        return True
    except:
        try:
            cst.parse_module(code)
            return True
        except:
            return False


def concatenate_and_validate(outputs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Step 2: Concatenate output+prompt and solution+prompt, filter out invalid AST data
    Returns list of valid entries with both output and solution concatenated
    """
    valid_entries = []
    skipped_count = 0

    for i, item in enumerate(outputs):
        prompt = item["prompt"]
        output = item["output"]
        solution = item["solution"]

        # Concatenate prompt + output and prompt + solution (prompt is function header)
        output_combined = normalize_solution(f"{prompt}\n{output}")
        solution_combined = normalize_solution(f"{prompt}\n{solution}")

        # Check if both can be parsed as valid AST
        if is_valid_ast(output_combined) and is_valid_ast(solution_combined):
            valid_entries.append(
                {
                    "index": i,
                    "prompt": prompt,
                    "output": output,
                    "solution": solution,
                    "output_combined": output_combined,
                    "solution_combined": solution_combined,
                }
            )
        else:
            skipped_count += 1

    print(f"AST解析结果: 成功 {len(valid_entries)} 条, 失败 {skipped_count} 条")
    return valid_entries


class BlockSwapTransformer(ast.NodeTransformer):
    """Legacy AST transformer (unused): kept for fallback compatibility."""

    def __init__(self):
        self.transformations = []

    def visit_If(self, node):
        return node


class VariableRenameStyle:
    CAPITALIZE = "capitalize"
    CAMEL = "camel"
    SNAKE = "snake"


def _split_words(name: str) -> List[str]:
    # Preserve leading underscores
    core = name.lstrip("_")
    lead = name[: len(name) - len(core)]
    # Split snake parts first
    parts: List[str] = []
    for token in core.split("_"):
        if not token:
            continue
        # Split Camel/Pascal boundaries: fooBar -> [foo, Bar]
        subparts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\b)", token)
        parts.extend(sp.lower() for sp in subparts if sp)
    # Only return semantic word parts; leading underscores handled elsewhere
    return parts


def convert_name(name: str, style: str) -> str:
    # Keep leading underscores intact
    lead = ""
    core = name
    while core.startswith("_"):
        lead += "_"
        core = core[1:]
    if not core:
        return name
    words = _split_words(core)
    if not words:
        return name
    if style == VariableRenameStyle.SNAKE:
        newcore = "_".join(words)
    elif style == VariableRenameStyle.CAMEL:
        newcore = words[0] + "".join(w.capitalize() for w in words[1:])
    else:  # CAPITALIZE
        newcore = words[0].capitalize() + "".join(w.capitalize() for w in words[1:])
    return lead + newcore


BUILTINS_SET: Set[str] = set(dir(builtins))
RESERVED_NAMES: Set[str] = {"self", "cls", "__class__"}
PY_KEYWORDS: Set[str] = set(keyword.kwlist)


def _collect_target_names(target: cst.CSTNode) -> Set[str]:
    names: Set[str] = set()

    class _TargetVisitor(cst.CSTVisitor):
        def visit_Name(self, node: cst.Name) -> Optional[bool]:
            names.add(node.value)
            return False

        def visit_Attribute(self, node: cst.Attribute) -> Optional[bool]:
            # Do not descend into attribute targets (obj.attr)
            return False

        def visit_Subscript(self, node: cst.Subscript) -> Optional[bool]:
            return False

    target.visit(_TargetVisitor())
    return names


class ScopeInfo:
    def __init__(self, scope_type: str):
        self.type = scope_type  # module|class|function|comprehension
        self.locals: Set[str] = set()
        self.imports: Set[str] = set()
        self.globals: Set[str] = set()
        self.nonlocals: Set[str] = set()
        self.params: Set[str] = set()
        self.protected: Set[str] = set()  # names protected due to inner nonlocal
        self.mapping: Dict[str, str] = {}


class SymbolCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst_metadata.ParentNodeProvider,)

    def __init__(self):
        self.stack: List[ScopeInfo] = []

    # Scope management
    def visit_Module(self, node: cst.Module):
        self.stack.append(ScopeInfo("module"))

    def leave_Module(self, node: cst.Module):
        self.stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef):
        self.stack.append(ScopeInfo("class"))

    def leave_ClassDef(self, node: cst.ClassDef):
        self.stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef):
        info = ScopeInfo("function")
        # Collect parameters to avoid renaming API
        params = node.params

        def _param_name(p: Optional[cst.Param]) -> Optional[str]:
            return p.name.value if p and isinstance(p.name, cst.Name) else None

        for p in params.posonly_params:
            n = _param_name(p)
            if n:
                info.params.add(n)
        for p in params.params:
            n = _param_name(p)
            if n:
                info.params.add(n)
        for p in params.kwonly_params:
            n = _param_name(p)
            if n:
                info.params.add(n)
        # *args/**kwargs in LibCST
        if getattr(params, "star_arg", None) and isinstance(params.star_arg, cst.Param):
            n = _param_name(params.star_arg)
            if n:
                info.params.add(n)
        if getattr(params, "star_kwarg", None) and isinstance(
            params.star_kwarg, cst.Param
        ):
            n = _param_name(params.star_kwarg)
            if n:
                info.params.add(n)
        self.stack.append(info)

    def leave_FunctionDef(self, node: cst.FunctionDef):
        self.stack.pop()

    def visit_ListComp(self, node: cst.ListComp):
        self.stack.append(ScopeInfo("comprehension"))

    def leave_ListComp(self, node: cst.ListComp):
        self.stack.pop()

    def visit_SetComp(self, node: cst.SetComp):
        self.stack.append(ScopeInfo("comprehension"))

    def leave_SetComp(self, node: cst.SetComp):
        self.stack.pop()

    def visit_DictComp(self, node: cst.DictComp):
        self.stack.append(ScopeInfo("comprehension"))

    def leave_DictComp(self, node: cst.DictComp):
        self.stack.pop()

    def visit_GeneratorExp(self, node: cst.GeneratorExp):
        self.stack.append(ScopeInfo("comprehension"))

    def leave_GeneratorExp(self, node: cst.GeneratorExp):
        self.stack.pop()

    # Helpers
    def _cur(self) -> ScopeInfo:
        return self.stack[-1]

    # Declarations / targets collecting
    def visit_Assign(self, node: cst.Assign):
        for t in node.targets:
            self._cur().locals.update(_collect_target_names(t.target))

    def visit_AnnAssign(self, node: cst.AnnAssign):
        self._cur().locals.update(_collect_target_names(node.target))

    def visit_AugAssign(self, node: cst.AugAssign):
        self._cur().locals.update(_collect_target_names(node.target))

    def visit_For(self, node: cst.For):
        self._cur().locals.update(_collect_target_names(node.target))

    def visit_With(self, node: cst.With):
        for it in node.items:
            if it.asname and isinstance(it.asname.name, cst.Name):
                self._cur().locals.add(it.asname.name.value)

    def visit_ExceptHandler(self, node: cst.ExceptHandler):
        if node.name and isinstance(node.name, cst.Name):
            self._cur().locals.add(node.name.value)

    # No generic Comprehension node in LibCST; collect in mapping builder

    def visit_NamedExpr(self, node: cst.NamedExpr):
        # walrus operator: target := value
        self._cur().locals.update(_collect_target_names(node.target))

    # Imports
    def visit_Import(self, node: cst.Import):
        for a in node.names:
            if isinstance(a, cst.ImportAlias):
                if a.asname and isinstance(a.asname.name, cst.Name):
                    self._cur().imports.add(a.asname.name.value)
                else:
                    # from module import name -> binds 'name'
                    if isinstance(a.name, cst.Name):
                        self._cur().imports.add(a.name.value)
                    elif isinstance(a.name, cst.Attribute) and isinstance(
                        a.name.attr, cst.Name
                    ):
                        self._cur().imports.add(a.name.attr.value)

    def visit_ImportFrom(self, node: cst.ImportFrom):
        # node.names can be ImportStar or a sequence of ImportAlias
        if isinstance(node.names, cst.ImportStar):
            return
        if node.names:
            for a in node.names:
                if isinstance(a, cst.ImportAlias):
                    if a.asname and isinstance(a.asname.name, cst.Name):
                        self._cur().imports.add(a.asname.name.value)
                    else:
                        if isinstance(a.name, cst.Name):
                            self._cur().imports.add(a.name.value)

    # Scope modifiers
    def visit_Global(self, node: cst.Global):
        for n in node.names:
            # LibCST uses NameItem(name=Name("x"))
            if isinstance(n, cst.NameItem) and isinstance(n.name, cst.Name):
                self._cur().globals.add(n.name.value)
            elif isinstance(n, cst.Name):
                self._cur().globals.add(n.value)

    def visit_Nonlocal(self, node: cst.Nonlocal):
        names: Set[str] = set()
        for n in node.names:
            if isinstance(n, cst.NameItem) and isinstance(n.name, cst.Name):
                names.add(n.name.value)
            elif isinstance(n, cst.Name):
                names.add(n.value)
        self._cur().nonlocals.update(names)
        # Protect same names in nearest outer function scope
        for si in reversed(self.stack[:-1]):
            if si.type == "function":
                si.protected.update(names)
                break


class VariableRenamer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst_metadata.ParentNodeProvider,)

    def __init__(self, mapping_stack: List[ScopeInfo], style: str):
        # We only need mapping, locals, params per scope
        self.style = style
        self.stack: List[ScopeInfo] = []
        # Store the precomputed mapping hierarchy to consult on push events
        self.precomputed: List[ScopeInfo] = mapping_stack
        self.transformations: List[Dict[str, Any]] = []

    # Scope push/pop mirroring collector
    def visit_Module(self, node: cst.Module) -> Optional[bool]:
        self.stack.append(self.precomputed.pop(0))

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        self.stack.pop()
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self.stack.append(self.precomputed.pop(0))

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self.stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self.stack.append(self.precomputed.pop(0))

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        self.stack.pop()
        return updated_node

    def visit_ListComp(self, node: cst.ListComp) -> Optional[bool]:
        self.stack.append(self.precomputed.pop(0))

    def leave_ListComp(
        self, original_node: cst.ListComp, updated_node: cst.ListComp
    ) -> cst.ListComp:
        self.stack.pop()
        return updated_node

    def visit_SetComp(self, node: cst.SetComp) -> Optional[bool]:
        self.stack.append(self.precomputed.pop(0))

    def leave_SetComp(
        self, original_node: cst.SetComp, updated_node: cst.SetComp
    ) -> cst.SetComp:
        self.stack.pop()
        return updated_node

    def visit_DictComp(self, node: cst.DictComp) -> Optional[bool]:
        self.stack.append(self.precomputed.pop(0))

    def leave_DictComp(
        self, original_node: cst.DictComp, updated_node: cst.DictComp
    ) -> cst.DictComp:
        self.stack.pop()
        return updated_node

    def visit_GeneratorExp(self, node: cst.GeneratorExp) -> Optional[bool]:
        self.stack.append(self.precomputed.pop(0))

    def leave_GeneratorExp(
        self, original_node: cst.GeneratorExp, updated_node: cst.GeneratorExp
    ) -> cst.GeneratorExp:
        self.stack.pop()
        return updated_node

    # Utility
    def _nearest_mapping_for_use(self, name: str) -> Optional[str]:
        # Starting from innermost scope, if name is local in that scope, use that mapping
        for i in range(len(self.stack) - 1, -1, -1):
            scope = self.stack[i]
            # If name is local in this scope
            if name in scope.mapping:
                return scope.mapping[name]
            # If name is declared as local/import/param in this scope but no mapping, it's shadowed here
            if (
                name in scope.locals
                or name in scope.params
                or name in scope.imports
                or name in scope.globals
                or name in scope.nonlocals
            ):
                return None
        return None

    def _should_skip_name(self, node: cst.Name) -> bool:
        try:
            parent = self.get_metadata(cst_metadata.ParentNodeProvider, node)
        except Exception:
            # If metadata is unavailable for this node, be conservative and skip renaming
            return True
        # Skip attribute.attr
        if isinstance(parent, cst.Attribute) and parent.attr is node:
            return True
        # Skip keyword names in function calls/definitions
        if isinstance(parent, cst.Arg) and parent.keyword is node:
            return True
        # Skip parameter names
        if isinstance(parent, cst.Param) and parent.name is node:
            return True
        # Skip function/class names
        if isinstance(parent, (cst.FunctionDef, cst.ClassDef)) and parent.name is node:
            return True
        # Skip import alias names
        if isinstance(parent, cst.ImportAlias):
            return True
        # Skip global/nonlocal declarations
        if isinstance(parent, (cst.Global, cst.Nonlocal)):
            return True
        return False

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.CSTNode:
        # Use original_node for metadata lookups; metadata is attached to the original tree
        if self._should_skip_name(original_node):
            return updated_node
        old = original_node.value
        # Builtins/keywords/reserved not renamed
        if old in BUILTINS_SET or old in PY_KEYWORDS or old in RESERVED_NAMES:
            return updated_node
        # Find mapping
        new = self._nearest_mapping_for_use(old)
        if new and new != old:
            self.transformations.append(
                {
                    "type": "var_rename",
                    "original_name": old,
                    "new_name": new,
                    "format_preserved": True,
                }
            )
            return updated_node.with_changes(value=new)
        return updated_node


def build_scope_mappings(module: cst.Module, style: str) -> List[ScopeInfo]:
    # First pass: collect symbols
    wrapper = cst_metadata.MetadataWrapper(module)
    collector = SymbolCollector()
    wrapper.visit(collector)

    # Now compute mappings per scope in the same traversal order we pushed them
    # To do this, re-run a lightweight walk that pushes the same scopes
    precomputed: List[ScopeInfo] = []

    class _MappingBuilder(cst.CSTVisitor):
        def __init__(self):
            self.stack: List[ScopeInfo] = []

        def _push(self, si: ScopeInfo):
            self.stack.append(si)
            precomputed.append(si)

        def _compute_mapping(self, si: ScopeInfo):
            if si.type not in ("function", "comprehension"):
                si.mapping = {}
                return
            candidates = (
                si.locals
                - si.imports
                - si.params
                - si.globals
                - si.nonlocals
                - si.protected
                - BUILTINS_SET
                - PY_KEYWORDS
                - RESERVED_NAMES
            )
            # Build a taken set for conflict avoidance
            taken = set(si.locals) | set(si.params) | set(si.imports)
            mapping: Dict[str, str] = {}
            for name in sorted(candidates):
                new = convert_name(name, style)
                if not new or new == name:
                    continue
                # Avoid conflicts; tweak with suffixes if needed
                base = new
                idx = 1
                while (
                    new in taken
                    or new in mapping.values()
                    or new in BUILTINS_SET
                    or new in PY_KEYWORDS
                    or new in RESERVED_NAMES
                ):
                    new = f"{base}_{idx}"
                    idx += 1
                mapping[name] = new
                taken.add(new)
            si.mapping = mapping

        def visit_Module(self, node: cst.Module):
            si = ScopeInfo("module")
            self._push(si)

        def leave_Module(self, node: cst.Module):
            self.stack.pop()

        def visit_ClassDef(self, node: cst.ClassDef):
            si = ScopeInfo("class")
            self._push(si)

        def leave_ClassDef(self, node: cst.ClassDef):
            self.stack.pop()

        def visit_FunctionDef(self, node: cst.FunctionDef):
            # Find corresponding collected scope info by recomputing locals/imports/etc. from collector stack order
            si = ScopeInfo("function")
            # Copy from the collector's matching function scope by index
            # To align with collector order, we cannot easily reference; instead recompute parameters here
            # However, we still have collector.stack emptied; we will not reuse; Instead, approximate by scanning node
            # For correctness, we rebuild sets here similarly
            # Params
            params = node.params

            def _param_name(p: Optional[cst.Param]) -> Optional[str]:
                return p.name.value if p and isinstance(p.name, cst.Name) else None

            for p in params.posonly_params:
                n = _param_name(p)
                if n:
                    si.params.add(n)
            for p in params.params:
                n = _param_name(p)
                if n:
                    si.params.add(n)
            for p in params.kwonly_params:
                n = _param_name(p)
                if n:
                    si.params.add(n)
            if getattr(params, "star_arg", None) and isinstance(
                params.star_arg, cst.Param
            ):
                n = _param_name(params.star_arg)
                if n:
                    si.params.add(n)
            if getattr(params, "star_kwarg", None) and isinstance(
                params.star_kwarg, cst.Param
            ):
                n = _param_name(params.star_kwarg)
                if n:
                    si.params.add(n)

            # Gather locals/imports/globals/nonlocals within this function body quickly using a small walker
            class _LocalWalker(cst.CSTVisitor):
                def __init__(self, si: ScopeInfo):
                    self.si = si

                def visit_Assign(self, n: cst.Assign):
                    for t in n.targets:
                        self.si.locals.update(_collect_target_names(t.target))

                def visit_AnnAssign(self, n: cst.AnnAssign):
                    self.si.locals.update(_collect_target_names(n.target))

                def visit_AugAssign(self, n: cst.AugAssign):
                    self.si.locals.update(_collect_target_names(n.target))

                def visit_For(self, n: cst.For):
                    self.si.locals.update(_collect_target_names(n.target))

                def visit_With(self, n: cst.With):
                    for it in n.items:
                        if it.asname and isinstance(it.asname.name, cst.Name):
                            self.si.locals.add(it.asname.name.value)

                def visit_ExceptHandler(self, n: cst.ExceptHandler):
                    if n.name and isinstance(n.name, cst.Name):
                        self.si.locals.add(n.name.value)

                def visit_Import(self, n: cst.Import):
                    for a in n.names:
                        if isinstance(a, cst.ImportAlias):
                            if a.asname and isinstance(a.asname.name, cst.Name):
                                self.si.imports.add(a.asname.name.value)
                            else:
                                if isinstance(a.name, cst.Name):
                                    self.si.imports.add(a.name.value)
                                elif isinstance(a.name, cst.Attribute) and isinstance(
                                    a.name.attr, cst.Name
                                ):
                                    self.si.imports.add(a.name.attr.value)

                def visit_ImportFrom(self, n: cst.ImportFrom):
                    if n.names and isinstance(n.names, list):
                        if isinstance(n.names, cst.ImportStar):
                            return
                        if n.names:
                            for a in n.names:
                                if isinstance(a, cst.ImportAlias):
                                    if a.asname and isinstance(a.asname.name, cst.Name):
                                        self.si.imports.add(a.asname.name.value)
                                    else:
                                        if isinstance(a.name, cst.Name):
                                            self.si.imports.add(a.name.value)

                def visit_Global(self, n: cst.Global):
                    for nm in n.names:
                        if isinstance(nm, cst.NameItem) and isinstance(
                            nm.name, cst.Name
                        ):
                            self.si.globals.add(nm.name.value)
                        elif isinstance(nm, cst.Name):
                            self.si.globals.add(nm.value)

                def visit_Nonlocal(self, n: cst.Nonlocal):
                    for nm in n.names:
                        if isinstance(nm, cst.NameItem) and isinstance(
                            nm.name, cst.Name
                        ):
                            self.si.nonlocals.add(nm.name.value)
                        elif isinstance(nm, cst.Name):
                            self.si.nonlocals.add(nm.value)

            # Visit only this function node subtree
            node.visit(_LocalWalker(si))
            self._compute_mapping(si)
            self._push(si)

        def leave_FunctionDef(self, node: cst.FunctionDef):
            self.stack.pop()

        def visit_ListComp(self, node: cst.ListComp):
            si = ScopeInfo("comprehension")
            # LibCST: ListComp.for_in is CompFor; may chain with additional for/ifs
            comp = node.for_in
            while comp:
                si.locals.update(_collect_target_names(comp.target))
                comp = (
                    comp.inner_for_in
                    if isinstance(comp.inner_for_in, cst.CompFor)
                    else None
                )
            self._compute_mapping(si)
            self._push(si)

        def leave_ListComp(self, node: cst.ListComp):
            self.stack.pop()

        def visit_SetComp(self, node: cst.SetComp):
            si = ScopeInfo("comprehension")
            comp = node.for_in
            while comp:
                si.locals.update(_collect_target_names(comp.target))
                comp = (
                    comp.inner_for_in
                    if isinstance(comp.inner_for_in, cst.CompFor)
                    else None
                )
            self._compute_mapping(si)
            self._push(si)

        def leave_SetComp(self, node: cst.SetComp):
            self.stack.pop()

        def visit_DictComp(self, node: cst.DictComp):
            si = ScopeInfo("comprehension")
            comp = node.for_in
            while comp:
                si.locals.update(_collect_target_names(comp.target))
                comp = (
                    comp.inner_for_in
                    if isinstance(comp.inner_for_in, cst.CompFor)
                    else None
                )
            self._compute_mapping(si)
            self._push(si)

        def leave_DictComp(self, node: cst.DictComp):
            self.stack.pop()

        def visit_GeneratorExp(self, node: cst.GeneratorExp):
            si = ScopeInfo("comprehension")
            comp = node.for_in
            while comp:
                si.locals.update(_collect_target_names(comp.target))
                comp = (
                    comp.inner_for_in
                    if isinstance(comp.inner_for_in, cst.CompFor)
                    else None
                )
            self._compute_mapping(si)
            self._push(si)

        def leave_GeneratorExp(self, node: cst.GeneratorExp):
            self.stack.pop()

    wrapper.visit(_MappingBuilder())
    return precomputed


def apply_variable_renaming_libcst(
    code: str, style: str = VariableRenameStyle.CAPITALIZE
) -> Tuple[str, List[Dict[str, Any]]]:
    module = cst.parse_module(code)
    mappings = build_scope_mappings(module, style)
    wrapper = cst_metadata.MetadataWrapper(module)
    renamer = VariableRenamer(
        mapping_stack=[copy.deepcopy(m) for m in mappings], style=style
    )
    new_module = wrapper.visit(renamer)
    return new_module.code, renamer.transformations


def apply_ast_transformation(
    code: str, style: str = VariableRenameStyle.CAPITALIZE
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Step 3: Apply Block and Operand Swap transformation with format preservation
    Returns transformed code and list of transformations applied
    """
    # Try LibCST first (preserves formatting) - variable renaming
    if LIBCST_AVAILABLE:
        try:
            transformed_code, transforms = apply_variable_renaming_libcst(
                code, style=style
            )
            return transformed_code, transforms

        except Exception as e:
            print(f"LibCST transformation failed, falling back to AST: {e}")

    # Fallback: no-op (we avoid risky AST renaming)
    try:
        return code, []

    except Exception as e:
        print(f"Error during AST transformation: {e}")
        return code, []


def apply_libcst_transformation_safe(
    code: str, style: str = VariableRenameStyle.CAPITALIZE
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Safe LibCST transformation with comprehensive error handling
    """
    if not LIBCST_AVAILABLE:
        return apply_ast_transformation(code)

    try:
        # First, try to parse as a module
        try:
            module = cst.parse_module(code)
        except Exception as _e1:
            # Some versions expose ParserSyntaxError; be flexible
            # If parsing as module fails, try parsing as a statement
            try:
                # Wrap in a function if it looks like function body
                if not code.strip().startswith("def ") and not code.strip().startswith(
                    "class "
                ):
                    wrapped_code = f"def temp_func():\n" + "\n".join(
                        f"    {line}" for line in code.split("\n")
                    )
                    module = cst.parse_module(wrapped_code)
                    is_wrapped = True
                else:
                    module = cst.parse_module(code)
                    is_wrapped = False
            except Exception as e:
                print(f"LibCST parsing failed: {e}")
                return apply_ast_transformation(code, style=style)
        else:
            is_wrapped = False

        # Apply variable renaming
        transformed_code, transformations = apply_variable_renaming_libcst(
            module.code, style=style
        )

        # If we wrapped the code, unwrap it
        if is_wrapped:
            lines = transformed_code.strip().split("\n")
            if lines and lines[0].strip().startswith("def temp_func():"):
                # Remove the wrapper function and unindent
                unwrapped_lines = []
                for line in lines[1:]:
                    if line.startswith("    "):
                        unwrapped_lines.append(line[4:])  # Remove 4-space indentation
                    elif line.strip() == "":
                        unwrapped_lines.append("")  # Keep empty lines
                    else:
                        unwrapped_lines.append(line)  # Keep non-indented lines as-is
                transformed_code = "\n".join(unwrapped_lines)

        return transformed_code, transformations

    except Exception as e:
        print(f"LibCST transformation error: {e}")
        return apply_ast_transformation(code, style=style)


def remove_prompt_from_code(combined_code: str, prompt: str) -> str:
    """
    Step 4: Remove prompt part from the combined code
    Since prompt is just a function header, it shouldn't be affected by AST transformation
    """
    # Find the prompt in the combined code and remove it
    # The prompt is now at the beginning of the combined code
    lines = combined_code.split("\n")
    prompt_lines = prompt.split("\n")

    # Check if the combined code starts with the prompt
    if len(lines) >= len(prompt_lines):
        # Compare the first lines with prompt lines
        if lines[: len(prompt_lines)] == prompt_lines:
            # Remove the prompt part (keep everything after prompt)
            return "\n".join(lines[len(prompt_lines) :]).strip()

    # If prompt not found at the beginning, try to find it anywhere
    for i in range(len(lines) - len(prompt_lines) + 1):
        if lines[i : i + len(prompt_lines)] == prompt_lines:
            # Remove the prompt part
            remaining_lines = lines[:i] + lines[i + len(prompt_lines) :]
            return "\n".join(remaining_lines).strip()

    # If prompt not found, return original code
    return combined_code


def process_ast_replacements(
    file_path: str,
    output_path: str = "ast_replacement_results.json",
    style: str = "capitalize",
):
    """
    Main function to process Variable Renaming following the 5-step strategy
    """
    print("Step 1: Reading outputs from file...")
    outputs = get_outputs(file_path)

    print("Step 2: Concatenating and validating AST...")
    valid_entries = concatenate_and_validate(outputs)

    if not valid_entries:
        print("No valid entries found!")
        return

    results = []

    # Statistics tracking
    stats = {
        "total_entries": len(outputs),
        "valid_entries": len(valid_entries),
        "skipped_entries": len(outputs) - len(valid_entries),
        "output_transformations": {
            "total_applied": 0,
            "entries_with_transforms": 0,
            "entries_without_transforms": 0,
        },
        "solution_transformations": {
            "total_applied": 0,
            "entries_with_transforms": 0,
            "entries_without_transforms": 0,
        },
        "successful_replacements": {
            "output_only": 0,
            "solution_only": 0,
            "both": 0,
            "neither": 0,
        },
    }

    print("Step 3-5: Applying Variable Renaming and processing...")
    for entry in valid_entries:
        # Apply format-preserving transformation to both output and solution
        output_transformed, output_transforms = apply_libcst_transformation_safe(
            entry["output_combined"], style=style
        )
        solution_transformed, solution_transforms = apply_libcst_transformation_safe(
            entry["solution_combined"], style=style
        )

        # Update statistics
        stats["output_transformations"]["total_applied"] += len(output_transforms)
        stats["solution_transformations"]["total_applied"] += len(solution_transforms)

        if output_transforms:
            stats["output_transformations"]["entries_with_transforms"] += 1
        else:
            stats["output_transformations"]["entries_without_transforms"] += 1

        if solution_transforms:
            stats["solution_transformations"]["entries_with_transforms"] += 1
        else:
            stats["solution_transformations"]["entries_without_transforms"] += 1

        # Categorize replacement success
        if output_transforms and solution_transforms:
            stats["successful_replacements"]["both"] += 1
        elif output_transforms and not solution_transforms:
            stats["successful_replacements"]["output_only"] += 1
        elif not output_transforms and solution_transforms:
            stats["successful_replacements"]["solution_only"] += 1
        else:
            stats["successful_replacements"]["neither"] += 1

        # Track format preservation statistics
        if "format_preservation" not in stats:
            stats["format_preservation"] = {
                "libcst_transformations": 0,
                "ast_fallback_transformations": 0,
                "total_transformations": 0,
            }

        # Count LibCST vs AST transformations
        for transform in output_transforms + solution_transforms:
            stats["format_preservation"]["total_transformations"] += 1
            if transform.get("format_preserved", False):
                stats["format_preservation"]["libcst_transformations"] += 1
            else:
                stats["format_preservation"]["ast_fallback_transformations"] += 1

        # Remove prompt from transformed codes
        output_final = remove_prompt_from_code(output_transformed, entry["prompt"])
        solution_final = remove_prompt_from_code(solution_transformed, entry["prompt"])

        # Store results
        result = {
            "index": entry["index"],
            "original": {
                "prompt": entry["prompt"],
                "output": entry["output"],
                "solution": entry["solution"],
            },
            "transformed": {"output": output_final, "solution": solution_final},
            "transformations": {
                "output_transforms": output_transforms,
                "solution_transforms": solution_transforms,
            },
            "replacement_success": {
                "output_transformed": len(output_transforms) > 0,
                "solution_transformed": len(solution_transforms) > 0,
                "total_transforms": len(output_transforms) + len(solution_transforms),
            },
        }
        results.append(result)

    # Save results with statistics
    final_results = {"statistics": stats, "results": results}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("VARIABLE RENAMING STATISTICS")
    print("=" * 60)

    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total entries processed: {stats['total_entries']}")
    print(f"  Valid entries (AST parseable): {stats['valid_entries']}")
    print(f"  Skipped entries (invalid AST): {stats['skipped_entries']}")
    print(f"  Success rate: {stats['valid_entries']/stats['total_entries']*100:.1f}%")

    print(f"\n🔄 OUTPUT RENAME TRANSFORMATIONS:")
    print(
        f"  Total transformations applied: {stats['output_transformations']['total_applied']}"
    )
    print(
        f"  Entries with transformations: {stats['output_transformations']['entries_with_transforms']}"
    )
    print(
        f"  Entries without transformations: {stats['output_transformations']['entries_without_transforms']}"
    )
    print(
        f"  Average transformations per entry: {stats['output_transformations']['total_applied']/stats['valid_entries']:.2f}"
    )

    print(f"\n🔄 SOLUTION RENAME TRANSFORMATIONS:")
    print(
        f"  Total transformations applied: {stats['solution_transformations']['total_applied']}"
    )
    print(
        f"  Entries with transformations: {stats['solution_transformations']['entries_with_transforms']}"
    )
    print(
        f"  Entries without transformations: {stats['solution_transformations']['entries_without_transforms']}"
    )
    print(
        f"  Average transformations per entry: {stats['solution_transformations']['total_applied']/stats['valid_entries']:.2f}"
    )

    print(f"\n✅ REPLACEMENT SUCCESS BREAKDOWN:")
    print(
        f"  Both output and solution transformed: {stats['successful_replacements']['both']}"
    )
    print(
        f"  Only output transformed: {stats['successful_replacements']['output_only']}"
    )
    print(
        f"  Only solution transformed: {stats['successful_replacements']['solution_only']}"
    )
    print(f"  Neither transformed: {stats['successful_replacements']['neither']}")

    # Calculate success rates
    total_valid = stats["valid_entries"]
    if total_valid > 0:
        both_rate = stats["successful_replacements"]["both"] / total_valid * 100
        any_rate = (
            (
                stats["successful_replacements"]["both"]
                + stats["successful_replacements"]["output_only"]
                + stats["successful_replacements"]["solution_only"]
            )
            / total_valid
            * 100
        )

        print(f"\n📈 SUCCESS RATES:")
        print(f"  Both output and solution: {both_rate:.1f}%")
        print(f"  At least one transformed: {any_rate:.1f}%")

    # Show format preservation statistics
    if (
        "format_preservation" in stats
        and stats["format_preservation"]["total_transformations"] > 0
    ):
        fp_stats = stats["format_preservation"]
        libcst_rate = (
            fp_stats["libcst_transformations"] / fp_stats["total_transformations"] * 100
        )
        print(f"\n🎨 FORMAT PRESERVATION STATISTICS:")
        print(f"  Total transformations: {fp_stats['total_transformations']}")
        print(
            f"  LibCST (format preserved): {fp_stats['libcst_transformations']} ({libcst_rate:.1f}%)"
        )
        print(
            f"  AST fallback (format lost): {fp_stats['ast_fallback_transformations']} ({100-libcst_rate:.1f}%)"
        )
        if LIBCST_AVAILABLE:
            print(f"  LibCST library: ✅ Available")
        else:
            print(f"  LibCST library: ❌ Not installed (run: pip install libcst)")
    else:
        print(f"\n🎨 FORMAT PRESERVATION: No transformations applied")
        print(
            f"  LibCST library: {'✅ Available' if LIBCST_AVAILABLE else '❌ Not installed'}"
        )

    # Show examples of successful transformations (LibCST)
    print(f"\n🔍 EXAMPLES OF SUCCESSFUL VARIABLE RENAMING (LibCST):")
    successful_examples = [
        r for r in results if r["replacement_success"]["total_transforms"] > 0
    ][:5]

    if successful_examples:
        for i, result in enumerate(successful_examples):
            print(f"\n--- Example {i+1} (Entry {result['index']}) ---")
            print(
                f"Total transformations: {result['replacement_success']['total_transforms']}"
            )
            print(
                f"Output transformed: {result['replacement_success']['output_transformed']}"
            )
            print(
                f"Solution transformed: {result['replacement_success']['solution_transformed']}"
            )

            if result["transformations"]["output_transforms"]:
                print("\nOutput transformations:")
                for transform in result["transformations"]["output_transforms"]:
                    if transform.get("type") == "var_rename":
                        print(
                            f"  - {transform['type']}: {transform['original_name']} → {transform['new_name']}"
                        )

            if result["transformations"]["solution_transforms"]:
                print("\nSolution transformations:")
                for transform in result["transformations"]["solution_transforms"]:
                    if transform.get("type") == "var_rename":
                        print(
                            f"  - {transform['type']}: {transform['original_name']} → {transform['new_name']}"
                        )

            # Show code comparison - full output without character limits
            print(f"\nOriginal output:")
            print(result["original"]["output"])

            print(f"\nTransformed output:")
            print(result["transformed"]["output"])

            if result["replacement_success"]["solution_transformed"]:
                print(f"\nOriginal solution:")
                print(result["original"]["solution"])

                print(f"\nTransformed solution:")
                print(result["transformed"]["solution"])
    else:
        print("No successful variable renaming examples found.")

    print("\n" + "=" * 60)

    # Generate CSV statistics report
    csv_path = output_path.replace(".json", "_statistics.csv")
    generate_statistics_csv(results, csv_path)
    print(f"\n📊 Detailed statistics saved to: {csv_path}")


def generate_statistics_csv(results: List[Dict], csv_path: str):
    """Generate a CSV file with detailed statistics for each entry"""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            [
                "index",
                "output_transformed",
                "solution_transformed",
                "total_transforms",
                "output_transform_count",
                "solution_transform_count",
                "original_output_length",
                "transformed_output_length",
                "original_solution_length",
                "transformed_solution_length",
                "output_length_change",
                "solution_length_change",
            ]
        )

        # Write data for each entry
        for result in results:
            original_output = result["original"]["output"]
            transformed_output = result["transformed"]["output"]
            original_solution = result["original"]["solution"]
            transformed_solution = result["transformed"]["solution"]

            writer.writerow(
                [
                    result["index"],
                    result["replacement_success"]["output_transformed"],
                    result["replacement_success"]["solution_transformed"],
                    result["replacement_success"]["total_transforms"],
                    len(result["transformations"]["output_transforms"]),
                    len(result["transformations"]["solution_transforms"]),
                    len(original_output),
                    len(transformed_output),
                    len(original_solution),
                    len(transformed_solution),
                    len(transformed_output) - len(original_output),
                    len(transformed_solution) - len(original_solution),
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply scope-aware Variable Renaming (format-preserving)"
    )
    parser.add_argument("file_path", help="Path to the JSON file")
    parser.add_argument(
        "--output", default="ast_replacement_results.json", help="Output file path"
    )
    parser.add_argument(
        "--style",
        choices=["capitalize", "camel", "snake"],
        default="capitalize",
        help="Variable renaming style: capitalize|camel|snake",
    )
    args = parser.parse_args()

    process_ast_replacements(args.file_path, args.output, style=args.style)
