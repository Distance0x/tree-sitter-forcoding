import csv
import json
import argparse
import re
import random
from typing import Optional, List, Dict, Any, Tuple, Set
import builtins
import keyword
from get_outputs import get_outputs

# Global cache and one-time error flag for Tree-sitter parser
_TS_PARSER: Optional[Any] = None
_TS_INIT_ERROR_ONCE: bool = False


def _get_parser() -> Optional[Any]:
    """Create and cache a Python parser using the official tree-sitter-python API.

    Implementation strictly follows:
        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)  (or Parser(); parser.set_language(PY_LANGUAGE))

    Prints only one error line on failure and returns None.
    """
    global _TS_PARSER, _TS_INIT_ERROR_ONCE
    if _TS_PARSER is not None:
        return _TS_PARSER

    # Strict per-docs path
    try:
        import tree_sitter_python as tspython  # type: ignore
        from tree_sitter import Language as TS_Language, Parser as TS_Parser  # type: ignore

        py_lang = TS_Language(tspython.language())

        parser: Any = None
        # Preferred: constructor with language
        try:
            parser = TS_Parser(py_lang)
        except TypeError:
            # Some versions require no-arg constructor
            parser = TS_Parser()
        except Exception:
            parser = TS_Parser()

        # Ensure language is configured (covers both ctor styles)
        try:
            if hasattr(parser, "set_language"):
                parser.set_language(py_lang)
            elif hasattr(parser, "language"):
                setattr(parser, "language", py_lang)
        except Exception:
            # If setting language fails, we still try to use parser if it was set via ctor
            pass

        if hasattr(parser, "parse"):
            _TS_PARSER = parser
            return _TS_PARSER
    except Exception as e:
        if not _TS_INIT_ERROR_ONCE:
            print(f"Tree-sitter parser init failed: {e}")
            _TS_INIT_ERROR_ONCE = True

    # Give up if strict path not available
    return None


def _wrap_in_func(s: str) -> str:
    indented = "\n".join(("    " + ln if ln.strip() else ln) for ln in s.split("\n"))
    return "def __snippet__():\n" + indented + "\n"


def is_valid_ts(code: str) -> bool:
    """Validate code with Tree-sitter only.

    Rules:
    - Parse with Tree-sitter; if root has errors, try wrapped-in-function once.
    - If parser is not available, treat as invalid (no AST fallback).
    """
    parser = _get_parser()
    if parser is None:
        return False
    try:
        tree = parser.parse(code.encode("utf-8"))
    except Exception:
        try:
            wrapped = _wrap_in_func(code)
            tree = parser.parse(wrapped.encode("utf-8"))
        except Exception:
            return False

    root = tree.root_node
    has_err_attr = getattr(root, "has_error", None)
    try:
        if callable(has_err_attr):
            return not bool(has_err_attr())
        return not bool(has_err_attr)
    except Exception:
        # If API differs, conservatively accept as valid
        return True


def _strip_code_fences_local(s: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    return m[0] if m else s


def normalize_solution_preserve_leading(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    s = _strip_code_fences_local(s)
    # normalize newlines and invisibles, expand tabs, but DO NOT dedent or strip leading newlines/spaces
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.lstrip("\ufeff")
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = s.expandtabs(4)
    # only trim trailing whitespace to avoid accumulating tail blanks
    s = s.rstrip()
    return s


def concatenate_and_validate(outputs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    valid_entries = []
    skipped_count = 0
    for i, item in enumerate(outputs):
        prompt = item["prompt"]
        output = item["output"]
        solution = item["solution"]

        prompt_norm = normalize_solution_preserve_leading(prompt)
        output_combined = normalize_solution_preserve_leading(f"{prompt}\n{output}")
        solution_combined = normalize_solution_preserve_leading(f"{prompt}\n{solution}")

        if is_valid_ts(output_combined) and is_valid_ts(solution_combined):
            valid_entries.append(
                {
                    "index": i,
                    "prompt": prompt,
                    "prompt_norm": prompt_norm,
                    "output": output,
                    "solution": solution,
                    "output_combined": output_combined,
                    "solution_combined": solution_combined,
                }
            )
        else:
            skipped_count += 1

    print(f"TS解析结果: 成功 {len(valid_entries)} 条, 失败 {skipped_count} 条")
    return valid_entries


# ----------------------- Name style helpers -----------------------


class VariableRenameStyle:
    CAPITALIZE = "capitalize"
    CAMEL = "camel"
    SNAKE = "snake"


class RenameMode:
    STYLE = "style"  # use convert_name with style
    UNDERSCORE = "underscore"  # insert underscores randomly
    NONE = "none"  # skip renaming


def _split_words(name: str) -> List[str]:
    core = name.lstrip("_")
    parts: List[str] = []
    for token in core.split("_"):
        if not token:
            continue
        subparts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\b)", token)
        parts.extend(sp.lower() for sp in subparts if sp)
    return parts


def convert_name(name: str, style: str) -> str:
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
    else:
        newcore = words[0].capitalize() + "".join(w.capitalize() for w in words[1:])
    return lead + newcore


def insert_underscores_random(
    name: str, prob: float, repeat: int, rng: random.Random
) -> str:
    """Insert underscores into variable name probabilistically.

    - Preserve leading underscores
    - Consider insertion positions between characters of the core (not at the very start)
    - With probability `prob`, insert '_' * k at a boundary, where k is in [1, repeat]
    - Ensure at least one insertion (force one random boundary if none chosen)
    """
    if repeat < 1:
        repeat = 1
    lead = ""
    core = name
    while core.startswith("_"):
        lead += "_"
        core = core[1:]
    if len(core) <= 1:
        return name
    # boundaries between characters: positions 1..len(core)-1
    chars = list(core)
    out: list[str] = [chars[0]]
    inserted_any = False
    for i in range(1, len(chars)):
        # decide whether to insert before chars[i]
        if rng.random() < prob:
            k = 1 if repeat == 1 else rng.randint(1, repeat)
            out.append("_" * k)
            inserted_any = True
        out.append(chars[i])
    # Ensure at least one insertion
    if not inserted_any:
        pos = rng.randint(1, len(chars) - 1)
        out = [chars[0]]
        for i in range(1, len(chars)):
            if i == pos:
                k = 1 if repeat == 1 else rng.randint(1, repeat)
                out.append("_" * k)
            out.append(chars[i])
    return lead + "".join(out)


BUILTINS_SET: Set[str] = set(dir(builtins))
RESERVED_NAMES: Set[str] = {"self", "cls", "__class__"}
PY_KEYWORDS: Set[str] = set(keyword.kwlist)


# ----------------------- Tree-sitter based rename -----------------------


class ScopeInfo:
    def __init__(self, scope_type: str):
        self.type = scope_type  # module|class|function|comprehension
        self.locals: Set[str] = set()
        self.imports: Set[str] = set()
        self.globals: Set[str] = set()
        self.nonlocals: Set[str] = set()
        self.params: Set[str] = set()
        self.protected: Set[str] = set()
        self.mapping: Dict[str, str] = {}


def _collect_identifiers_in_target(
    node, code_bytes: bytes, skip_attr_subscript=True
) -> Set[Tuple[int, int, str]]:
    """
    Collect identifiers from an LHS target expression.
    Returns a set of (start_byte, end_byte, name).
    Skips attributes (obj.attr) and subscripts (a[b]) when skip_attr_subscript=True.
    """
    out: Set[Tuple[int, int, str]] = set()

    def walk(n):
        t = n.type
        if t == "identifier":
            name = code_bytes[n.start_byte : n.end_byte].decode("utf-8")
            out.add((n.start_byte, n.end_byte, name))
            return
        if skip_attr_subscript and t in ("attribute", "subscript"):
            return
        for c in n.children:
            walk(c)

    walk(node)
    return out


## Tree-sitter parser init uses official tree-sitter-python API


def _node_equals(a, b) -> bool:
    return a is b or (
        a is not None
        and b is not None
        and a.start_byte == b.start_byte
        and a.end_byte == b.end_byte
        and a.type == b.type
    )


def _compute_scope_mappings_ts(
    root,
    code_bytes: bytes,
    style: str,
    rename_params: bool,
    mode: str,
    underscore_prob: float,
    underscore_repeat: int,
    rng: random.Random,
    var_prob: float = 1.0,
) -> List[ScopeInfo]:
    precomputed: List[ScopeInfo] = []
    stack: List[ScopeInfo] = []

    def push(si: ScopeInfo):
        stack.append(si)
        precomputed.append(si)

    def pop():
        stack.pop()

    def cur() -> ScopeInfo:
        return stack[-1]

    def compute_mapping(si: ScopeInfo):
        if mode == RenameMode.NONE:
            si.mapping = {}
            return
        if si.type not in ("function", "comprehension"):
            si.mapping = {}
            return
        # base candidates: locals; optionally include params when enabled
        base = set(si.locals)
        if rename_params:
            base |= set(si.params)
        # exclusions
        candidates = (
            base
            - si.imports
            - si.globals
            - si.nonlocals
            - si.protected
            - BUILTINS_SET
            - PY_KEYWORDS
            - RESERVED_NAMES
        )
        taken = set(si.locals) | set(si.params) | set(si.imports)
        mapping: Dict[str, str] = {}
        for name in sorted(candidates):
            if mode == RenameMode.UNDERSCORE:
                # variable-level sampling: only rename a fraction of variables
                if var_prob < 1.0 and rng.random() >= var_prob:
                    continue
                new = insert_underscores_random(
                    name, underscore_prob, underscore_repeat, rng
                )
            elif mode == RenameMode.NONE:
                continue
            else:
                new = convert_name(name, style)
            if not new or new == name:
                continue
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

    # Helpers to get child by field name safely
    def field(node, name: str):
        try:
            return node.child_by_field_name(name)
        except Exception:
            return None

    # DFS
    def visit(n):
        t = n.type
        # Enter scopes
        if t == "module":
            push(ScopeInfo("module"))
        elif t == "function_definition":
            # Collect params
            si = ScopeInfo("function")
            params = field(n, "parameters")
            if params is not None:
                # In tree-sitter-python, simple parameters may appear as direct 'identifier'
                # children of the 'parameters' node, while others are wrapped in specific nodes.
                containers = [params] + [
                    c for c in params.children if c.type == "parameters"
                ]

                def add_name_node(name_node):
                    if name_node and name_node.type == "identifier":
                        si.params.add(
                            code_bytes[
                                name_node.start_byte : name_node.end_byte
                            ].decode("utf-8")
                        )

                for cont in containers:
                    for ch in cont.children:
                        if ch.type == "identifier":
                            # Direct simple parameter like def f(x):
                            add_name_node(ch)
                        elif ch.type in (
                            "parameter",
                            "typed_parameter",
                            "default_parameter",
                            "typed_default_parameter",
                            "list_splat_pattern",
                            "dictionary_splat_pattern",
                        ):
                            add_name_node(field(ch, "name"))
                        # else: skip commas, stars, types, defaults, etc.
            push(si)
        elif t in (
            "list_comprehension",
            "set_comprehension",
            "dictionary_comprehension",
            "generator_expression",
        ):
            push(ScopeInfo("comprehension"))

        # Collect declarations within current scope
        if stack:
            # assignments
            if t in ("assignment", "augmented_assignment"):
                left = field(n, "left") or (n.children[0] if n.children else None)
                if left is not None:
                    for _, _, name in _collect_identifiers_in_target(left, code_bytes):
                        cur().locals.add(name)
            # walrus
            if t == "assignment_expression":
                left = field(n, "left")
                if left and left.type == "identifier":
                    name = code_bytes[left.start_byte : left.end_byte].decode("utf-8")
                    cur().locals.add(name)
            # for targets
            if t == "for_in_clause" or t == "for_statement":
                left = field(n, "left")
                if left is not None:
                    for _, _, name in _collect_identifiers_in_target(left, code_bytes):
                        cur().locals.add(name)
            # with as alias
            if t == "with_item":
                alias = field(n, "alias")
                if alias and alias.type == "identifier":
                    cur().locals.add(
                        code_bytes[alias.start_byte : alias.end_byte].decode("utf-8")
                    )
            # except name
            if t == "except_clause":
                name_node = field(n, "name")
                if name_node and name_node.type == "identifier":
                    cur().locals.add(
                        code_bytes[name_node.start_byte : name_node.end_byte].decode(
                            "utf-8"
                        )
                    )
            # imports
            if t == "import_statement":
                for ch in n.children:
                    if ch.type == "aliased_import":
                        alias = field(ch, "alias")
                        if alias and alias.type == "identifier":
                            cur().imports.add(
                                code_bytes[alias.start_byte : alias.end_byte].decode(
                                    "utf-8"
                                )
                            )
                        else:
                            name_node = field(ch, "name")
                            if name_node is not None:
                                # last identifier of dotted name
                                ids = [
                                    c
                                    for c in name_node.children
                                    if c.type == "identifier"
                                ]
                                if ids:
                                    last = ids[-1]
                                    cur().imports.add(
                                        code_bytes[
                                            last.start_byte : last.end_byte
                                        ].decode("utf-8")
                                    )
            if t == "import_from_statement":
                import_clause = field(n, "import_clause")
                if import_clause is not None:
                    for ch in import_clause.children:
                        if ch.type == "aliased_import":
                            alias = field(ch, "alias")
                            if alias and alias.type == "identifier":
                                cur().imports.add(
                                    code_bytes[
                                        alias.start_byte : alias.end_byte
                                    ].decode("utf-8")
                                )
                            else:
                                name_node = field(ch, "name")
                                if name_node is not None:
                                    ids = [
                                        c
                                        for c in name_node.children
                                        if c.type == "identifier"
                                    ]
                                    if ids:
                                        last = ids[-1]
                                        cur().imports.add(
                                            code_bytes[
                                                last.start_byte : last.end_byte
                                            ].decode("utf-8")
                                        )
            # globals/nonlocals
            if t == "global_statement":
                for ch in n.children:
                    if ch.type == "identifier":
                        cur().globals.add(
                            code_bytes[ch.start_byte : ch.end_byte].decode("utf-8")
                        )
            if t == "nonlocal_statement":
                names = set()
                for ch in n.children:
                    if ch.type == "identifier":
                        nm = code_bytes[ch.start_byte : ch.end_byte].decode("utf-8")
                        names.add(nm)
                cur().nonlocals.update(names)
                # protect in nearest outer function
                for si in reversed(stack[:-1]):
                    if si.type == "function":
                        si.protected.update(names)
                        break

        # Recurse
        for c in n.children:
            visit(c)

        # On leave, compute mapping and pop scope
        if t == "module":
            compute_mapping(cur())
            pop()
        elif t == "function_definition":
            compute_mapping(cur())
            pop()
        elif t in (
            "list_comprehension",
            "set_comprehension",
            "dictionary_comprehension",
            "generator_expression",
        ):
            compute_mapping(cur())
            pop()

    visit(root)
    return precomputed


def _apply_renaming_ts(
    code: str,
    style: str,
    rename_params: bool = False,
    mode: str = RenameMode.STYLE,
    underscore_prob: float = 0.25,
    underscore_repeat: int = 1,
    rng: Optional[random.Random] = None,
    var_prob: float = 1.0,
) -> Tuple[str, List[Dict[str, Any]]]:
    parser = _get_parser()
    if parser is None:
        return code, []

    is_wrapped = False
    try:
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
    except Exception:
        # try wrapping
        wrapped = _wrap_in_func(code)
        try:
            code_bytes = wrapped.encode("utf-8")
            tree = parser.parse(code_bytes)
            is_wrapped = True
        except Exception as e:
            print(f"Tree-sitter parse failed: {e}")
            return code, []

    root = tree.root_node
    if rng is None:
        rng = random.Random()
    mappings = _compute_scope_mappings_ts(
        root,
        code_bytes,
        style,
        rename_params,
        mode,
        underscore_prob,
        underscore_repeat,
        rng,
        var_prob,
    )

    # second pass: generate replacements for eligible identifier uses
    replacements: List[Tuple[int, int, str]] = []
    transforms: List[Dict[str, Any]] = []
    pre_list = list(mappings)
    scope_stack: List[ScopeInfo] = []

    def nearest_mapping(name: str) -> Optional[str]:
        for si in reversed(scope_stack):
            if name in si.mapping:
                return si.mapping[name]
            if (
                name in si.locals
                or name in si.params
                or name in si.imports
                or name in si.globals
                or name in si.nonlocals
            ):
                return None
        return None

    def field(node, name: str):
        try:
            return node.child_by_field_name(name)
        except Exception:
            return None

    def should_skip_ident(n, parent) -> bool:
        # attributes: obj.attr -> skip right side
        if parent and parent.type == "attribute":
            attr = field(parent, "attribute")
            if attr and _node_equals(attr, n):
                return True
        # keyword arg name
        if parent and parent.type == "keyword_argument":
            nm = field(parent, "name")
            if nm and _node_equals(nm, n):
                return True
        # parameter-related contexts: allow the parameter name itself when rename_params=True, skip others
        is_param_name_def = False
        # Case 1: identifier directly under the 'parameters' node (simple params like def f(x))
        if parent and parent.type == "parameters":
            is_param_name_def = True
            if not rename_params:
                return True
        if parent and parent.type in (
            "parameter",
            "typed_parameter",
            "default_parameter",
            "typed_default_parameter",
            "list_splat_pattern",
            "dictionary_splat_pattern",
        ):
            nm = field(parent, "name")
            if nm and _node_equals(nm, n):
                is_param_name_def = True
                if rename_params:
                    # allow renaming at definition site
                    pass
                else:
                    return True
        # any identifier inside the parameters list (defaults, annotations, etc.) should be skipped,
        # except the parameter name itself when rename_params is True
        anc = parent
        while anc is not None:
            if anc.type in (
                "parameters",
                "parameter",
                "typed_parameter",
                "default_parameter",
                "typed_default_parameter",
                "list_splat_pattern",
                "dictionary_splat_pattern",
            ):
                if not is_param_name_def or not rename_params:
                    return True
                else:
                    break
            anc = parents.get(id(anc))
        # skip type annotation contexts entirely
        anc = parent
        while anc is not None:
            if anc.type in (
                "type",
                "type_parameter",
                "type_argument_list",
                "type_alias_statement",
                "type_conversion",
            ):
                return True
            anc = parents.get(id(anc))
        # function/class names
        if parent and parent.type in ("function_definition", "class_definition"):
            nm = field(parent, "name")
            if nm and _node_equals(nm, n):
                return True
        # imports
        p = parent
        while p is not None:
            if p.type in ("import_statement", "import_from_statement"):
                return True
            if p.type in ("global_statement", "nonlocal_statement"):
                return True
            # walk up via external parents map
            p = parents.get(id(p))
        # builtins/keywords/reserved
        name = code_bytes[n.start_byte : n.end_byte].decode("utf-8")
        if name in BUILTINS_SET or name in PY_KEYWORDS or name in RESERVED_NAMES:
            return True
        return False

    # build parent links using external dict (Node objects are not writable)
    parents: Dict[int, Any] = {}

    def build_parents(n, parent=None):
        parents[id(n)] = parent
        for c in n.children:
            build_parents(c, n)

    build_parents(root)

    def visit(n):
        t = n.type
        # push scopes
        if t == "module":
            scope_stack.append(pre_list.pop(0))
        elif t == "function_definition":
            scope_stack.append(pre_list.pop(0))
        elif t in (
            "list_comprehension",
            "set_comprehension",
            "dictionary_comprehension",
            "generator_expression",
        ):
            scope_stack.append(pre_list.pop(0))

        # rename identifier uses
        if t == "identifier":
            parent = parents.get(id(n))
            if not should_skip_ident(n, parent):
                old = code_bytes[n.start_byte : n.end_byte].decode("utf-8")
                new = nearest_mapping(old)
                if new and new != old:
                    replacements.append((n.start_byte, n.end_byte, new))
                    transforms.append(
                        {
                            "type": "var_rename",
                            "original_name": old,
                            "new_name": new,
                            "format_preserved": True,
                        }
                    )

        for c in n.children:
            visit(c)

        # pop scopes
        if t in (
            "module",
            "function_definition",
            "list_comprehension",
            "set_comprehension",
            "dictionary_comprehension",
            "generator_expression",
        ):
            scope_stack.pop()

    visit(root)

    # apply replacements from right to left
    if not replacements:
        result_code = code_bytes.decode("utf-8")
    else:
        replacements.sort(key=lambda x: x[0], reverse=True)
        b = bytearray(code_bytes)
        for start, end, newtxt in replacements:
            b[start:end] = newtxt.encode("utf-8")
        result_code = b.decode("utf-8")

    # unwrap if needed
    if is_wrapped and result_code.strip().startswith("def __snippet__():"):
        lines = result_code.strip().split("\n")
        body = []
        for line in lines[1:]:
            body.append(line[4:] if line.startswith("    ") else line)
        result_code = "\n".join(body)

    return result_code, transforms


def _apply_parentheses_ts(code: str, layers: int) -> Tuple[str, List[Dict[str, Any]]]:
    """在赋值语句的左右两侧加括号，并对 RHS 外层再套多层括号。

    规则:
    - 仅当 layers > 0 时生效。
    - 处理 node.type in ("assignment", "augmented_assignment")。
    - 对简单 LHS 标识符 (identifier) 包一层括号: a -> (a)。若已有括号仍可再次嵌套，满足多层需求。
    - RHS 表达式整体加 layers 层括号: expr -> ((expr)) (layers=2)。
    - 不解析内部优先级，不拆分复杂结构；保证语义等价(括号只增加分组)。
    - 使用与变量重命名相同的字节区间右到左替换策略以保持其它文本格式。
    """
    if layers <= 0:
        return code, []
    parser = _get_parser()
    if parser is None:
        return code, []
    try:
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
    except Exception:
        return code, []

    root = tree.root_node
    replacements: List[Tuple[int, int, str]] = []
    transforms: List[Dict[str, Any]] = []

    def paren_n(text: str) -> str:
        for _ in range(layers):
            text = f"({text})"
        return text

    def paren_once(text: str) -> str:
        return f"({text})"

    def field(node, name: str):
        try:
            return node.child_by_field_name(name)
        except Exception:
            return None

    def visit(n):
        t = n.type
        if t in ("assignment", "augmented_assignment"):
            left = field(n, "left")
            right = field(n, "right")
            # 左侧: 只对简单 identifier 包一层括号
            if left is not None and left.type == "identifier":
                old_txt = code_bytes[left.start_byte:left.end_byte].decode("utf-8")
                new_txt = paren_once(old_txt)
                replacements.append((left.start_byte, left.end_byte, new_txt))
                transforms.append(
                    {
                        "type": "paren_wrap",
                        "side": "lhs",
                        "original": old_txt,
                        "new": new_txt,
                        "layers": 1,
                    }
                )
            # 右侧: 整体加 layers 层
            if right is not None:
                old_txt = code_bytes[right.start_byte:right.end_byte].decode("utf-8")
                new_txt = paren_n(old_txt)
                replacements.append((right.start_byte, right.end_byte, new_txt))
                transforms.append(
                    {
                        "type": "paren_wrap",
                        "side": "rhs",
                        "original": old_txt[:200],  # 截断避免过长
                        "new": new_txt[:200],
                        "layers": layers,
                    }
                )
        for c in n.children:
            visit(c)

    visit(root)
    if not replacements:
        return code, []
    # 右到左应用
    replacements.sort(key=lambda x: x[0], reverse=True)
    b = bytearray(code_bytes)
    for start, end, newtxt in replacements:
        b[start:end] = newtxt.encode("utf-8")
    return b.decode("utf-8"), transforms


def remove_prompt_from_code(combined_code: str, prompt_norm: str) -> str:
    """Remove the prompt prefix by line count using normalized prompt.

    This is robust even if the prompt region was transformed (e.g., identifiers renamed),
    since we always concatenated as prompt + "\n" + code and we only need the line count.
    """
    lines = combined_code.split("\n")
    cnt = len(prompt_norm.split("\n"))
    if cnt <= 0 or cnt > len(lines):
        return combined_code
    return "\n".join(lines[cnt:])


def process_ts_replacements(
    file_path: str,
    output_path: str = "treesitter_replacement_results.json",
    style: str = "capitalize",
    rename_params: bool = False,
    mode: str = RenameMode.STYLE,
    underscore_prob: float = 0.25,
    underscore_repeat: int = 1,
    seed: int = 12345,
    var_prob: float = 1.0,
    paren_layers: int = 0,
):
    print("Step 1: Reading outputs from file...")
    # sanitize var_prob
    try:
        var_prob = max(0.0, min(1.0, float(var_prob)))
    except Exception:
        var_prob = 1.0
    outputs = get_outputs(file_path)

    print("Step 2: Concatenating and validating TS...")
    valid_entries = concatenate_and_validate(outputs)
    if not valid_entries:
        print("No valid entries found!")
        return

    results = []
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

    print("Step 3-5: Applying Variable Renaming (Tree-sitter) and processing...")
    rng = random.Random(seed)
    for entry in valid_entries:
        out_code, out_trans = _apply_renaming_ts(
            entry["output_combined"],
            style,
            rename_params=rename_params,
            mode=mode,
            underscore_prob=underscore_prob,
            underscore_repeat=underscore_repeat,
            rng=rng,
            var_prob=var_prob,
        )
        if paren_layers > 0:
            out_code, extra = _apply_parentheses_ts(out_code, paren_layers)
            out_trans.extend(extra)
        sol_code, sol_trans = _apply_renaming_ts(
            entry["solution_combined"],
            style,
            rename_params=rename_params,
            mode=mode,
            underscore_prob=underscore_prob,
            underscore_repeat=underscore_repeat,
            rng=rng,
            var_prob=var_prob,
        )
        if paren_layers > 0:
            sol_code, extra2 = _apply_parentheses_ts(sol_code, paren_layers)
            sol_trans.extend(extra2)

        stats["output_transformations"]["total_applied"] += len(out_trans)
        stats["solution_transformations"]["total_applied"] += len(sol_trans)
        stats["output_transformations"]["entries_with_transforms"] += (
            1 if out_trans else 0
        )
        stats["output_transformations"]["entries_without_transforms"] += (
            0 if out_trans else 1
        )
        stats["solution_transformations"]["entries_with_transforms"] += (
            1 if sol_trans else 0
        )
        stats["solution_transformations"]["entries_without_transforms"] += (
            0 if sol_trans else 1
        )

        if out_trans and sol_trans:
            stats["successful_replacements"]["both"] += 1
        elif out_trans and not sol_trans:
            stats["successful_replacements"]["output_only"] += 1
        elif not out_trans and sol_trans:
            stats["successful_replacements"]["solution_only"] += 1
        else:
            stats["successful_replacements"]["neither"] += 1

        out_final = remove_prompt_from_code(out_code, entry["prompt_norm"])
        sol_final = remove_prompt_from_code(sol_code, entry["prompt_norm"])

        results.append(
            {
                "index": entry["index"],
                "original": {
                    "prompt": entry["prompt"],
                    "output": entry["output"],
                    "solution": entry["solution"],
                },
                "transformed": {"output": out_final, "solution": sol_final},
                "transformations": {
                    "output_transforms": out_trans,
                    "solution_transforms": sol_trans,
                },
                "replacement_success": {
                    "output_transformed": bool(out_trans),
                    "solution_transformed": bool(sol_trans),
                    "total_transforms": len(out_trans) + len(sol_trans),
                },
            }
        )

    final_results = {"statistics": stats, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")

    print("\n" + "=" * 60)
    print("TREE-SITTER VARIABLE RENAMING STATISTICS")
    print("=" * 60)
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total entries processed: {stats['total_entries']}")
    print(f"  Valid entries (Tree-sitter parseable): {stats['valid_entries']}")
    print(f"  Skipped entries (invalid Tree-sitter): {stats['skipped_entries']}")
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
        f"  Average transformations per entry: {stats['output_transformations']['total_applied']/max(1, stats['valid_entries']):.2f}"
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
        f"  Average transformations per entry: {stats['solution_transformations']['total_applied']/max(1, stats['valid_entries']):.2f}"
    )

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

    print(f"\n🔍 EXAMPLES OF SUCCESSFUL VARIABLE RENAMING (Tree-sitter):")
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
                for t in result["transformations"]["output_transforms"]:
                    if t.get("type") == "var_rename":
                        print(f"  - {t['original_name']} → {t['new_name']}")
            if result["transformations"]["solution_transforms"]:
                print("\nSolution transformations:")
                for t in result["transformations"]["solution_transforms"]:
                    if t.get("type") == "var_rename":
                        print(f"  - {t['original_name']} → {t['new_name']}")
            print(f"\nOriginal output:\n{result['original']['output']}")
            print(f"\nTransformed output:\n{result['transformed']['output']}")
            if result["replacement_success"]["solution_transformed"]:
                print(f"\nOriginal solution:\n{result['original']['solution']}")
                print(f"\nTransformed solution:\n{result['transformed']['solution']}")
    else:
        print("No successful variable renaming examples found.")


def generate_statistics_csv(results: List[Dict], csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
        description="Apply scope-aware Variable Renaming using Tree-sitter (format-preserving)"
    )
    parser.add_argument("file_path", help="Path to the JSON file")
    parser.add_argument(
        "--output",
        default="treesitter_replacement_results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--mode",
        choices=[RenameMode.STYLE, RenameMode.UNDERSCORE, RenameMode.NONE],
        default=RenameMode.STYLE,
        help="Renaming mode: 'style' uses --style; 'underscore' randomly inserts '_'; 'none' disables renaming",
    )
    parser.add_argument(
        "--style",
        choices=[
            VariableRenameStyle.CAPITALIZE,
            VariableRenameStyle.CAMEL,
            VariableRenameStyle.SNAKE,
        ],
        default=VariableRenameStyle.CAPITALIZE,
        help="Variable renaming style (effective when --mode=style)",
    )
    parser.add_argument(
        "--underscore-prob",
        type=float,
        default=0.25,
        help="Probability of inserting '_' at a boundary (when --mode=underscore)",
    )
    parser.add_argument(
        "--underscore-repeat",
        type=int,
        default=1,
        help="Number of '_' to insert each time (1..repeat chosen randomly if repeat>1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for underscore insertion reproducibility",
    )
    parser.add_argument(
        "--var-prob",
        type=float,
        default=1.0,
        help="Probability to rename a given variable (when --mode=underscore). 0..1",
    )
    parser.add_argument(
        "--rename-params",
        action="store_false",
        help="Also rename function parameters (e.g., 'offset' -> 'Offset')",
    )
    parser.add_argument(
        "--paren-layers",
        type=int,
        default=0,
        help="Wrap assignment RHS with this many layers of parentheses and LHS identifiers with one (0=disable)",
    )
    args = parser.parse_args()

    process_ts_replacements(
        args.file_path,
        args.output,
        style=args.style,
        rename_params=args.rename_params,
        mode=args.mode,
        underscore_prob=args.underscore_prob,
        underscore_repeat=args.underscore_repeat,
        seed=args.seed,
        var_prob=args.var_prob,
        paren_layers=args.paren_layers,
    )
