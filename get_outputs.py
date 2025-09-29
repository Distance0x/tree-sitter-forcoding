import csv
import json
import ast
import argparse
import collections
import re
import textwrap
from typing import Optional, List, Dict, Any
import warnings


def get_outputs(file_path):
    outputs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过空行
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


class StatsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.var_uses = collections.Counter()
        self.var_defs = collections.Counter()
        self.func_defs = collections.Counter()
        self.func_calls = collections.Counter()

    def visit_Name(self, node):
        self.var_uses[node.id] += 1
        # 只统计赋值和删除
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.var_defs[node.id] += 1
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg):
        self.var_defs[node.arg] += 1

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.func_defs[node.name] += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.func_defs[node.name] += 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.var_defs[name] += 1

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            name = alias.asname or alias.name
            self.var_defs[name] += 1

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name:
            # node.name 在3.11之前是 str，3.11起是 ast.Name
            if isinstance(node.name, str):
                self.var_defs[node.name] += 1
            elif isinstance(node.name, ast.Name):
                self.var_defs[node.name.id] += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = self._callable_name(node.func)
        if name:
            self.func_calls[name] += 1
        self.generic_visit(node)

    def _callable_name(self, func) -> Optional[str]:
        # 提取调用名：f(...) 或 obj.f(...) → "f" 或 "obj.f"
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            cur = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
        return None


def _strip_code_fences(s: str) -> str:
    # 提取 ```python ... ``` 或 ``` ... ``` 中的第一段代码
    m = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    return m[0] if m else s


def _sanitize_whitespace(s: str) -> str:
    # 统一换行/去 BOM/替换奇怪空白/展开 Tab/去公共缩进
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.lstrip("\ufeff")
    s = s.replace("\u00a0", " ").replace("\u200b", "")  # NBSP/零宽空格
    s = s.expandtabs(4)
    s = textwrap.dedent(s)
    # 去掉文件头/尾的空行
    s = s.lstrip("\n").rstrip()
    return s


def _fix_py2_print(s: str) -> str:
    # 粗略把 "print x, y" 改为 "print(x, y)"；跳过已是 print(...)
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
    """尽量把文本规整为可解析的 Python 源码"""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = _strip_code_fences(s)
    s = _sanitize_whitespace(s)
    s = _fix_py2_print(s)
    return s


def _wrap_in_func(s: str) -> str:
    # 把碎片或有顶层缩进的代码包进函数，缓解 unexpected indent
    indented = "\n".join(("    " + ln if ln.strip() else ln) for ln in s.split("\n"))
    return "def __snippet__():\n" + indented + "\n"


# ...existing code...


def analyze_solutions(solutions: List[str]) -> StatsVisitor:
    visitor = StatsVisitor()
    skipped = 0
    repaired = 0
    for i, code in enumerate(solutions):
        norm = normalize_solution(code)
        if not norm:
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                tree = ast.parse(norm, filename=f"solution_{i}")
            visitor.visit(tree)
        except SyntaxError:
            # 兜底：包进函数再试一次
            try:
                wrapped = _wrap_in_func(norm)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SyntaxWarning)
                    tree = ast.parse(wrapped, filename=f"solution_{i}")
                visitor.visit(tree)
                repaired += 1
            except SyntaxError as e2:
                skipped += 1
                if skipped <= 10:
                    print(f"SyntaxError in solution #{i}: {e2.msg} at line {e2.lineno}")
                continue
    if repaired:
        print(f"已通过包裹函数修复 {repaired} 条片段/缩进问题")
    if skipped:
        print(f"共跳过 {skipped} 条无法解析的 solution")
    return visitor


def write_stats_csv(visitor: StatsVisitor, out_path: str):
    # 排序规则：变量按 uses 降序，函数按 defs+calls 降序
    var_keys = sorted(
        visitor.var_uses.keys() | visitor.var_defs.keys(),
        key=lambda k: (-visitor.var_uses[k], k),
    )
    func_keys = sorted(
        visitor.func_defs.keys() | visitor.func_calls.keys(),
        key=lambda k: (-(visitor.func_defs[k] + visitor.func_calls[k]), k),
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "name", "uses", "defs", "calls"])
        for k in var_keys:
            w.writerow(["variable", k, visitor.var_uses[k], visitor.var_defs[k], ""])
        for k in func_keys:
            w.writerow(["function", k, "", visitor.func_defs[k], visitor.func_calls[k]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON file")
    parser.add_argument("file_path", help="Path to the JSON file")  # 添加参数
    parser.add_argument(
        "--out", default="stats_ai.csv", help="输出 CSV 路径，默认 stats.csv"
    )
    args = parser.parse_args()
    result = get_outputs(args.file_path)
    # 按你的需求：将 prompt + solution 拼接后再进行解析与统计
    combined_codes = [f"{item['prompt']}\n{item['output']}" for item in result]
    visitor = analyze_solutions(combined_codes)
    write_stats_csv(visitor, args.out)
