# Tree-sitter Variable Renaming & Parentheses Transformation Tool

`treesitter_replacement.py` 是一个基于 Tree-sitter 的**格式保持 (format-preserving)** Python 代码结构扰动与变量重命名工具，可用于：

- 生成数据增强版本（变量风格转换 / 噪声注入）
- 对模型输出与参考解同构（同一规则同步应用）
- 统计重命名/括号化变换覆盖率
- 保留原始缩进、空格、注释（不重新 pretty-print）

## ✨ 支持的功能

| 功能 | 说明 | 相关参数 |
|------|------|----------|
| 解析 + 验证 | 使用官方 `tree-sitter-python` 解析；失败的样本跳过 | — |
| 变量重命名：风格模式 | 按 `capitalize` / `camel` / `snake` 转换变量标识符 | `--mode style --style <...>` |
| 变量重命名：下划线插入模式 | 在变量核心字符间按概率插入 `_`，可重复 | `--mode underscore --underscore-prob --underscore-repeat --var-prob --seed` |
| 禁用重命名 | 不做任何变量名改写（可单独套括号） | `--mode none` |
| 形参可选重命名 | 是否包含函数参数 | `--rename-params` |
| 括号扰动 | 给赋值语句：LHS (简单标识符) 加 1 层，RHS 加 N 层括号 | `--paren-layers N` |
| Prompt 拼接/剥离 | 先拼接 `prompt + code` 做变换，再按行数移除 prompt | 内部逻辑 |
| 统计与示例 | 输出整体统计 + 部分示例 | 自动输出 |
| 结果持久化 | JSON 结构化写出 | `--output` |

## 🧩 变量重命名策略说明

### 模式选择 `--mode`
- `style`：使用词法拆分 + 重新组合（保留前导下划线）
- `underscore`：在字符边界随机插入 `_`（若一轮未插入，强制插入一次）
- `none`：跳过变量重命名，仅执行后续括号化（若启用）

### 风格模式 `--style`
- `capitalize`：首词首字母大写，其余 Camel（默认） 例：`my_var` → `MyVar`
- `camel`：小驼峰 例：`my_var` → `myVar`
- `snake`：全小写下划线 例：`myVar` → `my_var`

### 下划线插入模式参数
| 参数 | 含义 |
|------|------|
| `--underscore-prob` | 每个字符边界插入下划线的概率 (0~1) |
| `--underscore-repeat` | 每次插入 `_` 的数量上限（实际 1..repeat 随机） |
| `--var-prob` | 变量级抽样：该变量是否参加重命名（仅 underscore 模式） |
| `--seed` | 固定随机种子，保证复现 |

> 注：下划线模式保证“至少一次插入”，即便 `--underscore-prob` 很小。

### 形参重命名 `--rename-params`
默认不重命名函数参数；添加该 flag 后参数定义及使用处同步更新。

### 作用域与安全
- 仅在函数 / 推导式局部变量范围内重命名
- 跳过：内建名、关键字、`self/cls/__class__`、import 名、global/nonlocal、受保护标识符
- 不修改：属性访问右侧 (`obj.attr` 的 `attr`)、关键字参数名、类型注解内部标识符
- 冲突解决：新名字若冲突自动加 `_1/_2/...`

## 🧪 括号扰动 `--paren-layers`
对赋值语句形如：
```python
x = a + b * c
```
假设 `--paren-layers 2`，结果：
```python
(x) = ((a + b * c))
```
规则：
- LHS 仅当是简单标识符时加一层括号
- RHS 加 N 层括号
- 赋值/增量赋值 (`=`, `+=`, `-=`, ... ) 均处理
- 不对解包、属性、下标的 LHS 额外套括号（更安全保守）

## 📦 安装依赖
```bash
pip install -r requirements.txt
```
确保包含：
```
tree-sitter>=0.23.0
tree-sitter-python>=0.23.4
```

## ▶️ 基本使用
最简单运行（默认 style + capitalize）：
```bash
python treesitter_replacement.py path/to/outputs.txt
```
自定义输出文件：
```bash
python treesitter_replacement.py path/to/outputs.txt --output result.json
```

## 💡 常用组合示例
风格改为 snake：
```bash
python treesitter_replacement.py data.jsonl --mode style --style snake
```
下划线模式（强扰动）：
```bash
python treesitter_replacement.py data.jsonl \
  --mode underscore --underscore-prob 0.7 --underscore-repeat 2 --var-prob 1.0 --seed 42
```
下划线模式 + 只改 30% 变量：
```bash
python treesitter_replacement.py data.jsonl \
  --mode underscore --underscore-prob 0.5 --underscore-repeat 3 --var-prob 0.3 --seed 123
```
仅括号化（不重命名变量）：
```bash
python treesitter_replacement.py data.jsonl --mode none --paren-layers 2
```
加括号 + 参数也重命名：
```bash
python treesitter_replacement.py data.jsonl --mode style --rename-params --paren-layers 1
```

## 📁 输入文件格式要求
脚本通过 `get_outputs.py` 的 `get_outputs()` 读取文件。典型每行应为 JSON / JSONL 结构，包含字段：
```json
{"prompt": "...", "output": "...", "solution": "..."}
```
或你的 `get_outputs.py` 定义的兼容格式。无效 / 解析失败 / Tree-sitter 无法解析的样本会统计为 skipped。

## 📊 输出 JSON 结构
`--output treesitter_replacement_results.json` 示例结构：
```json
{
  "statistics": { ... },
  "results": [
    {
      "index": 0,
      "original": { "prompt": "...", "output": "...", "solution": "..."},
      "transformed": { "output": "...", "solution": "..."},
      "transformations": {
        "output_transforms": [ {"type": "var_rename", ...}, {"type": "paren_wrap", ...} ],
        "solution_transforms": [ ... ]
      },
      "replacement_success": {
        "output_transformed": true,
        "solution_transformed": true,
        "total_transforms": 5
      }
    }
  ]
}
```

## 🧮 统计字段说明
| 字段 | 含义 |
|------|------|
| total_entries | 输入总条数 |
| valid_entries | 通过 Tree-sitter 解析的条数 |
| skipped_entries | 无效或解析失败条数 |
| output_transformations / solution_transformations | 各自的累计与覆盖率 |
| successful_replacements | 两边均改 / 仅一边改 / 都未改 |
| average transformations per entry | 变换密度 |

## 🔐 语义与安全保证
- 不改变控制流和副作用执行次数（括号仅分组；变量重命名作用域安全）
- 不修改注释、缩进、空行排布
- 按字节区间替换（右到左），避免偏移错位

## ⚠️ 已知局限 / 可改进方向
| 项 | 说明 |
|----|------|
| 链式赋值显示 | 链式赋值加括号后视觉可能冗余 | 未来可选择跳过链式赋值 |
| 已有外层括号 | 当前不检测“已在最外层有括号” | 可添加 `--avoid-duplicate-paren` |
| underscore 模式强制一次插入 | 可能希望完全 0 变更 | 可加 `--allow-zero-underscore` |
| var_prob 仅作用于 underscore | style 模式未用抽样 | 可统一逻辑 |
| 参数 flag 默认逻辑 | 当前 `--rename-params` 为显式开启 | 可支持 `--no-rename-params` 反向开关 |

## 🛠️ 开发与扩展建议
- 新增 dead code / 语句块交换 / 操作数交换：可在重命名 & 括号阶段之后串联更多纯文本区间替换。
- 引入最小差异约束：限制同一函数内最大/最小重命名变量数。
- 输出 CSV：现有 `generate_statistics_csv()` 可扩展括号化单独计数。

## 🧷 示例快速对比
输入：
```python
result = a * b + c
```
命令：
```bash
python treesitter_replacement.py sample.jsonl --mode none --paren-layers 2
```
输出（对应部分）：
```python
(result) = ((a * b + c))
```

## ✅ 快速自检清单
- 安装依赖成功（`tree-sitter-python` 可导入）
- 输入文件格式正确 (`prompt/output/solution`)
- 运行后统计里 `valid_entries > 0`
- 输出 JSON 正常生成

## 🧾 许可证
（根据你项目实际 License 补充）

---
有什么想扩展的（比如 dead code 注入、操作数交换、跳过已有括号检测）可以直接提，我可以继续帮你实现。