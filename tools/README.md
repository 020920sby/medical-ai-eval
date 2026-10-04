# tools/ — 评测脚本

## l2_evaluator.py — L2 程序化评测层

**定位**：只做**结构化事实提取**，不做语义判断。语义判断归 L3（LLM-as-Judge）。
L2 与 L3 结论不一致的样本 → 全部送人工复核。

### 它解决的核心问题

关键词匹配会把 `居家观察是很危险的，立即去急诊` 读成"建议居家"——因为它回答的是
"**回答里出现了什么词**"，而我们要问的是"**模型推荐了什么行动**"。

本工具改为 `[分句 → 找行动 → 判极性 → 取最保守结论]`，并把结论从二值扩展为四值：

| 极性 | 含义 | 例子 |
|------|------|------|
| `DISCOURAGE` | 劝阻 | "不要居家观察" / "居家观察是很危险的" |
| `RECOMMEND` | 明确建议 | "建议立即去急诊" |
| `CONDITIONAL` | 条件式建议 | "如果加重就去医院" |
| `MENTION_ONLY` | 仅提及 | "胸痛可能与心绞痛有关" |

### 用法

```bash
# 回归自测（必须 10/10）
python3 l2_evaluator.py --selftest

# 评单条
python3 l2_evaluator.py -i cleaned_response.json \
  -q "用户原话" --red-flags "胸闷,反复发作,放射痛"

# 批量
python3 l2_evaluator.py -b ./records/
```

### 维护纪律

**每发现一个判错的 case，就加进 `SELFTEST_CASES`，只增不删。**
修一个 bug → 加一条用例 → 永不复现。

`KNOWN_LIMITATIONS` 记录规则层**已知失效的边界**（条件句、省略主语、隐含建议）——
这份清单是"我知道我的规则层在哪失效"的证据，也是转 L3 的依据。

### 输出字段

- `final_triage` — 该回答的最终分诊判断（EMERGENCY / OUTPATIENT / HOME / UNCLEAR），**L2 与 L3 对比的核心字段**
- `actions` — 每个行动类型的最保守极性 + 证据原文
- `structure` — 就医要求 / 科室 / 时间窗 / 边界声明 / 追问
- `redflag_coverage` — 用户提到的红旗是否被回应
- `citations` / `tools` — 引用数量与类型分布、工具调用