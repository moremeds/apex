# Claude Code handoff — market-data MCP

日期：2026-09-23。用途：接手实施的提示；此前完成的是设计 review，不是实现或生产验收。

## Reactivation prompt

你接手 Apex 的 market-data MCP 项目。先完整读取本文件及下列设计、计划、review，
核实当前源码和环境，再通过 herd 推进 implementation。不要从零重新设计，
也不要把过去的检查当成当前生产状态。提交、push、merge、release/deploy
按用户明确授权执行；交接文件不自动授予生产操作权限。

## 一、位置与必读文件

- Repo：Apex。
- Spec worktree：项目根目录下 `.worktrees/spec-market-data-mcp`。
- 分支：`spec/market-data-mcp`。
- 交接时 HEAD：`c690792b126eddfb63eb3f53fe98207675e0318b`（2026-09-23 从 v0.1.9 快进到最新 master）。
  期间 lake 读取代码零改动；新增 PG/UW 读 REST、`_db_auth.py`、`query_timeout`(504) 等，见 design §1.1。

按顺序读：

1. 项目 CLAUDE.md / AGENTS.md、private overlay，以及其中要求的 memory index。
2. [Design](2026-09-22-market-data-mcp-design.md)。
3. [Implementation plan](../plans/2026-09-22-market-data-mcp-plan.md)。
4. [Review evidence](2026-09-22-market-data-mcp-review.md)。
5. [原 handover](2026-09-22-market-data-mcp-handover.md)。

详细契约以这些文件为准，不要只凭本提示实施。若最新用户指令与文档冲突，用户指令优先。

## 二、这个项目做什么

在 Apex 内提供私有、只读的 market-data MCP，让 Claude Code / ChatGPT 查询现有
Livewire lake。REST 和 MCP 共用 application 查询逻辑；覆盖六个资产类别：
equity、volatility、fx、cmdty、futures、rates。

范围包括发现数据、bars/yields、coverage/gaps、corporate actions、delisting、
security resolution、membership、Silver/PIT revisions。最终是 **20 个 lake tools**。
2026-09-23 用户决定去掉指标（list_indicators/get_indicator），不是 22 个或 24 个。

两个顺序 PR：

- PR1：lake 查询层、相关 REST 接口、完整验证和 DuckDB 性能实验。
- PR2：MCP transport/auth/tools/client integration。

## 三、当前事实，不要误报

- Astra 已完成 spec/plan 修订及自己的源码复核。
- Claude Fable 5 High 和 Cursor Grok 4.7 High 均完成初审、debate/rebuttal、
  修订后 hash 对应的复核；结论是文档 APPROVE。
- 文档头部仍有 draft/review 字样，最终审核结论看 review.md。
- 尚未实现业务代码，未完成 candidate tests、PR1 macmini matrix，未提交、未合并、未部署。
- 2026-09-23 再查，spec worktree 仍在上述 HEAD，tracked status 干净。
  任务文档被 `/docs/` ignore，因此“git status 干净”不代表没有交付物。
- 主 checkout 有用户无关 config/uv.lock 修改和旧文档删除，必须保留并在接手时刷新状态。

文档交付时只 force-add 本任务指定文件，不收窄或重写全局 docs ignore。
不要因它们未跟踪就在新 worktree 中漏掉；保留原件并核对复制内容。
文档 APPROVE 不是提交、合并或生产发布授权。

已复核的 r2 SHA-256（本交接文件不属于该次 review snapshot）。
**注意：** 下表是 r2 快照。2026-09-23 的改动已由 r3 复审 APPROVE（Codex Astra + Cursor Grok
4.7 High，3 轮），r3 快照 hash 与结论见 review.md「2026-09-23 delta review (r3)」。
r3 之后唯一改动是本段文字。

| 文件 | SHA-256 |
|---|---|
| design | `f0ec2a649c464a39798d2895d6bd828c750b4a1151421d6de9f181c3472e2e74` |
| plan | `1fb0b78a5991615f5a3533bc1df0fb789641910a5496fcd9f55d955203381435` |
| 原 handover | `1152953bea850a8338a5ef4ed660596449097ba1d421dbe90c0510c3f20fb645` |

## 四、用户明确要求，不能降级

1. **Implementation 必须用 herd。** 你负责明确 worker 的文件范围、依赖、验收条件，
   然后审查、集成。按依赖顺序推进；worker 说完成不等于验收通过。
   不让多个 worker 并行修改相同文件，不丢弃别人的改动。
   原计划由 Astra 主导；接手执行不取消独立 review 和最终验收责任。
2. **Cursor Grok 用 4.7。** 启动后核实实际模型，不能静默退回 4.6。
3. **PostgreSQL 全部移出本项目**，另有 follow-up task。
   不增加 PG pool、数据库发现/查询、signals/regime 数据库工具。
   不破坏现有 REST 的数据库功能——master 已有 `/v1/db/*`、`/v1/uw/*` 和其 read pools，
   MCP 与 LakeServices 不碰它们。
4. **DuckDB 保留，catalog 保留，性能工作不能删除。** PR1 必须在 macmini 实测：
   A 现有 per-call connection；B lifespan parent + 每 worker 独立 cursor；
   C 确有收益时的 bounded cache。用相同查询语义、完整验证逻辑和真实输入比较，
   按计划阈值选择。不强推 LakeDb/LRU，也不能未经测量就宣布不需要优化。
   durable catalog 继续 read-only/open-per-call，支持原子替换。
5. **PR1 必须在 macmini 上运行 candidate，覆盖所有资产类别及计划定义的所有参数组合，
   完成后才允许进入 PR2。** 不用 laptop tests、旧生产 API、sampling/pairwise 或
   fixtures 替代。每个组合要有独立值级 oracle 和持久化结果。
   FAIL / NOT_RUN / BLOCKED_DATA / BLOCKED_DEPENDENCY 必须为零才能通过。
   无数据不是 N/A，正确拒绝也不能替代正向成功案例。
   2026-09-23 用户再次确认：**完整叉积**，不采用按维度覆盖（去掉指标后 equity 约 4.7 万格，
   为计算值）。PIT 由用户授权 livewire1 session 在生产 lake 发布一次 sp500/ndx100，
   已发布：两份均 PARTIAL。用户 2026-09-23 同意：PROVEN 不再是矩阵取值（apex 只回显状态，
   两者无代码路径差异），PR1 gate 不再依赖 PROVEN。

## 五、最容易做错的技术点

- 复用 reference.py，不新建重复 corporate-action reader。
- 保持 listing=listed|delisted|any 和 dual union 的现有 REST 契约。
- Historical Silver/PIT pin 只支持 daily equity/listed；当前 adjusted intraday
  可保留，但不得宣称 immutable history。
- pin_snapshot 返回 request-local copy；bars、bulk 必须用同一 pin，
  并显式传递有效 price_mode。
- PIT 不只是截断 end：必须按 manifest 的 member/session scope 过滤。apex **不重放血缘**
  （用户 2026-09-23：apex 只读，没有理由重复 livewire 的 verify）。只做：按编号解析
  manifest、路径包含检查、对要返回的 Silver artifact 做 sha256（同现有 revisions.py）、
  原样回显 publisher_status，永不提升 PARTIAL。
  不使用仅以 manifest 为 key 的验证缓存。
- raw Bronze 的 source_price_basis 按行保留；不为补这个字段把可变 Bronze join 到
  固定 Silver revision。
- REST 保留 legacy window/limit 语义；MCP 使用 bounded policy。
  Bulk MCP 默认 50 行，使 200 symbols × 50 不超过 10000。
- 指标整体移出本项目（用户 2026-09-23）：不加指标 tool、不改指标代码、矩阵不含指标。
  现有 REST 指标接口不动。已实测 regime_detector 经 yfinance 1.0/curl_cffi 访问 Yahoo
  （绕过 Python socket），同一真实 SPY 1d 输入下 1000 根中 140 根 regime 随网络可达性变化
  且不报错；其余 47 个指标无网络/写入。该问题归 regime/PG follow-up。
- 不新增闲置 optional REST auth；MCP auth 必须有。master 的 `_db_auth.py` 只守 PG/UW 路由，
  不要挂到 lake 路由；MCP 用独立 `APEX_MCP_API_KEY`、缺失即拒绝启动，比较方式沿用
  `secrets.compare_digest`。超时复用现有 `query_timeout`(504)，不另设 503。
- lake/security 路由注册在 `server.py` 的 "Literal namespaces must precede the asset-class
  catch-all" 区块内、`instruments_router` 之前。
- Docker 里复制 uv.lock 不等于安装遵循 lock，需验证实际依赖安装结果。

## 六、已知阻塞与历史验证

2026-09-22 的只读 macmini 检查：inspected lake root 的 catalog 存在，
`silver/pit-revisions/` 和 `repairs/unresolved.json` 不存在。

这是历史 presence check，不是 candidate run，必须刷新。
原因（2026-09-23 核实）：Livewire 已有 PIT 发布器和 `shepherd-silver publish` CLI，
但 macmini 没有任何 launchd/cron 调用它，从未发布过。PIT 按指数（sp500/ndx100）、仅日线，
且 as-of 不能早于 Silver revision 的 published_at，只能向前积累。发布属于 lake 写入，
需用户授权并由 Livewire 侧执行。
2026-09-23 已由 livewire1 发布（用户授权，apex 只读复核 hash 一致）：revision 1 = sp500、
revision 2 = ndx100，均 PARTIAL、silver_revision 77、verify 通过。PARTIAL 格现有真实样本；
**PROVEN 仍无**（原因在 corporate-action receipt 大量 unresolved，Livewire 侧未诊断），
PR1 gate 不再需要 PROVEN（见上）。注意 current.json 是跨指数的
单一指针（现指向 ndx100），适配器必须按编号读取，见 design §3.4。
livewire1 诊断（2026-09-23，只读）：PARTIAL 主因是 export 要求所有历史行都带逐行 provenance，
而 7 月旧行没有；export-only 修复后 sp500 435/458、ndx100 83/87，剩余需真实数据修复
（7 月误取消的 yahoo split 等）。ndx100 离 PROVEN 最近（4 个 symbol、23 条 split）。
源证据库只在宿主内置盘、容器看不到——apex 不重放源证据，所以无影响。
后续（同日）：livewire export 修复为 PR #144，用户已同意合并并重新发布（交给 livewire1）。revision 不可变且递增，重新发布会得到新编号
（如 3/4），rev 1/2 原样保留；rev 1/2 已无法通过 livewire verify，而 `publish()` 会先 verify
current，这一恢复问题由 livewire 解决。P0 按磁盘实际编号发现，不预设。yahoo split 修复入口
尚不存在，PROVEN 短期不可达。**上游数据缺陷**：Silver rev 77 有 45 个 symbol 的复权日线
仍含未复权的拆股跳变（apex 已只读确认 CMCSA 1999-05-06、HON 1997-09-16），apex 生产
adjusted 模式正在返回它；属 livewire 修复，见 design。如果仍无真实 PIT，PR1 正向
PIT gate 保持 BLOCKED_DATA。可以继续独立实现和测试，不能进入 PR2、伪造 manifests
或擅自发布/修复 lake。需要上游数据时向 Livewire owner/session 核实并记录具体依赖。

其他证据：

- installed pandas-market-calendars 5.2.4 的 XNYS 日历正确排除 2018-12-05 和
  2025-01-09；计划保留这两个回归案例。
- 一次现有测试收集因 isolated environment 缺 asyncpg/scipy 失败；不是通过证据，
  也不是已证明的业务回归。
- SDK API/lifespan/cancellation、当前部署 image/compose、真实数据矩阵、Tunnel 权限和
  实际 ChatGPT call 尚待验证。
- 原 reviewer panes 已关闭，不要假定旧 session 仍可用。

接手后复查命令（在 spec worktree 执行；RTK 可用时按规则使用）：

```sh
git status --short --branch
git log -1 --format='%H %s'
git check-ignore docs/superpowers/specs/2026-09-22-market-data-mcp-design.md
shasum -a 256 docs/superpowers/specs/2026-09-22-market-data-mcp-design.md docs/superpowers/plans/2026-09-22-market-data-mcp-plan.md docs/superpowers/specs/2026-09-22-market-data-mcp-handover.md
```

机器路径、tailnet 地址、凭证、部署命令从 private overlay/operator runbook 读取并只读复核，
不要写入公共 repo，也不要猜测端口可用或 tunnel 拓扑。

## 七、按顺序执行

1. **确认起点。** 核实 repo/worktree/HEAD/dirty state，读完四份核心文档及规则；
   汇报与交接的差异。既定设计无需重新 brainstorm。完成标准是明确当前基线和授权边界。
2. **通过 herd 执行 P0。** 核实源码/SDK/DuckDB 契约，完成只读 macmini inventory、
   完整 case manifest、数据缺口和性能基线。先暴露阻塞，不到 PR1 末尾才发现没有 PIT。
3. **隔离实施 PR1。** 使用项目 `.worktrees/`，按 P1.1–P1.8 的依赖推进。
   先跑最窄相关检查，每个任务审查集成后再交付依赖它的任务；不触碰无关或 frozen 子系统。
4. **在 macmini 验证真实 candidate。** 使用只读 lake mount、隔离的代码/报告位置和
   loopback；real REST routes 注入 LakeServices，禁用 production lifespan，
   不启动 PG/xenon/subscriptions，不重启或替换生产服务。
   保存 matrix.json、results.jsonl、summary.md、performance.json，绑定确切 candidate
   SHA、依赖版本及源数据 identity/hash。每个 case 结束就持久化，支持恢复。
5. **完成独立 review 和 PR1 验收。** 有未通过 cell，就报告确切组合、原因和所需解决条件，
   不缩减矩阵。PR1 gate 通过后才进入 PR2；缺数据时继续不依赖它的已授权工作。
6. **实施和验收 PR2。** 复用 PR1 查询逻辑，实现精确 20-tool 集合、HTTP/auth/lifespan、
   REST semantic parity 及实际客户端验证。merge、image release、production deployment
   和真实 ChatGPT 调用分别在获授权后执行、验收，不把一项当作其他项已经完成。

每次汇报先说结果，再列变更、实际检查结果、未闭环项。声称完成前重新检查 Git 状态、
相关测试与指定环境证据，并明确任何相对本交接的漂移。
不要把 code exists、tests pass、macmini real run、merged、deployed 混为一谈。
