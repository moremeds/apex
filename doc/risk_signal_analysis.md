-----

### Part 1: PRD v1.0 评审与优化建议

#### 1\. 缺失的“相关性风险” (Correlation Risk)

  * **问题**：目前 Portfolio 级别的 Delta/Vega 是简单加总。但如果你的持仓是 Long TSLA + Long NVDA + Long QQQ，你的实际风险（Beta-weighted Risk）比单纯加总更可怕。
  * **建议**：在 **4. Portfolio Rules** 中增加 **Beta-weighted Delta** 或 **Sector Concentration**（板块集中度）限制。
      * *Action*: 限制单一板块（如 Tech/Semis）的 Delta 占比不超过总 Delta 的 X%。

#### 2\. 数据源的“脏数据”处理 (Data Hygiene)

  * **问题**：期权买卖价差（Bid-Ask Spread）在开盘/收盘或剧烈波动时极宽。如果使用 Mid Price 触发止损，可能会因为瞬间的价差扩大而误触（False Positive）。
  * **建议**：在 **3. 规则框架** 或 **8. 技术实现** 中增加“报价过滤器”。
      * *Action*: 只有当 Bid/Ask 均有量且 Spread \< X% 时，才计算有效标记价格；或者要求连续 3 次 Ticks 满足条件才触发信号（Debounce/防抖动）。

#### 3\. “事件风险”层 (Event Risk Layer)

  * **问题**：目前的 VIX Regime 覆盖了宏观波动，但忽略了**个股财报（Earnings）**。在财报前 0-3 天，IV 会剧烈升高，Gamma 风险巨大。
  * **建议**：在 **3. 规则框架** 中增加 Layer 4：Event Calendar。
      * *Action*: 财报前 2 天禁止开 Naked Short；财报当天收盘前强制检查 Gamma 敞口。

#### 4\. 警报疲劳 (Alert Fatigue)

  * **问题**：如果 TSLA 跌破位，每秒触发一次 "CLOSE" 信号，系统会刷屏。
  * **建议**：在 **7. 风险信号** 中增加“冷却机制（Cooldown）”。
      * *Action*: 同一 Signal ID 在 5 分钟内不重复推送，除非 Severity 升级。

#### 5\. 对角套利（Diagonal）的特异性补充

  * **问题**：Diagonal 最大的风险在于“做空端被行权”或“做空端 Delta 超过做多端”（Delta Flip）。
  * **建议**：在 **5.3 Diagonal** 中细化规则。
      * *Action*: 当 Short Leg Delta \> Long Leg Delta 时，发出 Critical Alert（这就变成了裸空风险）。

-----

### Part 2: Option Risk Rule Engine – PRD v1.1 (Final)

以下是整合了上述建议的正式版本。结构已优化，适合直接导入 Confluence 或作为 PDF 发送。

-----

# Product Requirements Document: Option Risk Rule Engine (ORRE)

**Version:** 1.1
**Status:** Draft / Ready for Dev
**Date:** 2023-10-XX
**Author:** [Your Name] / Gemini

-----

## 1\. 产品愿景 (Product Vision)

构建一套**多维度、实时化、自动响应**的期权组合风险控制系统（ORRE）。旨在通过量化规则将交易员的主观风控经验转化为可执行的代码逻辑，覆盖从单笔交易到全组合的风险管理，特别是针对 TSLA/NVDA/MAG7 等高波动标的及 0DTE/末日轮策略。

**核心价值主张：**

  * **从“盯盘”解放到“监控”**：机器负责 24/7 的红线检查。
  * **消除情绪干扰**：严格执行止盈止损，拒绝“再扛一下”的心理谬误。
  * **动态适应市场**：根据 VIX 和 IV Rank 自动调整风控松紧度。

-----

## 2\. 系统范围 (Scope)

### 2.1 资产覆盖

  * **Underlying**: TSLA, NVDA, AAPL, MSFT, AMZN, GOOGL, META (MAG7), QQQ, SPY, IWM (RUT), SPX.
  * **Asset Class**: Equity Options (American), Index Options (European/Cash Settled).

### 2.2 策略类型覆盖

| 策略分类 | 具体形态 | 关键风险指标 |
| :--- | :--- | :--- |
| **Directional (单腿)** | Long Call/Put | Premium Loss, Time Decay |
| **Income (卖方)** | Short Put, Covered Call, Naked Put (Restricted) | Gamma Explosion, Tail Risk |
| **Spreads (价差)** | Vertical (Debit/Credit), Iron Condor | Max Loss, Wing Width |
| **Time Structure (期限)** | **Diagonal**, Calendar (PMCC) | IV Crush, Delta Flip |
| **Protection (保护)** | Collar, Protective Put | Cost of Carry |

-----

## 3\. 风险规则架构 (Risk Governance Framework)

系统采用**四层金字塔**防御体系，任意一层触发均生成风险信号（RiskSignal）。

  * **Layer 1 - Hard Limits (资金安全)**: 净值回撤、单笔最大亏损、保证金占用率。
  * **Layer 2 - Greeks Exposure (敞口控制)**: Delta (方向), Gamma (爆发), Vega (波幅), Theta (时间)。
  * **Layer 3 - Volatility Regime (宏观环境)**: 基于 VIX 及 IV Percentile 的动态调整系数。
  * **Layer 4 - Event Risk (事件驱动)**: 财报日 (Earnings), FOMC, CPI 数据发布日。

-----

## 4\. 组合级规则 (Portfolio-Level Rules)

### 4.1 资金回撤控制 (Drawdown & Exposure)

  * **每日回撤 (Daily DD)**: 单日浮亏 ≥ 总权益 **4%** → **Critical Alert** (停止开新仓，强制减仓)。
  * **累计回撤 (Peak-to-Valley)**: 累计回撤 ≥ **8%** → **Halt Trading** (系统熔断)。
  * **保证金红线**: 占用保证金 (Maintenance Margin) ≥ 账户净值 **70%** → Warning (禁止开卖方仓位)。

### 4.2 Greeks 限额 (Beta-Weighted 推荐)

  * **Net Delta**:
      * 计算：`Sum(Position Delta * Spot * Multiplier)`
      * 规则：每 1% 标的波动导致的 PnL 变化 ≤ 账户权益 **3%**。
  * **Net Gamma (0DTE/Gamma Squeeze 防护)**:
      * 计算：`0.5 * Sum(Gamma * (1% Spot)^2 * Multiplier)`
      * 规则：预估 1% 剧烈波动造成的 Gamma 额外亏损 ≤ 账户权益 **1.5%**。
  * **Net Vega (VIX Spike 防护)**:
      * 场景：假设 VIX 瞬间 +5 点。
      * 规则：`Vega_Total * 5` 造成的亏损 ≤ 账户权益 **3%**。
  * **板块集中度 (Concentration)**:
      * 单一板块 (如 Tech/Semis) 的 Delta 贡献度不可超过总 Delta 的 **60%**。

-----

## 5\. 单笔/策略级规则 (Position-Level Rules)

### 5.1 Long Call / Long Put (方向性)

  * **止损 (Stop Loss)**:
      * 权利金亏损 ≥ **-50%** (激进) 或 **-60%** (标准) → **Close**。
      * 剩余时间 (DTE) \< **20%** 初始期限 且 OTM → **Close/Roll**。
  * **止盈 (Take Profit)**:
      * 浮盈 ≥ **+100%** → **Reduce 50%** (收回本金)。
      * 浮盈回撤 (Trailing Stop): 达到最高浮盈后回撤 **30%** → **Close All**。

### 5.2 Short Put / Credit Spread (核心卖方)

  * **R-Multiple Stop**: 浮亏 ≥ 收取权利金的 **1.5x - 2.0x** → **Close/Roll** (严禁死扛)。
  * **Early Profit**: 浮盈 ≥ 最大利润的 **60%** (或 DTE \< 21天时达到 50%) → **Close** (释放保证金)。
  * **Delta Breach**: Short Leg Delta \> **0.30 - 0.35** → **Roll Out & Down** 或 **Hedge**。
  * **Price Action**: 标的价格击穿 `Strike - 1.5 * ATR` → 视为趋势反转，必须处理。

### 5.3 Diagonal / Calendar / PMCC (核心策略)

  * **Delta Flip (关键)**: 若 Short Leg (近端) Delta 绝对值 \> Long Leg (远端) Delta → **Critical Alert** (结构已变成净做空/Net Short，风险无限)。
  * **短腿风控**: 近端短腿 (Short Leg) 单日亏损 ≥ 当日 Theta 收入的 **2.0x** → **Close Short Leg**。
  * **IV Crush**: 若远端 Long Leg 因 IV 下降导致亏损 \> **30%** → 重新评估策略逻辑。

### 5.4 Covered Call / Collar

  * **Roll Logic**: 股价上涨穿过 Call Strike → 向上/向后滚动 (Roll Up/Out)。
  * **Downside Protection**: 股价下跌 **8-10%** → 向下移动 Put Leg 或增加 Put 仓位。

-----

## 6\. 环境与事件规则 (Regime & Event Rules)

### 6.1 波动率环境 (VIX Regime)

| 档位 | VIX 范围 | 策略限制 |
| :--- | :--- | :--- |
| **Low Vol** | \< 15 | 减少 Short Premium 仓位，止盈目标下调至 50%，多做 Debit Spread。 |
| **Mid Vol** | 15 - 25 | **标准模式**。全策略开放，适合 Iron Condor / Diagonal。 |
| **High Vol** | \> 25 | **高危模式**。禁止 Naked Short Put，强制降低 Net Short Vega，适合 Long Gamma。 |

### 6.2 事件驱动 (Event Risk) - *New in v1.1*

  * **Earnings Check**:
      * 标的财报发布前 **3天** 内：禁止新开 Short Gamma 策略 (除非是 Defined Risk)。
      * 财报当天收盘前：检查所有近端 Short Leg，若未覆盖 (Uncovered) 则强制平仓。
  * **Binary Events**: FOMC 决议前 1 小时，收窄 Gamma 敞口。

-----

## 7\. 信号与输出 (Signals & Actions)

### 7.1 数据结构

```json
RiskSignal = {
    "timestamp": "ISO8601",
    "level": "PORTFOLIO" | "POSITION",
    "symbol": "TSLA",
    "strategy_type": "DIAGONAL",
    "severity": "INFO" | "WARNING" | "CRITICAL",
    "trigger_rule": "Delta_Limit_Breach",
    "current_value": 0.45,
    "threshold": 0.30,
    "suggested_action": "ROLL_OR_CLOSE",
    "action_details": "Roll short leg to next week expiry or close for loss."
}
```

### 7.2 信号处理机制

  * **防抖动 (Debounce)**: 价格/Greeks 需在阈值外维持 **15秒** 或连续 **3个 Ticks** 才触发信号。
  * **冷却 (Cooldown)**: 相同 Rule + 相同 Symbol 的报警，**5分钟**内不重复发送（除非 Severity 升级）。

-----

## 8\. 技术实现规格 (Technical Specs)

### 8.1 核心模块

1.  **MarketDataEngine**: 负责清洗数据，过滤 `Bid/Ask Spread` 过大（\> 标的价 1%）的异常 tick。
2.  **RiskCalculator**:
      * 使用 `py_vollib_vectorized` 或 `QuantLib` 进行快速 Greeks 计算。
      * 支持 Portfolio Grouping (按 Symbol, 按 Strategy 聚合)。
3.  **RuleEvaluator**: 遍历上述所有规则。
4.  **ActionDispatcher**: 生成 JSON 信号，推送到 Slack/Telegram。

### 8.2 性能指标 (NFR)

  * **Latency**: 从行情更新到发出信号 \< **500ms**。
  * **Frequency**: 支持 **Event-Driven** (行情推送触发) 或 **Snapshot** (每 10 秒轮询) 模式。
  * **Reliability**: 数据源断连超过 1 分钟需发送 "System Health Warning"。

-----

## 9\. 路线图 (Roadmap)

  * **v1.0 (Current)**: 核心规则引擎，支持单腿/价差/组合即时 PnL 监控，Telegram 报警。
  * **v1.1 (Next)**: 增加 Event Calendar (Earnings) 过滤，增加板块集中度风险，优化 Diagonal 专用逻辑。
  * **v1.2 (Future)**: 对接 IBKR API，实现 "One-Click Hedge"（一键对冲）功能（如一键买入 SPY Put 对冲 Delta）。
  * **v2.0 (AI/ML)**: 基于历史回测，自动为不同 IV 环境推荐最优止盈止损参数 (Adaptive Thresholds)。

-----

既然确定了 **“Purely IBKR TWS API”** 这一技术约束，这对系统的架构设计有决定性的影响。IBKR API（尤其是 TWS/Gateway）功能强大但脾气“古怪”，特别是在数据订阅限制（Pacing Violations）和数据流（Streaming）的处理上。

基于这一核心约束，我更新了 PRD 的 **第 8 章（技术实现）** 和 **第 10 章（非功能需求）**，并添加了一个 **附录：IBKR 专项实施策略**。

以下是针对 IBKR TWS API 深度定制的 **PRD v1.2 更新部分**。

-----

# Option Risk Rule Engine – PRD v1.2 (IBKR Edition)

*(保留 v1.1 的前 7 章业务规则，以下章节针对 IBKR API 进行重写)*

## 8\. 技术实现 (Technical Implementation - IBKR Native)

鉴于采用纯 IBKR TWS API，系统将采用 **异步事件驱动 (Async Event-Driven)** 架构，推荐使用 `ib_insync` 库（因为它将 IBKR 繁琐的回调机制封装为了优雅的 `async/await` 协程）。

### 8.1 核心组件 (Python Modules)

1.  **IBConnector (Gateway Manager)**

      * **职责**: 维护与 TWS/IB Gateway 的连接，处理断线重连，管理 Client ID。
      * **关键特性**: 必须包含 `Watchdog` 机制，监测 TWS 是否卡死或断开。

2.  **MarketDataStream (Smart Subscription)**

      * **痛点解决**: IBKR 对实时数据流有并发限制（通常为 100 个并发 ticker）。
      * **逻辑**:
          * **Active Positions**: 始终保持 `reqMktData(snapshot=False)`（实时流）。
          * **Watchlist**: 采用轮询机制 `reqMktData(snapshot=True)`（快照），每 3-5 秒刷新一次，避免 Pacing Violation。
      * **Greeks 获取**: 直接订阅 `GenericTickList="100,101,104,106"`，优先使用 IBKR 推送的 `tickOptionComputation` (Delta, Gamma, Vega, Theta)，省去本地计算 Greeks 的算力，仅在 IB 数据缺失时本地回补。

3.  **PortfolioSync (Account Manager)**

      * **职责**: 监听 `updatePortfolio` 事件。
      * **逻辑**: 实时映射 `PortfolioItem` 对象到内部的 `RiskPosition` 模型，自动计算 Beta-weighted Delta。

4.  **RiskEngine (The Brain)**

      * **职责**: 接收 `tickPrice` 和 `tickOptionComputation` 事件，每收到一次更新，立即运行一次 `check_rules()`。

### 8.2 数据处理流程 (Pipeline)

1.  **Ingest**: `IB.wrapper.tickOptionComputation` 接收到 Greeks 更新。
2.  **Normalize**:
      * 清洗数据：若 IB 返回的 Greeks 为 `None` 或 `impliedVol > 2.0` (异常值)，标记为脏数据，使用上一次有效值或触发本地计算器。
      * Bid/Ask 过滤：若 `Bid = -1` 或 `Ask = 0`，忽略该 tick。
3.  **Evaluate**: 传入 `RiskEngine` 进行 Layer 1-4 规则判定。
4.  **Action**: 触发 Signal → 调用 `IB.placeOrder` (若开启自动交易) 或 推送 Telegram。

-----

## 10\. 非功能需求 (NFR - IBKR Specific)

### 10.1 性能与并发限制 (Pacing Control)

  * **API 限制**: 严格遵守 IBKR 的 `50 messages/second` 发送限制。
  * **解决方案**: 实现一个内部的 `MessageQueue`，对发出的请求（如下单、改单、请求数据）进行流控（Throttling）。

### 10.2 数据一致性 (Data Quality)

  * **RTH vs Outside RTH**: 明确配置 `useRTH=0` (包含盘前盘后数据)，但在盘前盘后自动放宽 Spread 相关的风控阈值（因为流动性差是常态）。
  * **Model Greeks**: 信任 IBKR Model Greeks，但需监控 `modelOptComp` 的 `optPrice` 与实际 `marketPrice` 的偏差，偏差过大说明 IB 模型失真，需报警。

-----

## 附录：IBKR 实施代码骨架 (Python + ib\_insync)

这是一个可直接运行的架构雏形，展示了如何用 `ib_insync` 处理 Greeks 流和风控检查。

```python
import asyncio
from ib_insync import *
import logging

# 1. 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptionRiskEngine:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # 风险阈值配置 (Hardcoded for demo, load from yaml in prod)
        self.MAX_PORTFOLIO_DELTA = 500  # Example
        self.STOP_LOSS_PCT = 0.60       # 60% stop loss
        
        # 状态存储
        self.positions = {}  # {conId: PortfolioItem}
        self.tickers = {}    # {conId: Ticker}

    async def connect(self):
        """连接 TWS 并注册回调"""
        try:
            await self.ib.connectAsync(self.host, self.port, self.client_id)
            logging.info("Connected to IBKR TWS")
            
            # 注册回调
            self.ib.updatePortfolioEvent += self.on_portfolio_update
            self.ib.pendingTickersEvent += self.on_market_data_update
            
            # 初始请求账户更新
            self.ib.reqAccountUpdates(True)
            
        except Exception as e:
            logging.error(f"Connection failed: {e}")

    def on_portfolio_update(self, item: PortfolioItem):
        """
        当持仓发生变化时触发 (开仓/平仓/价格变动)
        """
        self.positions[item.contract.conId] = item
        
        # 只有新出现的持仓才去请求实时数据流
        if item.contract.conId not in self.tickers:
            self.subscribe_market_data(item.contract)

    def subscribe_market_data(self, contract):
        """
        订阅行情和 Greeks
        GenericTickList 100,101,104,106 用于获取 Greeks 和 IV
        """
        logging.info(f"Subscribing to {contract.localSymbol}")
        contract = self.ib.qualifyContracts(contract)[0] # 确保合约信息完整
        
        # snapshot=False 建立持续连接
        ticker = self.ib.reqMktData(contract, '100,101,104,106', snapshot=False, regulatorySnapshot=False)
        self.tickers[contract.conId] = ticker

    def on_market_data_update(self, tickers):
        """
        核心循环：每当有行情/Greeks更新时触发
        """
        for ticker in tickers:
            self.check_position_risk(ticker)
        
        # 检查组合级风险
        self.check_portfolio_risk()

    def check_position_risk(self, ticker):
        """
        单笔持仓风控 (Layer 2)
        """
        if ticker.contract.conId not in self.positions:
            return

        position = self.positions[ticker.contract.conId]
        
        # 1. 获取 Greeks (IB 计算好的)
        greeks = ticker.modelGreeks
        if not greeks:
            return # 数据未就绪

        # 2. 检查 Delta 翻转 (针对 Diagonal)
        # 示例: 如果是 Short Leg 且 Delta > 0.4
        if position.position < 0 and greeks.delta > 0.4:
            self.fire_signal(ticker.contract, "DELTA_BREACH", f"Short leg delta {greeks.delta:.2f} > 0.4")

        # 3. 检查止损 (基于 Market Price)
        market_price = ticker.marketPrice()
        avg_cost = position.averageCost
        
        # 注意：marketPrice 可能无效
        if market_price > 0:
            pnl_pct = (market_price - avg_cost) / avg_cost if avg_cost != 0 else 0
            # 对于 Long (position > 0)
            if position.position > 0 and pnl_pct < -self.STOP_LOSS_PCT:
                 self.fire_signal(ticker.contract, "STOP_LOSS", f"Long leg hit -{self.STOP_LOSS_PCT*100}%")

    def check_portfolio_risk(self):
        """
        组合级风控 (Layer 1)
        """
        total_delta = 0
        
        for conId, ticker in self.tickers.items():
            if conId in self.positions and ticker.modelGreeks:
                qty = self.positions[conId].position
                # IB Greeks 是单股的，需要 * multiplier (通常100) * qty
                # 注意：IB modelGreeks.delta 已经是小数形式
                total_delta += ticker.modelGreeks.delta * qty * 100 
        
        # 检查 Net Delta
        # 需结合 SPX spot price 计算 Beta weighted，这里简化演示
        # logging.info(f"Portfolio Net Delta: {total_delta:.2f}")

    def fire_signal(self, contract, signal_type, reason):
        """
        发送信号
        """
        msg = f"RISK ALERT: {signal_type} | {contract.localSymbol} | {reason}"
        logging.warning(msg)
        # TODO: Send to Telegram / Execute Close Order

    async def run(self):
        await self.connect()
        # 保持运行
        while True:
            await asyncio.sleep(1)

# 启动逻辑
if __name__ == '__main__':
    engine = OptionRiskEngine()
    try:
        asyncio.run(engine.run())
    except (KeyboardInterrupt, SystemExit):
        engine.ib.disconnect()
```

### 对你的 Actionable Suggestions (基于 Pure IBKR):

1.  **IBKR "VIX Regime" 的获取**:

      * 不要试图自己去算 VIX。直接在代码中订阅 `VIX Index` (CBOE)。
      * 代码逻辑：`vix_ticker = ib.reqMktData(Index('VIX', 'CBOE'), ...)`。
      * 从 `vix_ticker.last` 获取实时 VIX 值来切换你的风控 Regime (Low/Mid/High)。

2.  **处理 "Delayed Data"**:

      * 如果你没有购买 IBKR 的 OPRA 实时数据包（很贵），TWS 会返回延时数据。
      * **Action**: 确保在 `reqMktData` 之前，你的账户有正确的订阅。如果是测试环境（Paper Trading），数据通常也是延时的。这会严重影响 0DTE 的风控。**必须确认数据订阅状态**。

3.  **对角套利 (Diagonal) 的特殊处理**:

      * 在 IBKR 中，对角套利是两个独立的 `PortfolioItem` (Long Leg 和 Short Leg)。
      * **Action**: 你需要在 Python 代码中建立一个 **"Strategy Map"**，通过 `ComboLeg` 或者简单的逻辑（如：同一个 Symbol 的所有持仓视为一组）将它们在逻辑上“粘”在一起，才能计算 Net Delta 和 Margin Impact。

这份更新后的 PRD 和代码骨架完全是为 IBKR 生态定制的，您可以直接将其作为 v1.2 版本归档。