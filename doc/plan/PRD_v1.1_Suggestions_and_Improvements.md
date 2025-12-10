# Live Risk Management System PRD v1.1 - 改进建议与分析

**文档版本**: 1.0  
**日期**: 2025-11-24  
**审阅者**: CX  
**文档性质**: 战略分析与技术建议

---

## 目录

1. [执行摘要](#执行摘要)
2. [架构层面改进](#架构层面改进)
3. [功能模块优化](#功能模块优化)
4. [性能与可扩展性](#性能与可扩展性)
5. [风险控制增强](#风险控制增强)
6. [运维与监控](#运维与监控)
7. [技术债务与长期规划](#技术债务与长期规划)
8. [MVP优先级调整](#mvp优先级调整)
9. [实施路线图修订](#实施路线图修订)

---

## 执行摘要

### 总体评价

PRD v1.1 在 v1.0 基础上做了重要的增强，特别是在数据完整性（MDQC）、仓位对账（Reconciliation）和场景分析（Scenario Shocks）方面。然而，当前设计存在**过度工程化**和**功能蔓延**的风险，可能导致MVP交付延迟和系统复杂度失控。

### 关键发现

**优势**：
- ✅ 数据质量控制体系完善（MDQC）
- ✅ 分离了Adapter与Risk Engine，可测试性强
- ✅ 细粒度风控（按到期日分桶、软硬限制分层）
- ✅ 混合仓位管理（手工+IBKR）

**问题**：
- ⚠️ 希腊字母计算模型过于复杂（IB → BSM → Bachelier三层回退）
- ⚠️ Suggester模块试图解决优化问题，超出监控职责范围
- ⚠️ What-if Simulator增加了状态管理复杂度
- ⚠️ Phase 1功能过多，交付风险高

### 核心建议

**建议1**: 削减MVP范围，聚焦"准确观测+及时告警"  
**建议2**: 推迟Suggester优化逻辑到v1.2  
**建议3**: 优先建立稳定的数据管道和核心计算引擎  
**建议4**: 增强生产环境运维能力（监控、日志、恢复）

---

## 架构层面改进

### 1. 模块化分层重构

**现状问题**：  
当前PRD虽然提出了分离Adapter和Risk Engine，但缺乏清晰的分层架构定义，容易导致职责混淆。

**改进建议**：

```
┌─────────────────────────────────────────┐
│       Presentation Layer (Dashboard)     │  ← Terminal UI / Future Web UI
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│      Application Layer (Orchestrator)    │  ← Main Loop, Workflow Control
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│        Domain Layer (Risk Engine)        │  ← Core Business Logic
│  - RiskCalculator                        │
│  - RuleEngine                            │
│  - Reconciler                            │
│  - ShockEngine                           │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│    Infrastructure Layer (Adapters)       │  ← External Integrations
│  - IbAdapter (IBKR)                      │
│  - FileLoader (YAML/CSV)                 │
│  - Logger (JSON Logging)                 │
└─────────────────────────────────────────┘
```

**具体措施**：

1. **创建Orchestrator**: 负责主循环、定时任务调度、错误恢复
2. **Domain层纯净化**: Risk Engine不依赖任何Adapter具体实现，仅依赖接口
3. **适配器可插拔**: 通过依赖注入，未来可轻松接入其他券商（Futu, TDA）

**代码示例**：

```python
# domain/interfaces.py
from abc import ABC, abstractmethod
from typing import List
from models import Position, MarketData

class PositionProvider(ABC):
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        pass

class MarketDataProvider(ABC):
    @abstractmethod
    async def get_market_data(self, symbols: List[str]) -> List[MarketData]:
        pass

# infrastructure/ib_adapter.py
class IbAdapter(PositionProvider, MarketDataProvider):
    async def get_positions(self) -> List[Position]:
        # IBKR specific implementation
        pass
    
    async def get_market_data(self, symbols: List[str]) -> List[MarketData]:
        # IBKR specific implementation
        pass

# domain/risk_engine.py
class RiskEngine:
    def __init__(self, 
                 position_provider: PositionProvider,
                 market_data_provider: MarketDataProvider):
        self.position_provider = position_provider
        self.market_data_provider = market_data_provider
    
    async def compute_snapshot(self) -> RiskSnapshot:
        positions = await self.position_provider.get_positions()
        market_data = await self.market_data_provider.get_market_data(...)
        # Core risk calculation logic
```

---

### 2. 事件驱动架构引入

**现状问题**：  
当前设计基于轮询（每30秒刷新），无法快速响应市场异动。

**改进建议**：  
引入事件驱动机制，在关键事件发生时立即触发计算。

**事件类型定义**：

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class EventType(Enum):
    POSITION_CHANGED = "position_changed"
    MARKET_DATA_UPDATED = "market_data_updated"
    LIMIT_BREACHED = "limit_breached"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"

@dataclass
class Event:
    event_type: EventType
    timestamp: datetime
    payload: dict
    severity: str  # INFO, WARNING, CRITICAL
```

**事件处理器注册**：

```python
class EventBus:
    def __init__(self):
        self._handlers = {}
    
    def subscribe(self, event_type: EventType, handler: callable):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def publish(self, event: Event):
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            await handler(event)

# Usage
event_bus = EventBus()
event_bus.subscribe(EventType.LIMIT_BREACHED, send_alert)
event_bus.subscribe(EventType.LIMIT_BREACHED, log_breach)
```

**优势**：
- 解耦各模块，降低模块间直接调用
- 支持多个响应动作（告警、日志、自动对冲等）
- 便于后续增加复杂工作流

---

### 3. 配置管理分级

**现状问题**：  
所有配置混在一个YAML文件中，难以区分系统配置、业务配置、环境配置。

**改进建议**：  
分层配置体系，支持覆盖和继承。

**配置结构**：

```
config/
├── base.yaml              # 基础配置（开发+生产共享）
├── dev.yaml               # 开发环境覆盖
├── prod.yaml              # 生产环境覆盖
├── risk_limits.yaml       # 风控限制（业务配置）
├── instruments.yaml       # 标的物特殊配置（保证金、手续费）
└── secrets.yaml           # 敏感信息（API密钥，不入库）
```

**配置加载逻辑**：

```python
import yaml
from typing import Dict, Any

class ConfigManager:
    def __init__(self, env: str = "dev"):
        self.config = self._load_config(env)
    
    def _load_config(self, env: str) -> Dict[str, Any]:
        base = self._read_yaml("config/base.yaml")
        env_config = self._read_yaml(f"config/{env}.yaml")
        risk_limits = self._read_yaml("config/risk_limits.yaml")
        
        # Deep merge
        config = {**base, **env_config}
        config['risk_limits'] = risk_limits
        return config
    
    def get(self, key_path: str, default=None):
        # Support dot notation: "ibkr.host"
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key)
            if value is None:
                return default
        return value
```

**优势**：
- 环境切换简单（dev/prod一行参数）
- 风控限制可独立版本管理
- 敏感信息隔离，安全性提升

---

## 功能模块优化

### 1. 希腊字母计算简化（关键改进）

**现状问题**：  
PRD v1.1提出了三层回退机制（IB → BSM → Bachelier），这在MVP阶段增加了不必要的复杂度：

- BSM需要准确的无风险利率曲线和股息率数据
- Bachelier模型对于正常市场条件下不如BSM
- 本地计算的Greeks缺乏隐含波动率输入时会产生误导性结果

**改进建议**：  
MVP阶段仅使用IBKR提供的Greeks，当数据缺失时标记为"数据不可用"而非使用可能不准确的本地计算。

**决策树**：

```
IBKR Greeks可用？
├─ YES → 使用IBKR Greeks ✓
└─ NO  → 标记为 DATA_MISSING，触发告警 ⚠️
         （不进行本地计算）
```

**实施方案**：

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class GreekSource(Enum):
    IB = "ib"
    MISSING = "missing"

@dataclass
class Greeks:
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    source: GreekSource = GreekSource.MISSING
    last_update: Optional[datetime] = None

class GreekCalculator:
    def get_greeks(self, position: Position, market_data: MarketData) -> Greeks:
        # Try IBKR first
        if market_data.ib_greeks_available:
            return Greeks(
                delta=market_data.delta,
                gamma=market_data.gamma,
                vega=market_data.vega,
                theta=market_data.theta,
                source=GreekSource.IB,
                last_update=market_data.timestamp
            )
        
        # If missing, return MISSING indicator
        self.logger.warning(f"Greeks missing for {position.symbol}")
        return Greeks(source=GreekSource.MISSING)
```

**后续演进路径**（v1.2+）：

当系统稳定后，再考虑增加本地计算能力：

1. 建立完善的市场数据基础设施（利率曲线、股息率、历史波动率）
2. 实现本地Greeks计算库，严格测试精度
3. 仅在用户明确配置时启用本地计算作为备选

**配置示例（未来版本）**：

```yaml
greeks:
  primary_source: ib
  fallback_enabled: false  # MVP默认关闭
  fallback_sources:
    - bsm
    - bachelier
  fallback_conditions:
    - ib_stale_sec: 30
    - ib_missing: true
```

---

### 2. Suggester模块重新定位

**现状问题**：  
当前Suggester试图做两件事：
1. 告诉用户风险来源（哪个仓位贡献了Delta）
2. 建议如何对冲（Hedge Efficiency Scoring, Cross-Asset Hedging）

第2项属于交易优化问题，而非风险监控职责。

**改进建议**：  
MVP阶段的Suggester仅做"诊断"，不做"处方"。

**简化后的Suggester职责**：

```python
@dataclass
class BreachDiagnostics:
    breached_metric: str  # e.g., "portfolio_delta"
    current_value: float
    limit_value: float
    breach_severity: str  # SOFT / HARD
    top_contributors: List[ContributorInfo]  # Top 5 positions driving the metric

@dataclass
class ContributorInfo:
    symbol: str
    contribution: float  # Contribution to the metric
    percentage: float    # % of total metric
    position_size: int
    suggestion: str      # Simple text: "Consider reducing TSLA position"

class SimpleSuggester:
    def diagnose_breach(self, 
                        snapshot: RiskSnapshot, 
                        breach: Breach) -> BreachDiagnostics:
        # Identify top contributors
        contributors = self._find_top_contributors(
            snapshot, 
            breach.metric_name
        )
        
        return BreachDiagnostics(
            breached_metric=breach.metric_name,
            current_value=breach.current_value,
            limit_value=breach.limit_value,
            breach_severity=breach.severity,
            top_contributors=contributors[:5]  # Top 5
        )
    
    def _find_top_contributors(self, 
                               snapshot: RiskSnapshot, 
                               metric: str) -> List[ContributorInfo]:
        if metric == "portfolio_delta":
            # Sort positions by absolute delta contribution
            sorted_positions = sorted(
                snapshot.positions,
                key=lambda p: abs(p.delta * p.quantity * p.multiplier),
                reverse=True
            )
            # Convert to ContributorInfo
            return [
                ContributorInfo(
                    symbol=p.symbol,
                    contribution=p.delta * p.quantity * p.multiplier,
                    percentage=...,
                    position_size=p.quantity,
                    suggestion=f"Consider adjusting {p.symbol} position"
                )
                for p in sorted_positions
            ]
```

**显示效果**：

```
⚠️  SOFT BREACH: Portfolio Delta
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current:  +42,500   Limit: ±50,000   (85% utilized)

Top Contributors:
┌─────────┬───────────────┬──────────┬──────────┐
│ Symbol  │ Delta Contrib │ % Total  │ Position │
├─────────┼───────────────┼──────────┼──────────┤
│ TSLA    │ +28,400       │ 66.8%    │ +500     │
│ NVDA    │ +12,100       │ 28.5%    │ +200     │
│ AAPL    │ +2,000        │  4.7%    │ +100     │
└─────────┴───────────────┴──────────┴──────────┘

💡 Suggestion: TSLA dominates delta exposure. 
   Consider reducing long equity or adding short delta hedge.
```

**优势**：
- 清晰展示风险来源，便于快速决策
- 避免实现复杂的优化算法
- 后续可逐步增强（v1.2引入Hedge Optimizer作为独立模块）

---

### 3. What-if Simulator推迟到v1.2

**理由**：

1. **状态管理复杂**：需要维护"真实状态"和"假设状态"的分离
2. **非核心需求**：属于pre-trade工具，而系统核心是post-trade监控
3. **MVP聚焦**：先确保实时监控准确无误

**建议**：

将What-if Simulator移至v1.2，MVP阶段专注于：
- 稳定的数据获取
- 准确的风险计算
- 及时的告警触发

**未来设计预留**：

在架构设计时预留接口，方便后续集成：

```python
# domain/interfaces.py
class RiskSimulator(ABC):
    @abstractmethod
    def simulate_trade(self, 
                       base_snapshot: RiskSnapshot, 
                       hypothetical_trade: Trade) -> RiskSnapshot:
        """Return a new snapshot reflecting the hypothetical trade"""
        pass
```

---

### 4. 场景分析（Scenario Shocks）简化

**现状问题**：  
PRD中提出了多种Shock类型：
- Spot Shocks (±%)
- IV Shocks (绝对 / 相对)
- Combined Shocks

对于MVP而言，Combined Shocks增加了实现和测试复杂度。

**改进建议**：  
MVP仅实现Spot Shocks，占据90%的场景需求。

**简化后的ShockEngine**：

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class SpotShock:
    underlying: str
    shock_pct: float  # e.g., -0.05 for -5%

@dataclass
class ShockResult:
    scenario_name: str
    shocked_metrics: Dict[str, float]  # metric_name -> shocked_value
    delta_vs_base: Dict[str, float]    # Change from baseline

class SimpleShockEngine:
    def __init__(self, shock_percentages: List[float]):
        # e.g., [-0.10, -0.05, -0.03, 0.03, 0.05, 0.10]
        self.shock_percentages = shock_percentages
    
    def run_spot_shocks(self, 
                        snapshot: RiskSnapshot) -> List[ShockResult]:
        results = []
        base_pnl = snapshot.total_pnl
        
        for shock_pct in self.shock_percentages:
            shocked_snapshot = self._apply_spot_shock(snapshot, shock_pct)
            shocked_pnl = shocked_snapshot.total_pnl
            
            results.append(ShockResult(
                scenario_name=f"Spot {shock_pct:+.1%}",
                shocked_metrics={
                    'pnl': shocked_pnl,
                    'delta': shocked_snapshot.portfolio_delta,
                    'gamma': shocked_snapshot.portfolio_gamma
                },
                delta_vs_base={
                    'pnl': shocked_pnl - base_pnl
                }
            ))
        
        return results
    
    def _apply_spot_shock(self, 
                          snapshot: RiskSnapshot, 
                          shock_pct: float) -> RiskSnapshot:
        # Create a copy and adjust prices
        shocked_snapshot = copy.deepcopy(snapshot)
        for position in shocked_snapshot.positions:
            if position.asset_type == AssetType.STOCK:
                position.mark_price *= (1 + shock_pct)
            elif position.asset_type == AssetType.OPTION:
                # Approximate option P&L change using delta and gamma
                underlying_price = snapshot.get_underlying_price(position.underlying)
                spot_move = underlying_price * shock_pct
                delta_pnl = position.delta * spot_move * position.quantity * position.multiplier
                gamma_pnl = 0.5 * position.gamma * (spot_move ** 2) * position.quantity * position.multiplier
                position.mark_price += (delta_pnl + gamma_pnl) / (position.quantity * position.multiplier)
        
        return shocked_snapshot
```

**配置简化**：

```yaml
scenarios:
  enabled: true
  spot_shocks: [-0.10, -0.05, -0.03, 0.03, 0.05, 0.10]
  # IV shocks and combined shocks deferred to v1.2
```

**未来扩展**（v1.3+）：
- IV Shocks：需要完整的Greeks重新计算
- Combined Shocks：需要建立Spot-Vol相关性模型
- Custom Scenarios：用户自定义历史事件重演

---

## 性能与可扩展性

### 1. 计算性能优化

**现状挑战**：  
随着仓位数量增长，风险计算可能成为瓶颈。

**优化策略**：

#### (a) 并行计算

使用Python的`concurrent.futures`并行计算不同标的的Greeks：

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

class ParallelRiskEngine:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def compute_snapshot(self, positions: List[Position]) -> RiskSnapshot:
        # Group positions by underlying
        by_underlying = self._group_by_underlying(positions)
        
        # Submit parallel tasks
        futures = {
            self.executor.submit(self._compute_underlying_risk, underlying, pos_list): underlying
            for underlying, pos_list in by_underlying.items()
        }
        
        # Collect results
        underlying_risks = {}
        for future in as_completed(futures):
            underlying = futures[future]
            underlying_risks[underlying] = future.result()
        
        # Aggregate
        return self._aggregate_snapshot(underlying_risks)
```

#### (b) 增量计算

仅重新计算发生变化的仓位：

```python
class IncrementalRiskEngine:
    def __init__(self):
        self.last_snapshot = None
        self.position_cache = {}  # position_id -> greeks
    
    def compute_snapshot(self, 
                         positions: List[Position],
                         changed_positions: set) -> RiskSnapshot:
        # Only recalculate changed positions
        for position in positions:
            if position.id in changed_positions or position.id not in self.position_cache:
                self.position_cache[position.id] = self._calculate_greeks(position)
        
        # Aggregate from cache
        return self._aggregate_from_cache(positions)
```

#### (c) 缓存Greeks

对于流动性差的期权，Greeks变化缓慢，可以使用缓存：

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachedGreekCalculator:
    def __init__(self, ttl_seconds: int = 60):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache = {}  # (symbol, timestamp) -> Greeks
    
    def get_greeks(self, position: Position, market_data: MarketData) -> Greeks:
        cache_key = (position.symbol, market_data.timestamp)
        
        # Check cache
        if cache_key in self.cache:
            cached_greeks, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.ttl:
                return cached_greeks
        
        # Calculate and cache
        greeks = self._calculate(position, market_data)
        self.cache[cache_key] = (greeks, datetime.now())
        return greeks
```

**性能目标**：

| 仓位数量 | 刷新延迟 | 目标 |
|---------|---------|------|
| < 100   | < 100ms | ✓    |
| 100-250 | < 250ms | ✓    |
| 250-500 | < 500ms | ✓    |
| 500+    | < 1s    | 🎯   |

---

### 2. 内存管理

**现状风险**：  
长时间运行可能导致内存泄漏，特别是缓存未正确清理时。

**改进措施**：

#### (a) 定期清理过期缓存

```python
import threading
from datetime import datetime, timedelta

class CacheCleanupManager:
    def __init__(self, cleanup_interval: int = 300):  # 5 minutes
        self.cleanup_interval = cleanup_interval
        self.caches = []
        self.cleanup_thread = None
    
    def register_cache(self, cache: dict):
        self.caches.append(cache)
    
    def start(self):
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        while True:
            time.sleep(self.cleanup_interval)
            self._cleanup_expired()
    
    def _cleanup_expired(self):
        now = datetime.now()
        for cache in self.caches:
            expired_keys = [
                k for k, (_, timestamp) in cache.items()
                if now - timestamp > timedelta(seconds=600)  # 10 min TTL
            ]
            for key in expired_keys:
                del cache[key]
            
            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
```

#### (b) 限制历史数据保留

```python
from collections import deque

class BoundedHistoryKeeper:
    def __init__(self, max_snapshots: int = 1000):
        self.snapshots = deque(maxlen=max_snapshots)
    
    def add_snapshot(self, snapshot: RiskSnapshot):
        self.snapshots.append(snapshot)
        # Old snapshots automatically dropped when maxlen exceeded
```

#### (c) 内存监控告警

```python
import psutil

class MemoryWatchdog:
    def __init__(self, warning_threshold: float = 0.80):
        self.warning_threshold = warning_threshold
    
    def check_memory(self):
        memory = psutil.virtual_memory()
        usage_pct = memory.percent / 100.0
        
        if usage_pct > self.warning_threshold:
            self.logger.warning(
                f"High memory usage: {usage_pct:.1%} "
                f"(Used: {memory.used / 1e9:.1f}GB / Total: {memory.total / 1e9:.1f}GB)"
            )
            return True
        return False
```

---

### 3. 数据持久化策略

**现状问题**：  
PRD中提到"不做历史P&L持久化"，但实际生产环境中需要基本的审计日志。

**改进建议**：  
实现轻量级的快照持久化，便于问题排查和历史回溯。

**实施方案**：

#### (a) 快照存储（SQLite）

```python
import sqlite3
from datetime import datetime
import json

class SnapshotRepository:
    def __init__(self, db_path: str = "./data/snapshots.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_pnl REAL,
                portfolio_delta REAL,
                portfolio_gamma REAL,
                portfolio_vega REAL,
                portfolio_theta REAL,
                margin_used REAL,
                margin_available REAL,
                breach_count INTEGER,
                snapshot_json TEXT,
                INDEX idx_timestamp (timestamp)
            )
        """)
        self.conn.commit()
    
    def save_snapshot(self, snapshot: RiskSnapshot):
        self.conn.execute("""
            INSERT INTO snapshots (
                timestamp, portfolio_pnl, portfolio_delta, 
                portfolio_gamma, portfolio_vega, portfolio_theta,
                margin_used, margin_available, breach_count, snapshot_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            snapshot.total_pnl,
            snapshot.portfolio_delta,
            snapshot.portfolio_gamma,
            snapshot.portfolio_vega,
            snapshot.portfolio_theta,
            snapshot.margin_used,
            snapshot.margin_available,
            len(snapshot.breaches),
            json.dumps(snapshot.to_dict())
        ))
        self.conn.commit()
    
    def get_snapshots_between(self, start: datetime, end: datetime):
        cursor = self.conn.execute("""
            SELECT * FROM snapshots 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, (start.isoformat(), end.isoformat()))
        return cursor.fetchall()
```

#### (b) 配置项

```yaml
persistence:
  enabled: true
  db_path: ./data/snapshots.db
  snapshot_interval_sec: 60  # Save every 60 seconds
  retention_days: 30  # Auto-delete snapshots older than 30 days
```

#### (c) 自动清理

```python
class SnapshotCleanup:
    def __init__(self, repository: SnapshotRepository, retention_days: int = 30):
        self.repository = repository
        self.retention_days = retention_days
    
    def cleanup_old_snapshots(self):
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted = self.repository.conn.execute("""
            DELETE FROM snapshots WHERE timestamp < ?
        """, (cutoff_date.isoformat(),))
        self.repository.conn.commit()
        self.logger.info(f"Deleted {deleted.rowcount} snapshots older than {self.retention_days} days")
```

---

## 风险控制增强

### 1. 动态限制调整

**现状问题**：  
PRD中的限制都是静态配置，无法根据市场环境动态调整。

**改进建议**：  
实现基于VIX或波动率的动态限制调整机制。

**实施方案**：

```python
from enum import Enum

class MarketRegime(Enum):
    NORMAL = "normal"         # VIX < 15
    ELEVATED = "elevated"     # 15 <= VIX < 25
    HIGH_VOL = "high_vol"     # 25 <= VIX < 35
    CRISIS = "crisis"         # VIX >= 35

class DynamicLimitManager:
    def __init__(self, base_limits: dict):
        self.base_limits = base_limits
        self.regime_multipliers = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.ELEVATED: 0.85,
            MarketRegime.HIGH_VOL: 0.70,
            MarketRegime.CRISIS: 0.50
        }
    
    def get_adjusted_limits(self, current_vix: float) -> dict:
        regime = self._determine_regime(current_vix)
        multiplier = self.regime_multipliers[regime]
        
        adjusted_limits = {}
        for key, base_value in self.base_limits.items():
            if isinstance(base_value, (int, float)):
                adjusted_limits[key] = base_value * multiplier
            else:
                adjusted_limits[key] = base_value
        
        self.logger.info(
            f"Market regime: {regime.value} (VIX: {current_vix:.1f}), "
            f"Limit multiplier: {multiplier:.2f}"
        )
        return adjusted_limits
    
    def _determine_regime(self, vix: float) -> MarketRegime:
        if vix < 15:
            return MarketRegime.NORMAL
        elif vix < 25:
            return MarketRegime.ELEVATED
        elif vix < 35:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.CRISIS
```

**配置示例**：

```yaml
risk_limits:
  dynamic_adjustment:
    enabled: true
    vix_symbol: VIX
    check_interval_sec: 300  # Check every 5 minutes
  
  # Base limits (applied when VIX < 15)
  base_limits:
    max_total_gross_notional: 5000000
    portfolio_delta_range: [-50000, 50000]
    portfolio_gamma_range: [-5000, 5000]
```

---

### 2. 仓位集中度监控增强

**现状问题**：  
PRD仅监控单个标的的Notional集中度，未考虑行业/板块集中度。

**改进建议**：  
增加行业维度的风险聚合和限制。

**实施方案**：

```python
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SectorExposure:
    sector: str
    gross_notional: float
    net_notional: float
    delta: float
    vega: float
    positions: List[Position]

class SectorRiskAnalyzer:
    def __init__(self, sector_mapping: Dict[str, str]):
        # symbol -> sector mapping
        self.sector_mapping = sector_mapping
    
    def analyze_sector_risk(self, 
                           snapshot: RiskSnapshot) -> Dict[str, SectorExposure]:
        sector_groups = {}
        
        for position in snapshot.positions:
            sector = self.sector_mapping.get(position.underlying, "UNKNOWN")
            
            if sector not in sector_groups:
                sector_groups[sector] = {
                    'positions': [],
                    'gross_notional': 0,
                    'net_notional': 0,
                    'delta': 0,
                    'vega': 0
                }
            
            notional = abs(position.quantity * position.mark_price * position.multiplier)
            signed_notional = position.quantity * position.mark_price * position.multiplier
            
            sector_groups[sector]['positions'].append(position)
            sector_groups[sector]['gross_notional'] += notional
            sector_groups[sector]['net_notional'] += signed_notional
            sector_groups[sector]['delta'] += position.delta * position.quantity * position.multiplier
            sector_groups[sector]['vega'] += position.vega * position.quantity * position.multiplier
        
        return {
            sector: SectorExposure(
                sector=sector,
                gross_notional=data['gross_notional'],
                net_notional=data['net_notional'],
                delta=data['delta'],
                vega=data['vega'],
                positions=data['positions']
            )
            for sector, data in sector_groups.items()
        }
```

**配置文件**：

```yaml
# instruments.yaml
sector_mapping:
  TSLA: "EV_AUTO"
  NVDA: "SEMICONDUCTORS"
  AMD: "SEMICONDUCTORS"
  AAPL: "TECH_HARDWARE"
  MSFT: "SOFTWARE"
  GOOGL: "INTERNET"
  META: "INTERNET"
  SPY: "INDEX_ETF"
  QQQ: "INDEX_ETF"

sector_limits:
  max_sector_gross_notional:
    default: 1500000
    SEMICONDUCTORS: 2000000
    INDEX_ETF: 3000000
  
  max_sector_concentration_pct: 0.40  # Max 40% of portfolio in one sector
```

---

### 3. 相关性风险监控

**现状问题**：  
当前设计忽略了仓位间的相关性，可能低估组合风险。

**改进建议（v1.3+）**：  
引入相关性矩阵，计算组合的真实风险暴露。

**概念设计**：

```python
import numpy as np
from typing import Dict

class CorrelationRiskEngine:
    def __init__(self, correlation_matrix: np.ndarray, symbols: List[str]):
        self.corr_matrix = correlation_matrix
        self.symbols = symbols
        self.symbol_index = {sym: i for i, sym in enumerate(symbols)}
    
    def compute_portfolio_variance(self, 
                                   position_deltas: Dict[str, float]) -> float:
        """
        Compute portfolio variance considering correlations
        Var(Portfolio) = w^T * Corr * w
        where w is the vector of position deltas
        """
        # Build weight vector
        n = len(self.symbols)
        weights = np.zeros(n)
        for symbol, delta in position_deltas.items():
            if symbol in self.symbol_index:
                weights[self.symbol_index[symbol]] = delta
        
        # Compute variance
        portfolio_var = weights.T @ self.corr_matrix @ weights
        return portfolio_var
    
    def compute_diversification_ratio(self, 
                                     position_deltas: Dict[str, float]) -> float:
        """
        Diversification Ratio = Sum(individual volatilities) / Portfolio volatility
        Higher ratio indicates better diversification
        """
        individual_vol_sum = sum(abs(delta) for delta in position_deltas.values())
        portfolio_vol = np.sqrt(self.compute_portfolio_variance(position_deltas))
        
        return individual_vol_sum / portfolio_vol if portfolio_vol > 0 else 1.0
```

**数据来源**（未来版本）：
- 从历史价格数据计算滚动相关性
- 订阅第三方相关性数据服务
- 使用因子模型（如Fama-French）估计相关性

---

## 运维与监控

### 1. 健康检查机制

**现状问题**：  
PRD提到了Watchdog和Heartbeat，但未详细设计健康检查体系。

**改进建议**：  
实现多层次的健康检查系统。

**实施方案**：

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    component: str
    status: HealthStatus
    last_check: datetime
    message: str
    metrics: dict = None

class HealthMonitor:
    def __init__(self):
        self.checks = {}
        self.check_interval = 10  # seconds
    
    def register_check(self, name: str, check_func: callable, interval: int = 10):
        self.checks[name] = {
            'func': check_func,
            'interval': interval,
            'last_run': None,
            'result': None
        }
    
    async def run_checks(self) -> Dict[str, HealthCheck]:
        results = {}
        now = datetime.now()
        
        for name, check_info in self.checks.items():
            last_run = check_info['last_run']
            interval = check_info['interval']
            
            # Skip if recently checked
            if last_run and (now - last_run).total_seconds() < interval:
                results[name] = check_info['result']
                continue
            
            # Run check
            try:
                result = await check_info['func']()
                check_info['result'] = result
                check_info['last_run'] = now
                results[name] = result
            except Exception as e:
                results[name] = HealthCheck(
                    component=name,
                    status=HealthStatus.UNHEALTHY,
                    last_check=now,
                    message=f"Check failed: {str(e)}"
                )
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheck]) -> HealthStatus:
        if any(r.status == HealthStatus.UNHEALTHY for r in results.values()):
            return HealthStatus.UNHEALTHY
        if any(r.status == HealthStatus.DEGRADED for r in results.values()):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

# Specific health checks
class IbConnectionHealthCheck:
    async def __call__(self) -> HealthCheck:
        if not self.ib_adapter.is_connected():
            return HealthCheck(
                component="ib_connection",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(),
                message="IB connection down"
            )
        
        # Check last data update
        last_update = self.ib_adapter.last_data_timestamp
        if datetime.now() - last_update > timedelta(seconds=30):
            return HealthCheck(
                component="ib_connection",
                status=HealthStatus.DEGRADED,
                last_check=datetime.now(),
                message=f"No data update for {(datetime.now() - last_update).total_seconds():.0f}s"
            )
        
        return HealthCheck(
            component="ib_connection",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            message="Connected and receiving data"
        )

class PositionReconciliationHealthCheck:
    async def __call__(self) -> HealthCheck:
        mismatches = await self.reconciler.get_mismatches()
        
        if len(mismatches) > 10:
            return HealthCheck(
                component="position_reconciliation",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(),
                message=f"{len(mismatches)} position mismatches detected"
            )
        elif len(mismatches) > 0:
            return HealthCheck(
                component="position_reconciliation",
                status=HealthStatus.DEGRADED,
                last_check=datetime.now(),
                message=f"{len(mismatches)} minor mismatches"
            )
        
        return HealthCheck(
            component="position_reconciliation",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            message="All positions reconciled"
        )
```

**Dashboard展示**：

```
┌─────────────────────────────────────────────────────┐
│ System Health Status: HEALTHY                       │
├─────────────────────────────────────────────────────┤
│ Component                Status    Last Check       │
├─────────────────────────────────────────────────────┤
│ IB Connection           ✓ HEALTHY  14:32:45         │
│ Market Data Feed        ✓ HEALTHY  14:32:44         │
│ Position Reconciliation ✓ HEALTHY  14:32:30         │
│ Risk Calculation        ✓ HEALTHY  14:32:46         │
│ Rule Engine             ✓ HEALTHY  14:32:46         │
│ Memory Usage            ⚠ DEGRADED 14:32:00 (78%)   │
└─────────────────────────────────────────────────────┘
```

---

### 2. 结构化日志增强

**现状问题**：  
PRD提到使用JSON日志，但未定义标准的日志格式和分类。

**改进建议**：  
建立统一的日志规范和字段标准。

**日志Schema**：

```python
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    SYSTEM = "system"
    CONNECTION = "connection"
    POSITION = "position"
    RISK = "risk"
    BREACH = "breach"
    RECONCILIATION = "reconciliation"
    PERFORMANCE = "performance"

@dataclass
class LogEntry:
    timestamp: str
    level: str
    category: str
    message: str
    component: str
    context: dict = None
    error: dict = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

class StructuredLogger:
    def __init__(self, name: str):
        self.name = name
    
    def log(self, 
            level: LogLevel, 
            category: LogCategory, 
            message: str,
            context: dict = None,
            error: Exception = None):
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            level=level.value,
            category=category.value,
            message=message,
            component=self.name,
            context=context or {},
            error={'type': type(error).__name__, 'message': str(error)} if error else None
        )
        
        print(entry.to_json())
        
        # Also send to log aggregation service (future)
        # self._send_to_aggregator(entry)
    
    def info_breach(self, breach: Breach):
        self.log(
            LogLevel.WARNING,
            LogCategory.BREACH,
            f"{breach.severity} breach detected: {breach.metric_name}",
            context={
                'metric': breach.metric_name,
                'current_value': breach.current_value,
                'limit_value': breach.limit_value,
                'severity': breach.severity,
                'utilization_pct': breach.utilization_pct
            }
        )
    
    def info_position_change(self, old_qty: int, new_qty: int, symbol: str):
        self.log(
            LogLevel.INFO,
            LogCategory.POSITION,
            f"Position changed: {symbol}",
            context={
                'symbol': symbol,
                'old_quantity': old_qty,
                'new_quantity': new_qty,
                'delta_quantity': new_qty - old_qty
            }
        )
```

**日志输出示例**：

```json
{
  "timestamp": "2025-11-24T14:32:45.123Z",
  "level": "WARNING",
  "category": "breach",
  "message": "SOFT breach detected: portfolio_delta",
  "component": "RiskEngine",
  "context": {
    "metric": "portfolio_delta",
    "current_value": 42500,
    "limit_value": 50000,
    "severity": "SOFT",
    "utilization_pct": 85.0
  }
}
```

**日志查询便利化**：

```bash
# 查找所有BREACH事件
cat logs/risk_system.log | jq 'select(.category=="breach")'

# 统计过去1小时的WARNING数量
cat logs/risk_system.log | jq 'select(.level=="WARNING")' | wc -l

# 查找特定标的的所有事件
cat logs/risk_system.log | jq 'select(.context.symbol=="TSLA")'
```

---

### 3. 告警系统集成

**现状问题**：  
PRD仅提到Terminal Display，未考虑外部告警通道。

**改进建议（v1.2）**：  
集成多渠道告警系统。

**架构设计**：

```python
from abc import ABC, abstractmethod
from typing import List

class AlertChannel(ABC):
    @abstractmethod
    async def send_alert(self, alert: Alert):
        pass

@dataclass
class Alert:
    severity: str  # INFO, WARNING, CRITICAL
    title: str
    message: str
    timestamp: datetime
    context: dict

class TerminalAlertChannel(AlertChannel):
    async def send_alert(self, alert: Alert):
        # Display in terminal with rich formatting
        console.print(f"[bold red]🚨 {alert.title}[/bold red]")
        console.print(alert.message)

class TelegramAlertChannel(AlertChannel):
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
    
    async def send_alert(self, alert: Alert):
        # Send via Telegram Bot API
        import aiohttp
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': f"🚨 {alert.title}\n\n{alert.message}",
            'parse_mode': 'Markdown'
        }
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload)

class SlackAlertChannel(AlertChannel):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_alert(self, alert: Alert):
        # Send via Slack Webhook
        import aiohttp
        payload = {
            'text': f"🚨 {alert.title}",
            'blocks': [
                {
                    'type': 'section',
                    'text': {'type': 'mrkdwn', 'text': alert.message}
                }
            ]
        }
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=payload)

class AlertManager:
    def __init__(self):
        self.channels: List[AlertChannel] = []
    
    def register_channel(self, channel: AlertChannel):
        self.channels.append(channel)
    
    async def send_alert(self, alert: Alert):
        # Filter by severity (e.g., only send CRITICAL to Telegram)
        for channel in self.channels:
            if self._should_send(channel, alert):
                await channel.send_alert(alert)
    
    def _should_send(self, channel: AlertChannel, alert: Alert) -> bool:
        # Example filtering logic
        if isinstance(channel, TelegramAlertChannel):
            return alert.severity == "CRITICAL"
        return True
```

**配置示例**：

```yaml
alerts:
  enabled: true
  channels:
    - type: terminal
      enabled: true
    
    - type: telegram
      enabled: true
      bot_token: ${TELEGRAM_BOT_TOKEN}  # From environment
      chat_id: ${TELEGRAM_CHAT_ID}
      min_severity: CRITICAL
    
    - type: slack
      enabled: false
      webhook_url: ${SLACK_WEBHOOK_URL}
      min_severity: WARNING
```

---

## 技术债务与长期规划

### 1. 代码质量保障

**措施**：

#### (a) 类型检查（mypy）

```bash
# pyproject.toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
strict = true

# Run type checking
mypy risk_system/
```

#### (b) 代码格式化（black + isort）

```bash
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 100

# Auto-format
black risk_system/
isort risk_system/
```

#### (c) 代码质量检查（flake8 + pylint）

```bash
# .flake8
[flake8]
max-line-length = 100
exclude = .git,__pycache__,venv
ignore = E203,W503

# Run linting
flake8 risk_system/
pylint risk_system/
```

#### (d) 单元测试覆盖率（pytest + coverage）

```bash
# Run tests with coverage
pytest --cov=risk_system --cov-report=html

# Enforce minimum coverage
pytest --cov=risk_system --cov-fail-under=85
```

---

### 2. 文档体系建设

**建议**：

#### (a) API文档（Sphinx）

```python
# 使用docstring标准格式
class RiskEngine:
    """
    Core risk calculation engine.
    
    This engine aggregates positions, computes portfolio Greeks,
    and evaluates risk metrics against configured limits.
    
    Args:
        position_provider: Source of position data
        market_data_provider: Source of market data
        config: Risk configuration
    
    Example:
        >>> engine = RiskEngine(ib_adapter, ib_adapter, config)
        >>> snapshot = await engine.compute_snapshot()
        >>> print(f"Portfolio Delta: {snapshot.portfolio_delta}")
    """
```

#### (b) 架构文档（ADR - Architecture Decision Records）

```markdown
# ADR-001: Use IBKR Greeks Only for MVP

## Status
Accepted

## Context
We need to calculate portfolio Greeks. Multiple options exist:
1. IBKR-provided Greeks
2. Local BSM calculation
3. Multi-model fallback chain (IB → BSM → Bachelier)

## Decision
For MVP, use IBKR Greeks exclusively. Mark as MISSING if unavailable.

## Rationale
- Simplifies implementation
- Avoids complex model maintenance
- IBKR Greeks are generally reliable for liquid options
- Local calculations require extensive market data infrastructure

## Consequences
- Fast MVP delivery
- Clear data quality indicators
- May need to address illiquid options in v1.2
```

#### (c) 运维手册（Runbook）

```markdown
# Runbook: Handle IB Connection Failure

## Symptoms
- Dashboard shows "IB connection down"
- No position updates
- Stale market data warnings

## Investigation Steps
1. Check IB Gateway/TWS is running
2. Verify port configuration (default: 7497 for paper, 4001 for live)
3. Check network connectivity: `ping 127.0.0.1`
4. Review logs for connection errors

## Resolution
1. Restart IB Gateway/TWS
2. Verify API access is enabled in TWS settings
3. Restart risk system: `python main.py`

## Prevention
- Enable IB Gateway auto-reconnect
- Monitor IB Gateway process health
```

---

### 3. 技术栈升级路径

**当前栈**：
- Python 3.10+
- ib_async
- pandas, numpy
- rich (terminal UI)
- YAML/JSON config

**未来演进**：

| 版本 | 技术升级 | 理由 |
|------|---------|------|
| v1.1 | 引入FastAPI (可选) | 为Web API做准备 |
| v1.2 | Redis缓存 | 提升性能，支持分布式 |
| v1.3 | PostgreSQL/TimescaleDB | 时序数据存储 |
| v2.0 | React前端 | 替代Terminal UI |
| v2.5 | Kubernetes部署 | 容器化、高可用 |

---

## MVP优先级调整

基于上述分析，建议对PRD v1.1的MVP范围进行调整：

### 保留功能（Core MVP）

| 功能 | 优先级 | 理由 |
|------|-------|------|
| Position Management (FR-101~104) | P0 | 核心基础 |
| Position Reconciliation (FR-105) | P0 | 数据完整性关键 |
| Market Data Quality Check (FR-302~305) | P0 | 防止垃圾数据 |
| Real-time P&L (FR-201~204) | P0 | 核心价值 |
| Greek Aggregation (FR-301, 仅IB来源) | P0 | 风险监控核心 |
| Notional Calculation (FR-303) | P0 | 杠杆监控 |
| Margin Monitoring (FR-306) | P0 | 防止强平 |
| Rule Engine (FR-401~403) | P0 | 告警核心 |
| Terminal Dashboard (FR-701~703) | P0 | 可视化 |
| IbAdapter with reconnect (FR-601~604) | P0 | 连接稳定性 |

### 简化功能（Simplified MVP）

| 功能 | 原v1.1设计 | 简化方案 | 推迟至 |
|------|-----------|---------|--------|
| Greeks计算 | IB→BSM→Bachelier | 仅IB，缺失标记 | v1.2 |
| Scenario Shocks | Spot+IV+Combined | 仅Spot Shocks | v1.2 |
| Suggester | Efficiency Scoring | 仅Top Contributors | v1.2 |
| Expiry Buckets | Full细分 | 简化为0DTE vs Others | v1.1 (可保留) |

### 推迟功能（Defer to v1.2+）

| 功能 | 推迟理由 | 目标版本 |
|------|---------|---------|
| What-if Simulator (FR-800) | 属于Pre-trade工具，非监控核心 | v1.2 |
| Cross-Asset Hedging (FR-504) | 优化问题，超出监控范围 | v1.3 |
| Multi-Model Greeks (FR-309) | 需要复杂市场数据基础设施 | v1.2 |
| Combined Shocks (FR-308) | 实现和测试复杂度高 | v1.2 |

---

## 实施路线图修订

### Week 1: 核心基础设施

**目标**: 数据流畅通，基本架构就位

**任务**:
1. 创建项目结构和模块划分
   ```
   risk_system/
   ├── models/          # Data classes
   ├── domain/          # Business logic
   ├── infrastructure/  # Adapters
   ├── application/     # Orchestrator
   └── config/          # Configuration
   ```

2. 实现核心Data Models
   ```python
   # models/position.py
   @dataclass
   class Position:
       symbol: str
       underlying: str
       asset_type: AssetType
       # ... (完整字段见PRD)
   
   # models/market_data.py
   @dataclass
   class MarketData:
       symbol: str
       bid: float
       ask: float
       last: float
       timestamp: datetime
       ib_greeks: Optional[Greeks]
   
   # models/risk_snapshot.py
   @dataclass
   class RiskSnapshot:
       timestamp: datetime
       positions: List[Position]
       total_pnl: float
       portfolio_delta: float
       # ... (完整字段见PRD)
   ```

3. 实现IbAdapter骨架
   ```python
   # infrastructure/ib_adapter.py
   class IbAdapter:
       async def connect(self):
           pass
       
       async def get_positions(self) -> List[Position]:
           pass
       
       async def get_market_data(self, symbols: List[str]) -> List[MarketData]:
           pass
   ```

4. 实现配置加载
   ```python
   # infrastructure/config_manager.py
   config = ConfigManager(env="dev")
   ```

5. 设置结构化日志
   ```python
   # infrastructure/logging.py
   logger = StructuredLogger("RiskSystem")
   ```

**验收标准**:
- [ ] 能连接到IBKR Paper账户
- [ ] 能获取账户中的仓位列表
- [ ] 能订阅市场数据并接收行情
- [ ] 日志正确输出到JSON文件

---

### Week 2: 风险计算与对账

**目标**: 准确计算组合风险，检测仓位差异

**任务**:
1. 实现RiskEngine核心计算
   ```python
   # domain/risk_engine.py
   class RiskEngine:
       def compute_snapshot(self, 
                           positions: List[Position],
                           market_data: Dict[str, MarketData]) -> RiskSnapshot:
           # P&L calculation
           # Greek aggregation
           # Notional calculation
           pass
   ```

2. 实现Reconciler
   ```python
   # domain/pos_reconciler.py
   class Reconciler:
       def reconcile(self,
                    ib_positions: List[Position],
                    manual_positions: List[Position],
                    cached_positions: List[Position]) -> ReconciliationResult:
           # Detect MISSING, DRIFT, STALE
           pass
   ```

3. 实现MDQC基础规则
   ```python
   # domain/mdqc.py
   class MarketDataQualityCheck:
       def validate(self, market_data: MarketData) -> List[DataQualityIssue]:
           # Bid/Ask validation
           # Stale data detection
           # Zero price detection
           pass
   ```

4. 单元测试
   ```python
   # tests/test_risk_engine.py
   def test_pnl_calculation():
       # Test long stock P&L
       # Test short option P&L
       # Test multi-leg P&L
       pass
   
   def test_greek_aggregation():
       # Test delta sum
       # Test gamma sum
       pass
   ```

**验收标准**:
- [ ] 能正确计算股票和期权的P&L
- [ ] 能正确聚合组合Greeks（仅使用IB Greeks）
- [ ] 能检测IBKR与手工仓位的差异
- [ ] 能标记Stale和Zero价格数据
- [ ] 单元测试覆盖率 > 85%

---

### Week 3: 限制监控与Dashboard

**目标**: 实现告警功能和可视化界面

**任务**:
1. 实现RuleEngine
   ```python
   # domain/rule_engine.py
   class RuleEngine:
       def evaluate(self, 
                   snapshot: RiskSnapshot,
                   limits: RiskLimits) -> List[Breach]:
           # Check each limit
           # Classify OK / SOFT / HARD
           pass
   ```

2. 实现SimpleSuggester
   ```python
   # domain/suggester.py
   class SimpleSuggester:
       def diagnose_breach(self, 
                          snapshot: RiskSnapshot,
                          breach: Breach) -> BreachDiagnostics:
           # Find top contributors
           # Generate simple suggestions
           pass
   ```

3. 实现Terminal Dashboard
   ```python
   # tui/terminal_dashboard.py
   class TerminalDashboard:
       def render(self, snapshot: RiskSnapshot, breaches: List[Breach]):
           # Portfolio summary panel
           # Positions table
           # Breach alerts
           # System health
           pass
   ```

4. 实现Watchdog
   ```python
   # infrastructure/watchdog.py
   class Watchdog:
       async def monitor(self):
           # Check snapshot freshness
           # Check connection health
           # Check memory usage
           pass
   ```

**验收标准**:
- [ ] 能正确检测Soft/Hard Breach
- [ ] Dashboard能清晰展示所有关键信息
- [ ] 能识别并展示风险贡献最大的仓位
- [ ] Watchdog能检测系统异常

---

### Week 4: 集成测试与优化

**目标**: 端到端验证，生产准备

**任务**:
1. **Paper Trading Soak Test** (关键)
   - 在IBKR Paper账户开仓（股票+期权）
   - 运行系统4小时以上
   - 验证P&L准确性（对比TWS）
   - 验证Greeks聚合（手工计算vs系统计算）

2. **Breach Simulation Test**
   - 临时降低风控限制，触发SOFT breach
   - 进一步降低限制，触发HARD breach
   - 验证告警正确触发
   - 验证Suggester诊断准确

3. **Reconnection Test**
   - 运行系统后，关闭IB Gateway
   - 验证系统检测到连接断开
   - 验证自动重连机制
   - 验证数据恢复正常

4. **Reconciliation Test**
   - 在TWS中手工平仓某个仓位
   - 在YAML手工文件中保留该仓位
   - 验证Reconciler检测到MISSING差异
   - 修正YAML文件后验证差异消失

5. **Performance Profiling**
   - 使用`py-spy`或`cProfile`分析性能瓶颈
   - 优化慢速函数
   - 验证100个仓位时刷新延迟 < 100ms

6. **文档完善**
   - README with Quick Start
   - 配置文件模板和说明
   - 常见问题FAQ

**验收标准**:
- [ ] 能正确处理50+ positions
- [ ] P&L计算误差 < 0.1%（vs TWS）
- [ ] 自动重连成功率100%（测试10次）
- [ ] 无内存泄漏（8小时运行后内存增长 < 20%）
- [ ] 文档完整，新用户可独立部署

---

## 总结与行动建议

### 关键要点

1. **MVP聚焦**: 削减v1.1中的过度设计，专注"准确监控+及时告警"
2. **数据质量优先**: MDQC和Reconciliation是基础，必须稳固
3. **延迟优化功能**: Suggester优化、What-if Simulator推迟到v1.2
4. **架构可扩展**: 分层设计、事件驱动、依赖注入为未来演进铺路
5. **运维能力**: 健康检查、结构化日志、告警系统从Day 1建立

### 立即行动项

**第一步（今天）**：
- [ ] Review本文档，确认削减范围
- [ ] 确认技术栈选型（Python版本、依赖库）
- [ ] 创建项目仓库，初始化项目结构

**第二步（本周）**：
- [ ] 实现`models.py`（Position, MarketData, RiskSnapshot）
- [ ] 实现`IbAdapter`骨架，测试连接IBKR Paper账户
- [ ] 实现`ConfigManager`和结构化日志

**第三步（未来4周）**：
- 严格按照修订后的实施路线图执行
- 每周五Review进度，调整计划
- Week 4结束时交付可用的MVP

---

## 附录

### A. 削减前后对比

| 模块 | v1.1原设计复杂度 | 简化后复杂度 | 节省开发时间 |
|------|----------------|------------|------------|
| Greeks计算 | 3模型回退链 | 仅IB | ~3天 |
| Scenario Shocks | Spot+IV+Combined | 仅Spot | ~2天 |
| Suggester | 效率评分+优化 | 仅诊断 | ~4天 |
| What-if Simulator | 完整实现 | 推迟 | ~5天 |
| **总计** | - | - | **~14天** |

### B. 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|-------|------|---------|
| IBKR Greeks缺失率高 | 低 | 中 | 先Paper Trading验证数据质量 |
| 性能不达标 | 中 | 中 | Week 4性能测试，必要时引入缓存 |
| 仓位对账复杂度高 | 中 | 高 | 充分单元测试，逐步rollout |
| Dashboard刷新卡顿 | 低 | 低 | 异步渲染，限制表格行数 |

### C. 成功标准

**MVP视为成功当且仅当**：
- ✅ 能稳定连接IBKR并实时获取仓位
- ✅ P&L计算准确度 > 99.9%
- ✅ 能检测并告警所有Soft/Hard Breach
- ✅ 能在5秒内检测到连接断开并尝试重连
- ✅ 连续运行8小时无崩溃
- ✅ 代码测试覆盖率 > 85%
- ✅ 文档完整，可独立部署

---

**文档结束**

如需进一步讨论任何章节或需要代码实现协助，请随时提出。

Good luck with your MVP! 🚀
