# 并发 c ≤ np,排队由 runner 自己的 FIFO 负责

runner 持有 FIFO 队列,任何时刻在飞请求数 ≤ c ≤ np。runner 启动时校验 `/slots` 条数 ≥ c,否则拒绝运行。

如果 c > np,多出的请求会在 llama-server 内部排队(`requests_deferred`),排队时间不再受我们的调度器控制,"自己控制的 FIFO"和 Phase 2 的调度策略都会失效,排队时间也无法归因。
