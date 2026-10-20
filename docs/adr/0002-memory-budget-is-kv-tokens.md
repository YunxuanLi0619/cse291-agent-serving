# 内存预算 B 定义为服务启动的总 KV token 数(`-c`)

内存预算 B = llama-server 的 `-c`;换算字节 = B × 114,688 B/token(Llama-3.2-3B,f16 KV:2 × 28 层 × 8 kv head × 128 × 2 字节;B=16384 → 1792 MiB,与日志 `CPU KV buffer size` 一致)。

选它的原因:Step 5 实测 KV 在启动时一次性分配,`footprint` 在负载下几乎不变(3789 → 3818 MB),所以运行中的内存不会"涨",内存压力只能表现为"请求超出 slot 容量而失败/排队"。因此用启动时的 KV 容量做预算,用 runner 统计的 `live_tokens / B` 表示运行中的利用率。

## Consequences

- 仅当 `n_ctx / np` 是 256 的倍数时 B = `-c`;否则 llama-server 会把 `n_ctx` 向上改写(源码结论,未实测:np=3 → 16896)。系列 S 的 c ∈ {1,2,4,8}、B=16384 不受影响;换其他 np 时要以日志里实际的 `n_ctx` 为准。

- `--no-kv-unified` 时每 slot 容量 = B / np;`-kvu` 时所有 slot 共享一个容量为 B 的池(`/slots` 对每个 slot 都报 n_ctx = B)。
- 同一配置下内存由启动参数决定,与负载无关(实验发现,见 lab-notebook Step 5)。Phase 2 的准入控制针对的是 KV 池,而不是进程内存的增长。
