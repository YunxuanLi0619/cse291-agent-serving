# FIFO 的排队单位是步骤(请求),不是任务

并发名额(c)只在一次 HTTP 请求期间占用。agent 任务的某一步完成后,tool delay 期间不占名额;delay 结束的那一刻,下一步按"就绪时刻"进入 FIFO 队尾。

选它的原因:Phase 2 的候选机制之一是"prioritizing ready agent steps",基线队列里排的必须是步骤,才有可对比的东西。E3/E4 要测的现象是:步间空闲时 slot 里的 KV 被别的请求按 LCP/LRU 占走,续步丢失前缀复用;任务级 FIFO(任务占名额直到做完)下这个现象测不到。

## Consequences

- 任务延迟包含续步在队尾重新排队的时间;任务级超时包含所有排队时间。
- 续步落到哪个 slot 由 llama-server 的 slot 选择(LCP 相似度,否则 LRU)决定,runner 不指定。
