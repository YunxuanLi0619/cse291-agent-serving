# llama.cpp 以 CPU-only 方式编译(`-DGGML_METAL=OFF`)

Phase 1 基线在 M4 笔记本上用纯 CPU 后端测量。课程文档允许"小型 AWS 实例或笔记本",没有要求 GPU(最终测量是否必须在 AWS 上尚未向助教确认)。CPU 后端下模型权重和 KV 都在进程内存里,更接近 CPU 云实例;Metal 下它们进入 GPU 统一内存,`RSS`/`footprint` 能否如实反映 KV 占用未验证(手册的担心,未实测)。

## Considered Options

- **Metal(GPU):** 预计更快(prefill 尤其明显),并发扩展性也可能更好;未实测。放弃原因:Step 4–5.5 的全部数字(内存基线、T_solo、λ 网格、deadline、20.6 h 预算)要重做,内存口径需重新验证,而 Phase 1 死线是 10/28。

## Consequences

- CPU 算力是主导约束(校准:c=4 聚合吞吐仅为 c=1 的 1.5×),这可能让内存压力在本平台上显得不如 GPU 平台突出;必须写进报告的局限性。
- 所有速度数字只对"这台 M4 + 固定 commit + 固定线程数"有效,换平台(例如 AWS)必须重做 Step 5.5。
