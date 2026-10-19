# Phase 1 交接:Step 0–5.5 之后(2026-10-19)

新会话(`/grill-with-docs` 等)从这里开始读。详细数据见 `docs/lab-notebook.md`,原始数据见 `results/raw/`。

## 已定的事实(可直接写进 GLOSSARY / ADR)
- **ADR 候选:** ① CPU-only 编译(`-DGGML_METAL=OFF`);② 内存预算 B = 总 KV token 数(`-c`),字节 = B × 114,688;③ 不变量 c ≤ np;④ 峰值内存主口径 = `footprint`,RSS 仅作次口径(macOS 压缩内存使 RSS 不可信);⑤ 同一配置下内存由启动参数决定,与负载无关。
- **口径表:** 超时、deadline、goodput、冷启动、失败分类,见 lab-notebook「5.5-C」。
- **固定版本:** llama.cpp `1b29621bacc6a446bd523fc790316a24a06e886f`(b11741),模型 `Llama-3.2-3B-Instruct-Q4_K_M.gguf` sha256 `6c1a2b41…28ff`,二进制 sha256 见 `results/raw/env/env_snapshot.txt`。
- **校准数字(电池下测得):** T_solo short 4.3 / longctx 35.0 / longgen 22.5 / agent(4 步,delay 1 s)35.2 s;S̄=20.26 s;μ≈0.074 task/s;λ 网格 {0.022, 0.044, 0.067, 0.089};总机时 ≈20.6 h(依赖 E4-on、E5 的 0.9μ 复用 E1 数据)。

## 与手册(phase1-steps.md)不一致、实现时以此为准
1. `--no-mmap` 在该 commit 不存在,改用 `-lm none`(`-lm mmap|none|mlock|mmap+mlock`)。
2. 默认日志不含超参/KV buffer,需要 `-lv 4`。
3. `docs/action-guide.md` §3.3 假设"RSS 随负载增长"——已被实测推翻,作废。
4. 流式 chunk 与 `/slots` **没有 `id_slot`**:`requests.jsonl` 的该字段留空,或用 server 日志 `launch_slot_: id N | task T` 事后关联。
5. token 自检:chat template 约 +32 token,short(目标 256)实测 290;生成器以 `usage.prompt_tokens` 为准校准 ±5%。
6. `timings` 在最后一个数据 chunk(`finish_reason` 那个,`[DONE]` 之前)。`ignore_eos` 在 OpenAI 兼容接口上生效。断连 ≈0.7 s 释放 slot。
7. 新发现的参数:`--kv-unified-per-slot N`(每 slot 上下文上限),尚未试。

## Step 9 之前必须做 / 尚未验证
- [ ] **插电后复测 T_solo**(每类 3 个任务);与 5.5 偏差 >15% 就重新校准 λ 网格、deadline、预算。校准当时在电池供电、低电量模式已关。
- [ ] **线程数敏感性:** 本机 4 个性能核 + 6 个能效核,`-t 8` = 4P+4E。`-t 4` / `-t 8` / `-t 10` 的速度和稳定性没比过;若要改,必须在重新校准之前定下来(改了就要重测一切速度数字)。
- [ ] c=8 的聚合吞吐没测(预算里假设 = c=4)。
- [ ] 冷盘启动没测(所有冷启动数字都是模型已在页缓存里)。
- [ ] 共享池(`-kvu`)耗尽时 server 的实际行为未测(系列 U 要观察);`/research` 待跑。
- [ ] `/research` 的 5 个问题(见 phase1-steps.md「/research 要回答的问题」)——其中 slot 选择(LCP 相似度)、断连取消、`id_slot` 缺失已有日志证据,但没有读源码核实。
- [ ] 系统采样要用 `footprint`(约 0.2–0.5 s/次),采样间隔不要低于 0.5 s;`rss_footprint_sampler.sh` 已是这个做法。

## 环境提醒
- 批跑前:插电、关低电量模式、`caffeinate -dimsu`、记录 `pmset -g therm`。本机是 MacBook Air(无风扇),长时间满载会降频。
- 当前有两个多余的 `caffeinate -i -t 300`(非本会话启动的 5 分钟定时器)和一个 `caffeinate -dimsu`(本会话启动,批跑结束后应手动 kill)。
- git 作者邮箱是 `851489600@qq.com`,与 GitHub 账号邮箱不同;课程 PDF 在 `.gitignore` 里,不入库。
