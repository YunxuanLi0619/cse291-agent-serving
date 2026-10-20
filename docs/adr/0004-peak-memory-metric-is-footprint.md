# 峰值内存主口径用 macOS `footprint`,RSS 仅作次口径

在 macOS 上,压缩内存不计入 RSS:同一配置不同次启动的空闲 RSS 为 689 / 4678 / 3577 MB,负载下 RSS 在 592 → 2536 → 170 MB 间剧烈起伏,不能代表占用。`footprint`(物理占用)在同样条件下稳定。因此峰值内存 = 采样期间 `footprint` 的最大值;RSS 同时记录,但在报告里标注不可靠的原因。

`footprint` 口径是下界:默认 mmap 时约 300 MiB 权重是文件映射、不计入;`-lm none` 的 footprint 高约 294 MB(4083 vs 3789 MB)。主实验用默认 mmap,报告里同时给出两个数。`footprint` 单次采样约 0.2–0.5 s,采样间隔不低于 0.5 s。

## Consequences

- 这条只适用于 macOS。监控把"读取进程内存"做成可替换的接口(macOS:`footprint`;Linux:`/proc/<pid>/status`),`manifest.json` 和数据带平台标签,不同平台的数据不混画一张图。
