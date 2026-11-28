# Git 作者身份修正

按 LeutOrca 于2026-11-28的要求，将本工作优化历史及归档分支中使用虚拟身份的提交更正为：

`LeutOrca <161549012+LeutOrca@users.noreply.github.com>`

Author、Committer及两份新增 annotated tag 的 Tagger 均已修正。共105个提交、32个fork远端ref经独立CPU-AMD121233审核，状态PUSHED_AND_VERIFIED；GitHub已将新提交识别为LeutOrca账户。上游真实作者保留。

作者信息属于Git提交对象，因此提交号随之改变。每个提交的源码树、消息与日期保持原值；运行回执和原始数值数据不改写。旧对象仍保存在本地备份bundle及original-authors refs中，供原快照追溯。

| 原提交 | 作者修正后的同源码提交 |
|---|---|
| `13c9328e2c5e1a177e6b301078904be595d42e97` | `c61fd3353cd419124b8a35188f8b5b546e600ba3` |
| `909c6bafcb445e36505dcfb5edc8e257015ada29` | `bf8880c8e0ad0288d9cef474879322415833f98a` |

[AUTHOR-MAP.csv](AUTHOR-MAP.csv)提供105个提交的完整映射及未变化的tree对象；[REF-MAP.csv](REF-MAP.csv)记录本地分支与标签映射；[AUDIT.json](AUDIT.json)记录远端核对、备份哈希及身份。旧报告中的SHA按此映射解释，不能把作者修正误认为重新计算或改变数值资格。

本仓库后续提交继续使用上述LeutOrca身份，禁止为这些用户工作设置虚拟作者或提交者。
