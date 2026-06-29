# IEEE T-BIOM 投稿前模拟审稿报告

## Review setup

- **稿件**：*Reliability-Guided Secure Sketch for Deep Fingerprint Template Protection*
- **输入范围**：完整 11 页 PDF，包括方法、算法、全部实验、图表和参考文献。
- **评价边界**：本报告依据稿件自身公开的信息评价技术成立性、证据充分性、领域意义、可读性和 T-BIOM 投稿准备度；未进行系统性外部先验技术检索，因此不对“全球首次”作确定判断。
- **期刊契合度**：主题直接涉及指纹识别、可撤销生物特征、模板保护、纠错编码和安全评估，属于 T-BIOM 的核心兴趣范围。
- **共享主张概括**：稿件提出 RGSS，利用深度哈希网络的样本级 tanh 激活幅值选择可靠比特，将认证密钥放入这些位置，并以 BCH 纠错，从而提升 GAR–“security level”折衷。
- **可见证据基础**：
  - FVC2004 上 RS、BCH、Polar/SCL 和 RGSS 的比较；
  - CTM、StableCTM、BioHashing 与五种通道选择策略消融；
  - FVC2002、FVC2006 跨数据集实验；
  - 可撤销性、不可链接性、部分密钥泄漏、旧模板辅助攻击、可靠性感知猜测和边际熵分析；
  - 5 个随机划分下的 `k50` 比较。
- **影响置信度的缺失信息**：
  - BCH 的准确码型、缩短方式、生成矩阵/校验矩阵、`c` 的准确含义和实现代码；
  - 全部认证与攻击实验的配对协议、阈值、攻击预算及置信区间；
  - 完整存储记录 `(SHA256(s), h, c, R)` 的不可逆性和不可链接性分析；
  - 网络训练、预处理、数据增强和跨数据库协议的可复现细节。

---

## Reviewer 1：技术成立性与密码学安全

### Overall assessment

论文的问题动机清晰，样本级可靠性选择也具有直观合理性；但当前“secure sketch”构造存在可能动摇全文安全结论的根本问题。在准确说明和重新分析公开辅助数据泄漏之前，`k50` 不能被解释为安全位数，全文最核心的 GAR–security tradeoff 因而尚未成立。

### Who would be interested in the results, and why

生物特征模板保护、指纹认证、模糊提取器/安全草图以及可靠性感知纠错领域的研究者会关注该工作，因为它尝试把深度网络的不确定性直接用于模板保护，而不只是用于识别置信度估计。

### Major strengths

1. 从深度二值模板的非均匀翻转现象出发，问题定义自然。
2. RGSS 与随机、最差通道、群体翻转率和数据集级统计进行了直接消融。
3. 论文没有只报告识别准确率，还尝试覆盖可撤销性、不可链接性和辅助数据攻击。
4. 算法流程和系统图较清楚，读者能够定位公开存储项及验证过程。

### Major concerns

#### 1. 公开 BCH 信息可能使密钥安全度远低于 `k`

算法 1 明确存储 `c ← BCH.encode(s)`、`SHA256(s)`、`h` 和 `R`。如果 `c` 是系统 BCH 的公开校验比特，则它是密钥 `s` 的确定性线性函数：

`c = P s`。

因此，条件安全度应取决于 `rank(P)`，而不是直接等于 `k`：

`H∞(s | c) ≤ k - rank(P)`。

按照稿件使用的冗余近似 `r ≈ mt`，在 `k=264, m=9, t=27` 时，公开校验信息长度约为 243 位。若该映射接近满秩，剩余搜索空间最多约为 21 位，而不是 264 位。公开的 `SHA256(s)` 又为候选密钥提供了离线验证器。如果 `c` 实际上是完整 BCH 码字，问题会更严重。

上述数值取决于具体 BCH 实现和缩短方式，因此作者必须提供准确矩阵秩或严格的条件最小熵分析。但无论具体结果如何，目前直接把 `k` 标为 “Security Level” 是没有依据的。

#### 2. helper data 直接公开了 `G-k` 个模板比特

论文定义：

`h = r_e ⊕ s_pad`，

而 `s_pad[i]=0` 对所有 `i∉R`。因此：

`h[¬R] = r_e[¬R]`。

也就是说，公开 helper data 会原样泄露可靠集合之外的 `G-k` 个 StableCTM 模板比特。在 `G=512, k=264` 时，共有 248 位被直接公开。稿件没有分析这种确定性泄漏对模板重建、多记录融合、交叉匹配和 CTM 密钥泄漏后的影响。

#### 3. 完整存储记录的不可链接性没有被评价

稿件报告的 `Dsys` 主要基于可撤销模板距离，但攻击者实际可见的是 `(h,c,R,SHA256(s))`。特别是 `R` 是由样本可靠性产生并公开存储的生物相关侧信息。

更关键的是，图 13(b) 的可靠位置集合分析给出 **linkability EER = 9.3%**。稿件此前明确把 50% 视为理想不可链接状态，因此 9.3% 表示明显可区分，而不是“不是系统漏洞”。将其归因于不同用户可靠模式的反相关并不能消除可链接性；攻击者只关心该信号能否用于区分同源与异源记录。

当前 `Dsys≈0.046` 与 `R` 上的 9.3% linkability EER 指向不同结论，原因很可能是二者评价的对象不同。必须以完整公开记录为对象重新执行不可链接性评估。

#### 4. BCH 参数存在直接算术矛盾

第 IV-C 节写道：

`t = floor((511-k)/9) ≈ 56 at k=208`。

但按该公式，`k=208` 时应为 `t=33`，不是 56。第 V-I3 节对 `k=264` 又正确计算出 `t=27`。这是涉及方法可行性和核心性能解释的严重不一致，必须核对全部曲线的真实编码参数。

此外，论文称 BCH(511, k)，但模板长度为 512；RS 使用 8 位符号的 RS(255, k)，理论码长为 2040 位，而输入仅为 512 位。缩短、填充、截断和 parity 存储方式均未说明，无法判断比较是否公平或是否可复现。

#### 5. “零成功攻击”不足以证明安全

部分密钥泄漏、旧模板辅助攻击和可靠性感知猜测均只进行了 1000 次试验。0/1000 次成功仅意味着单次攻击成功率的 95% 上界约为 0.3%，不能证明密码学抵抗性。

当 `k=264` 且泄漏 90% 密钥时，未知部分仅约 26 位。评估 1000 个随机猜测与完整枚举约 `2^26` 个候选不是同一攻击预算。实验需要报告计算复杂度、查询次数、离线/在线能力以及是否利用公开 hash 和 BCH 约束。

### Technical failings that need to be addressed before the case is established

1. 给出准确 BCH 构造和实现，并计算 `I(s;c)`、`H∞(s|c,h,R)` 或保守安全下界。
2. 重新设计 helper-data 构造，避免公开未掩码模板位；优先考虑规范的 syndrome secure sketch 或 fuzzy extractor 构造。
3. 对完整公开记录而非中间模板进行不可链接性、不可逆性和多记录攻击分析。
4. 正确解释图 13(b) 的 9.3% linkability EER；如果该结果成立，当前“near-perfect unlinkability”表述必须撤回。
5. 纠正 BCH 参数，并公开每个 `k` 对应的 `(n,k,t)`、实际码率、校验长度和解码器设置。
6. 将经验攻击成功率与理论攻击复杂度分开，不再用 1000 次随机试验替代安全证明。

### Assessment against T-BIOM-oriented criteria

- **Originality**：样本级 tanh 可靠性用于通道选择具有潜在新意，但安全构造本身是否有效尚未建立。
- **Scientific importance**：如果能形成严格且安全的可靠性感知模板保护机制，领域价值较高；当前结果主要证明了可靠比特更易纠错。
- **Field readership**：对模板保护和深度生物特征研究者有明确吸引力。
- **Technical soundness**：当前不充分，公开 parity、未掩码模板位和完整记录可链接性是阻断性问题。
- **Readability**：方法叙述清楚，但“安全”“不可链接”“有效熵”等术语使用强于证据。

### Recommendation posture

**当前版本不建议直接投稿；需要重构安全定义和辅助数据方案后再作实质性预审。** 如果公开 parity 的条件熵分析确认安全空间显著坍缩，则仅修改文字和补充实验不足以解决问题。

---

## Reviewer 2：创新性、定位与主张强度

### Overall assessment

RGSS 的核心工程思想简洁：利用每个 enrollment 样本的激活幅值筛选低翻转通道。实验显示该选择优于随机和群体统计，这一点具有发表潜力。然而，论文把一个“可靠性选择带来的纠错增益”扩展表述为完整的安全模板保护突破，主张明显超过当前证据。

### Who would be interested in the results, and why

深度哈希、生物特征质量评估、纠错编码和隐私保护认证研究者会关注该工作；工程人员也可能对不重新训练网络即可改善密钥恢复率的方式感兴趣。

### Major strengths

1. 核心机制易解释且具有实际可实施性。
2. 五种通道选择策略的消融能够隔离样本级可靠性信号的贡献。
3. 跨 FVC 年份的数据集实验说明该信号可能具有一定传感器泛化能力。
4. 论文主动报告了负面结果，如 RS 和 Polar/SCL 的低性能。

### Major concerns

#### 1. RS “fundamentally unsuitable” 的结论过度泛化

论文只评价了特定 8 位符号、特定映射和特定实现下的 RS 基线。实际 RS 性能还受到交织、符号映射、缩短码、擦除信息、级联编码及错误相关性的影响。`1-(1-p)^8` 还隐含比特独立假设，而深度哈希比特可能相关。

现有证据最多支持：

> byte-aligned RS under the evaluated configuration is poorly matched to the observed dispersed bit-error pattern.

它不能支持“RS-based secure sketches are fundamentally unsuitable”或“necessary prerequisite for any effective deep-template secure sketch”。

#### 2. “oracle” 命名和解释不准确

所谓 oracle 使用测试集上的群体级逐位翻转率，本质上是拥有测试标签的 population-level baseline，而不是所有策略的理论上界。样本条件信号优于群体平均量并不违反任何上界，因此“RGSS surpasses the oracle”容易制造不必要的戏剧性。

建议改称 `test-set population flip-rate baseline`，并避免“ceiling achievable by any population-level statistic”之外更广泛的暗示。

#### 3. 可靠性假设缺乏直接校准

全文的关键科学假设是 `|tanh(z_i)|` 能预测未来同类采集中的翻转概率，但没有直接报告：

- 可靠性幅值与真实翻转概率的 calibration curve；
- Spearman/Pearson 相关或排序 AUC；
- 不同图像质量、手指、传感器和数据库下的校准稳定性；
- enrollment 与 probe 互换时的敏感性；
- 温度参数 `β=32` 对可靠性排序和校准的影响。

下游 GAR 增益支持该信号“有用”，但不足以完整解释它为何可靠，也不足以排除训练网络置信度失准。

#### 4. 现有工作定位过薄且存在引用问题

参考文献共 26 篇，近年的深度模板保护、可靠性辅助 fuzzy extractor、soft-information decoding 和生物特征不确定性研究覆盖不足。对于 2026 年拟投稿论文，“first method” 的论断不能只依靠当前 Related Work。

另外，文中称参考文献 [4] “pioneered the application of deep hashing to fingerprint template protection”，但 [4] 是 2012 年的 minutiae-based bit-string 工作，其标题和年代都不能直接支持“deep hashing”这一表述。该引用错误会削弱审稿人对文献梳理的信任。

#### 5. Table I 的二元分类过度简化

“Formal U”“Deep”“Reliability”以勾叉方式概括多类方法，却没有统一的定义和评价协议。例如 IoM 是否适用于深度特征、fuzzy commitment 是否可与可撤销前端结合，都不能由单一勾叉充分表达。该表更像宣传性定位，而不是严谨比较。

#### 6. `k50` 的实际意义有限

`k50` 是最后一个 GAR≥50% 的离散 `k`。50% GAR 通常不是可接受的认证工作点，而且 `k` 每次增加 16 位会造成量化效应。建议至少同时报告：

- `k` at GAR≥90%、95%、99%；
- 固定 FAR 下的 GAR/FRR；
- failure-to-enroll/failure-to-acquire；
- bootstrap 置信区间；
- 实际条件安全位数，而不是原始消息长度。

### Technical failings that need to be addressed before the case is established

1. 缩小 RS、BCH 必要性、oracle 上界和“first”类主张。
2. 增加可靠性预测的直接校准和稳定性实验。
3. 系统更新相关工作，并纠正 [4] 的错误定位。
4. 用实用认证工作点和经泄漏修正后的安全量替代单一 `k50`。
5. 确保所有基线具有相同公开信息、相同码长/码率、相同攻击模型和相同训练信息。

### Assessment against T-BIOM-oriented criteria

- **Originality**：可靠性引导的样本级选择可能构成清晰贡献，但“首次”尚未由文献综述建立。
- **Scientific importance**：更接近有价值的领域技术改进，而非目前表述的普适安全突破。
- **Field readership**：对 T-BIOM 读者较为相关。
- **Technical soundness**：性能趋势可信度尚可，但基线定义、安全量和主张范围需要显著收紧。
- **Readability**：叙事清楚，不过重复强调“confirm”“establish”“fundamentally”等强结论，使文章显得过度推销。

### Recommendation posture

**方法有潜力，但需要重大修改才能形成可信的创新性和意义论证。** 在安全构造问题解决后，论文应把主线集中为“sample-adaptive reliability improves error correction under a rigorously defined secure-sketch model”。

---

## Reviewer 3：实验设计、统计、泛化与可复现性

### Overall assessment

实验数量较多，图表组织也较完整，但当前评测协议缺少足够细节，若干结论由很少的试验次数或不匹配的统计方法支撑。跨数据库结果是亮点，不过目前只证明了 GAR 趋势，尚未证明安全与泛化同时成立。

### Who would be interested in the results, and why

关注指纹模板保护系统评测、跨传感器泛化和安全—准确率权衡的 T-BIOM 读者会感兴趣。该稿也可能为其他生物特征模态的可靠性筛选提供实验范式。

### Major strengths

1. 包括方法消融、前端消融、码型比较、模板长度比较和跨数据库测试。
2. 使用 subject-disjoint 划分，至少意识到训练与测试身份隔离的重要性。
3. 报告多个安全场景，而不是只给一条 ROC 曲线。
4. 图 1 的整体流程较易理解，主要实验趋势在图表中可见。

### Major concerns

#### 1. 认证协议不完整

论文没有明确说明：

- 每个身份使用哪一枚图像 enrollment；
- 其余 impressions 如何组成 genuine pairs；
- impostor pairs 如何采样；
- 一个用户是否生成多个 `s`、`κ` 和 `R`；
- GAR 曲线是否对 enrollment 图像、密钥和配对重复平均；
- FAR、EER 和 GAR 是否使用相同阈值；
- 失败解码和 hash 冲突如何计入；
- DB1–DB3 与 A/B 子集如何合并。

没有这些信息，数值无法独立复现，也难以判断配对相关性是否导致置信度过高。

#### 2. 数据集称谓和标准协议需要澄清

稿件把 330 个手指样本称为 330 subjects。生物特征数据集中 finger identity 与 human subject 不一定等价，训练/测试划分应明确按人、按手指还是按数据库条目进行。

同时，FVC 的 A/B 子集通常具有不同用途。将多个数据库和 A/B 子集汇总训练或测试是否符合标准协议，需要明确说明，并给出每个数据库/传感器的独立结果。

#### 3. 跨数据库结果被过度解释

FVC2002 和 FVC2006 仅报告 pooled `k50` 与 GAR@208。若要声称“authentication accuracy and security tradeoff are simultaneously stable”，至少还应提供：

- 各 DB/传感器的 GAR、FAR、EER；
- 每个外部数据库上的 `Dsys` 和完整记录可链接性；
- 可靠性校准和 entropy/min-entropy；
- 传感器间方差及置信区间。

较低的平均翻转率本身就会扩大可纠错区域，因此外部数据集上 +80 位并不自动等同于更强泛化。

#### 4. 五次划分的统计检验不足

`n=5` 时使用 paired t-test 并报告极小 p 值，对正态性和独立性假设非常敏感。`k50` 还是按 16 位步长离散化的统计量。

更需澄清的是：“固定 VGG-19 backbone 并重复五个 subject-disjoint split”可能有两种含义：

1. 每个 split 重新训练网络，则不应称固定 backbone；
2. 网络只在一个 split 上训练，再改变测试 split，则新的测试集合可能包含网络训练时见过的身份。

作者必须明确训练身份，并确保每个统计重复均无身份泄漏。建议使用嵌套、完全独立的重复训练或固定外部测试集，并报告 bootstrap/permutation 置信区间，而不只报告 t-test。

#### 5. 网络训练和预处理细节不足

VGG-19 在约两千量级图像上微调，但未说明图像尺寸、指纹 ROI、归一化、增强、采样器、metric loss 具体形式、类别/配对构造、早停、模型选择依据、随机种子和硬件。`β=32` 的选择也没有验证。

这些因素会直接影响 tanh 饱和度，而 tanh 幅值恰好是论文核心可靠性变量。因此它们不是次要实现细节。

#### 6. 多处表格数值不一致

- FVC2004 的 `GAR@208` 在 Table IV 为 54.3%，Table XI 为 53.9%，正文又出现 86.8%/85.6% 等不同 RGSS 数值；
- Exp. 5 报告 RGSS `k50=264`，Exp. 7 的 `G=512` 报告 256，之后用“不同 split”解释；
- 五 split 均值为 275.2，但摘要和结论突出单 split 的 264。

这些差异可以由划分造成，但稿件必须在每张表中明确 split/seed，并选择一个预先规定的主结果，避免选择性强调。

#### 7. 图表可读性

整体排版整洁，但许多图在双栏版式中字号过小，尤其是图 8、10、13、16 的图例和坐标。正式投稿前应减少子图信息密度，统一字体、颜色和曲线命名。

### Technical failings that need to be addressed before the case is established

1. 公开完整实验协议和每个实验的样本/配对数量。
2. 重新设计无身份泄漏的重复训练与测试流程。
3. 按数据库和传感器分别报告性能、安全及置信区间。
4. 使用与离散 `k50` 更匹配的 bootstrap、置换检验或配对非参数分析。
5. 统一所有主结果的 split、seed 和数值，并提供代码和配置。
6. 增加可靠性校准、质量分层、参数敏感性和失败案例分析。

### Assessment against T-BIOM-oriented criteria

- **Originality**：实验设计围绕明确的新机制展开。
- **Scientific importance**：若协议严格，结果可形成有用的模板保护经验；当前统计和协议缺口限制了结论强度。
- **Field readership**：领域相关性高。
- **Technical soundness**：趋势较完整，但可复现性和统计证据尚不足。
- **Readability**：结构清晰，图表需放大，实验设置需集中成可复现表格。

### Recommendation posture

**需要重大实验与报告修订。** 单纯补充文字不足，至少应重新执行一部分安全、统计和跨数据库实验。

---

## Cross-review synthesis

### Consensus strengths

1. 选题与 T-BIOM 高度匹配。
2. 利用 enrollment 样本的 tanh 幅值选择可靠比特，是直观、简洁且可能有价值的技术思路。
3. RGSS 相对随机和群体级统计的 GAR 增益在稿件数据中较一致。
4. 实验覆盖面和文章整体结构优于只报告单一识别性能的普通方法稿。

### Consensus technical risks

1. **最高风险：`k` 不是已经证明的安全位数。** 公开 BCH 信息和 hash 可能大幅降低密钥条件熵。
2. **helper data 直接泄露 `G-k` 个模板位。**
3. **图 13(b) 的 9.3% linkability EER 与“不可链接”结论直接冲突。**
4. BCH 参数存在 `k=208` 时 56/33 的算术矛盾，RS/BCH/Polar 实现和公平性不透明。
5. 1000 次随机攻击不能支撑密码学“抵抗性”结论。
6. 认证协议、重复划分、跨数据库评测和网络训练细节不足。

### Where emphasis differs across reviewers

- Reviewer 1 认为安全构造可能需要重做，属于阻断性问题。
- Reviewer 2 认为核心可靠性选择仍有发表潜力，但必须缩小主张并重建文献定位。
- Reviewer 3 认为现有趋势值得保留，但部分实验应按更严格协议重新运行。

### Broad-interest / significance readout

该工作对生物特征模板保护领域具有明确兴趣，但目前尚不能被视为已经建立的“安全草图突破”。最可信的现有结论是：

> 在该深度指纹表示与评测设置中，样本级 tanh 幅值可用于选择更稳定的比特，从而提高 BCH 密钥恢复成功率。

“安全位数提升 56–80 位”“近完美不可链接”“抵抗高级攻击”和“有效安全熵约 261.5 位”目前均未被充分建立。

### Most important issues to resolve before a strong T-BIOM case is established

#### P0：投稿前必须解决

1. 明确 `c` 的数学定义和 BCH 实现，计算公开信息后的真实条件安全度。
2. 重新设计或严格证明 helper-data 构造，处理 `h[¬R]=r_e[¬R]` 的直接模板泄漏。
3. 以完整存储记录评价不可链接性和不可逆性，正面处理 `R` 的 9.3% linkability EER。
4. 修正 BCH 参数和全部码型配置；若核心实验参数错误，重新运行相关实验。

#### P1：强烈建议完成

5. 用理论攻击复杂度和足够攻击预算重做 key leakage、旧模板和 helper-known 攻击。
6. 补充可靠性校准、质量分层和 `β` 敏感性。
7. 明确认证配对协议，并重新设计五次独立实验以排除身份泄漏。
8. 按传感器分别报告跨数据库准确率和安全指标。

#### P2：写作与定位

9. 更新相关工作并纠正 [4]；删除或弱化 `fundamentally`、`necessary prerequisite`、`oracle ceiling`、`first` 等未充分支持的表述。
10. 使用实际 GAR/FAR 工作点和条件安全位数替代单一 `k50`。
11. 统一不同 split 的结果，放大图中文字，减少重复结论。

### Overall pre-submission posture

**当前稿件：不建议直接提交 T-BIOM。**

它不是“语言润色后即可投稿”的状态，而是需要一次以安全构造为中心的结构性大修。若作者能够修复公开辅助数据泄漏问题、重新定义真实安全度，并在严格协议下保留明显的可靠性增益，论文仍具有较好的 T-BIOM 潜力。

---

## Risk / unsupported claims

| 稿件当前主张 | 评估 |
|---|---|
| `k` 等于 security level | **未支持**；必须扣除公开 BCH 信息、helper data、`R` 和多记录泄漏 |
| RGSS 有约 261.5 bits effective entropy | **未支持**；边际 Shannon 熵之和不是联合最小熵或攻击安全度 |
| near-perfect formal unlinkability | **部分指标支持，但完整系统不支持**；`R` 的 linkability EER=9.3% 是反证信号 |
| resistance to advanced attacks | **证据不足**；1000 trials 和未定义攻击预算不能建立密码学抵抗 |
| RS fundamentally unsuitable | **过度泛化**；只评价了一个未充分说明的配置 |
| BCH is a necessary prerequisite | **未支持**；未排除其他 bit-level、soft-decoding 或级联构造 |
| dataset-level oracle 是上界 | **命名不当**；它只是使用测试集标签的群体统计基线 |
| cross-dataset security stable | **未支持**；外部数据库主要报告 GAR/`k50`，缺少完整安全指标 |
| “first method” | **不可从当前文献综述确认** |
| 可靠比特不会更可预测 | **仅有边际熵证据**；未评价联合相关性、最小熵和公开侧信息后的可预测性 |

