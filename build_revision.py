from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


OUT = Path(r"D:\PaperProject\GyxPaper2\ST-MGPrompt_revised_manuscript.docx")


def set_run_font(run, latin="Times New Roman", east="SimSun", size=10.5, bold=False, italic=False, color=None):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_para_spacing(style_or_para, before=0, after=6, line=1.25, first_line=0):
    fmt = style_or_para.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    if first_line:
        fmt.first_line_indent = Cm(first_line)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa):
    total = sum(widths_dxa)
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.insert(0, tbl_w)
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    layout = tbl_pr.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    grid = tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_keep_with_next(para, value=True):
    ppr = para._p.get_or_add_pPr()
    node = ppr.find(qn("w:keepNext"))
    if value and node is None:
        ppr.append(OxmlElement("w:keepNext"))
    elif not value and node is not None:
        ppr.remove(node)


def add_para(doc, text="", style="Normal", bold_prefix=None, italic=False, align=None, first_line=True):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        set_run_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_prefix):])
        set_run_font(r2, italic=italic)
    else:
        r = p.add_run(text)
        set_run_font(r, italic=italic)
    if style == "Normal" and first_line:
        p.paragraph_format.first_line_indent = Cm(0.74)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_number(doc, text):
    p = doc.add_paragraph(style="List Number")
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for cell, text in zip(hdr.cells, headers):
        set_cell_shading(cell, "E8EEF5")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(text))
        set_run_font(r, bold=True, size=9.5)
    for rowdata in rows:
        row = table.add_row()
        for cell, text in zip(row.cells, rowdata):
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(text))
            set_run_font(r, size=9.2)
    set_table_geometry(table, widths)
    return table


def add_equation(doc, text):
    p = doc.add_paragraph(style="Equation")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_run_font(r, latin="Cambria Math", east="Cambria Math", size=10.5)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    r = p.add_run(text)
    set_run_font(r, bold=True, size={1: 15, 2: 12.5, 3: 11.5}[level])
    set_keep_with_next(p)
    return p


doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.0)
sec.bottom_margin = Cm(2.0)
sec.left_margin = Cm(2.0)
sec.right_margin = Cm(2.0)
sec.header_distance = Cm(1.25)
sec.footer_distance = Cm(1.25)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Times New Roman"
normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
normal.font.size = Pt(10.5)
set_para_spacing(normal, after=6, line=1.25, first_line=0.74)

for name, size, before, after in [("Heading 1", 15, 18, 8), ("Heading 2", 12.5, 12, 5), ("Heading 3", 11.5, 8, 4)]:
    st = styles[name]
    st.font.name = "Times New Roman"
    st._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    st._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "SimHei")
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    set_para_spacing(st, before=before, after=after, line=1.15)

for list_style in ["List Bullet", "List Number"]:
    st = styles[list_style]
    st.font.name = "Times New Roman"
    st._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    st._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    st.font.size = Pt(10.5)
    set_para_spacing(st, after=4, line=1.2)
    st.paragraph_format.left_indent = Cm(0.95)
    st.paragraph_format.first_line_indent = Cm(-0.55)

eq_style = styles.add_style("Equation", WD_STYLE_TYPE.PARAGRAPH)
eq_style.font.name = "Cambria Math"
eq_style.font.size = Pt(10.5)
set_para_spacing(eq_style, before=3, after=5, line=1.15)

# Header and footer: quiet, journal-like furniture.
header = sec.header.paragraphs[0]
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
r = header.add_run("ST-MGPrompt | Wind power forecasting")
set_run_font(r, size=8.5, color="666666")
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = footer.add_run("ST-MGPrompt")
set_run_font(r, size=8.5, color="666666")

# Title block.
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
r = p.add_run("基于时空多粒度协同对齐的高维非平稳风电功率预测")
set_run_font(r, size=16, bold=True, east="SimHei")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(3)
r = p.add_run("多任务多粒度时空风电功率预测\nMulti-task and multi-granularity spatiotemporal wind power forecasting")
set_run_font(r, size=11.5, bold=True)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(12)
r = p.add_run("ST-MGPrompt (Spatio-Temporal Multi-Granularity Prompting Network)")
set_run_font(r, size=10.5, italic=True, color="404040")

add_heading(doc, "摘要", 1)
add_para(doc, "准确预测风电功率是提升新能源消纳能力和电网运行安全的重要基础。风电功率并不是单一风机在单一时间尺度上的独立变化，而是由历史风速、风向、机组状态、气象变量以及风机之间的空间传播共同决定。面对高频局部波动与低频长期趋势同时存在的非平稳序列，单一时间粒度容易在细节保留与趋势提取之间失衡；而直接把不同粒度序列强行对齐，也可能破坏气象序列的连续性。为此，本文构建 ST-MGPrompt，以固定双粒度表示、趋势先验自适应空间图、双向扩散时空编码、Macro-Trend Prompt、对称时空多粒度交叉融合以及 MS-MG-DWU 动态损失为主要组成。模型以过去 144 个 10 min 时间步的多变量历史观测为输入，面向 134 台风机同时输出未来 10 个时间步的功率预测，并在 H3、H6 和 H10 三个预测视界上进行评价。与依赖固定时间聚合和零填充对齐的方案不同，ST-MGPrompt 让粗粒度分支负责保持宏观演化方向，让细粒度分支负责刻画局部扰动；二者在同一 ST-MG Coupling Block 中通过双向交叉注意力相互修正。本文进一步给出统一的实验协议、组件消融、缺失数据鲁棒性和可解释性分析方案。现有稿件尚未提供完整的数值结果表，因此本文不额外编造性能数值，而是对模型原理、实验设置和定性结论作出严格整理。")
add_para(doc, "关键词：风电功率预测；非平稳时间序列；多粒度建模；时空图；Prompt 对齐；交叉注意力；动态损失", style="Normal", first_line=False)

add_heading(doc, "1 引言", 1)
add_heading(doc, "1.1 背景与研究动机", 2)
add_para(doc, "风电功率预测同时具有时间依赖、空间耦合和状态切换三类特征。对单个风机而言，近期风速、风向和机组状态会直接影响功率；对风电场而言，上游风机产生的扰动可能在一定延迟后传递至下游风机；在更长的时间范围内，气压、温度、湿度等气象因素又会改变整个区域的风况。因此，不同风机之间的联系并不是固定不变的，不同时间尺度上的有效信息也不完全相同。")
add_para(doc, "这一特点带来一个直接的问题：模型既要看清阵风、尾流和局部爬坡等短时变化，又要识别能够延续到未来的低频趋势。如果只使用高频序列，模型容易把噪声当作规律；如果过度平滑或只使用粗粒度序列，又可能遗漏决定短期预测误差的局部突变。因而，风电功率预测需要在时间粒度、空间拓扑和跨粒度信息对齐之间建立清晰的协同关系。")
add_heading(doc, "1.2 现有方法的局限", 2)
add_para(doc, "现有方法通常从三个方向进行改进：一是通过时间聚合获得多粒度表示，二是通过图神经网络建模多风机之间的空间关系，三是通过注意力或 Prompt 机制增强不同特征流之间的信息交互。这些思路各有价值，但在本任务中仍有三点需要进一步处理。")
add_number(doc, "固定时间聚合对突变的适应性有限。以 Exponential Almon 为代表的固定权重聚合隐含了较稳定的历史影响规律，而真实风场中的阵风、尾流和极端天气变化往往具有明显的非线性和阶段性。固定窗口可能过度平滑高频变化，也可能在平稳阶段引入不必要的噪声。")
add_number(doc, "分层掩码和零填充可能造成不自然的对齐。粗粒度信息映射到高频时间轴时，若空缺位置直接填入 0，模型看到的就不再是连续的气象演化，而是带有人工断点的序列。这种做法不仅削弱跨粒度信息的语义一致性，也难以应对真实传感器缺失。")
add_number(doc, "空间建模与粒度建模容易被拆成前后相继的步骤。若空间图只作为一次性预处理，粗粒度趋势和细粒度扰动就无法在同一层内相互校正；若图完全依赖高频数据自由学习，远距离站点之间的偶然相关又可能被误认为稳定的空间联系。")
add_number(doc, "全局平均损失难以同时照顾不同预测视界和不同风机。短视界与长视界、平稳风机与高波动风机的学习难度并不一致。若所有位置共享相同的优化权重，模型容易优先拟合简单样本，而忽略真正决定泛化能力的困难时段和困难站点。")
add_heading(doc, "1.3 本文思路与贡献", 2)
add_para(doc, "针对上述问题，本文围绕一条明确的主线组织模型：多变量历史输入先形成 Fine/Coarse 双粒度表示；随后，趋势先验与数据驱动的自适应图共同约束空间传播；在此基础上，ST-MG Coupling Block 让空间增强后的粗、细粒度特征在层内双向交互；最后，Macro-Trend Prompt 和 ST Prompt 为不同风机、不同未来步和不同粒度提供条件化解码信息，MS-MG-DWU 再从预测视界和风机站点两个维度调节优化重点。本文的主要工作可概括为：")
add_bullet(doc, "采用固定 0.5/0.5 的 Fine/Coarse 双分支作为最终正式模型，分别承载局部波动与宏观趋势，避免把最终结论建立在不稳定的动态门控之上。")
add_bullet(doc, "构建趋势先验自适应双图。低频趋势或空间关系提供可解释的先验边界，自适应图在此基础上学习数据驱动的补充联系；双向扩散进一步表达可能存在的方向性传播。")
add_bullet(doc, "在 ST-MG Coupling Block 内完成空间增强、Macro-Trend Prompt 生成和 Fine/Coarse 对称交叉融合，使空间关系与多粒度关系同步更新，而不是简单串联。")
add_bullet(doc, "采用带站点动态权重和粒度难度权重的 MS-MG-DWU Loss，并以统一的掩码协议处理无效目标，从而使多视界、多站点预测具有可比较的优化基础。")

add_heading(doc, "2 问题定义与正式模型协议", 1)
add_heading(doc, "2.1 预测任务", 2)
add_para(doc, "对每个预测样本，模型读取连续的历史观测窗口。输入张量定义为：", first_line=False)
add_equation(doc, "X ∈ R[B, T, N, C]，其中 T = 144，N = 134，C = 16。")
add_para(doc, "这里，B 表示 batch size，T 表示历史时间窗口长度，N 表示风机数量，C 表示输入变量数量。历史窗口由过去 144 个 10 min 时间步组成，覆盖风速、风向、气象变量以及历史功率等信息。模型一次性输出未来 10 个时间步的预测结果：", first_line=False)
add_equation(doc, "Ŷ ∈ R[B, N, 10]。")
add_para(doc, "实验在 H3、H6 和 H10 三个预测视界上评价。目标变量使用 Patv_raw；输入端使用清洗后的 Patv_clean_for_input；所有损失和评价指标都由 valid_target_mask 控制，只在有效目标位置上计算。三者分别对应原始目标、用于输入的清洗功率以及用于屏蔽无效标签的掩码，不能混用。", first_line=True)
add_table(doc, ["项目", "正式设置"], [
    ("模型名称", "FULL_FIXED_DUAL_DYNAMIC_MSMGDWU"),
    ("正式实例", "full_fixed_dual_keep_msmgdwu_seed2026"),
    ("实验别名", "CANONICAL_FULL = P0 = A0；P0、A0 仅是不同实验表中的引用"),
    ("历史窗口", "144 个 10 min 时间步"),
    ("风机数量", "N = 134"),
    ("输入特征", "C = 16"),
    ("输出与评价", "未来 10 步；H3、H6、H10"),
    ("目标与掩码", "Patv_raw；Patv_clean_for_input；valid_target_mask"),
], [1900, 7460])

add_heading(doc, "2.2 模型主线", 2)
add_para(doc, "ST-MGPrompt 并不是把图网络、Prompt 和多尺度模块机械叠加，而是围绕“哪些信息应当被保留、哪些信息应当被交互、哪些任务应当被优先学习”三个问题进行设计。整个过程可以按以下顺序理解：")
add_number(doc, "先把历史序列分成 Fine 和 Coarse 两条表示。Fine 分支保留短期局部波动，Coarse 分支描述较平滑的宏观趋势。最终正式模型固定两条分支的融合门控为 Fine gate = 0.5、Coarse gate = 0.5。")
add_number(doc, "再为两条分支建立不同侧重的空间图。Coarse 分支使用更强调低频趋势相似性的宏观气象联动图，Fine 分支使用更强调近邻局部扰动和尾流影响的微观局部图。")
add_number(doc, "空间增强后，Coarse 表示被压缩为 Macro-Trend Prompt，Fine/Coarse 表示则通过对称交叉注意力互相更新。这样，宏观趋势可以约束局部预测，局部扰动也可以反向修正宏观状态。")
add_number(doc, "解码时，ST Prompt 将风机 ID、预测视界和粒度分辨率编码为条件信息，帮助模型区分“预测哪台风机、预测未来哪一步、当前使用哪种粒度”。")
add_number(doc, "训练时，MS-MG-DWU 根据不同 horizon 和不同风机的学习难度动态调整损失权重，同时忽略无效目标。")

add_heading(doc, "3 方法", 1)
add_heading(doc, "3.1 Fixed Dual 双粒度表示", 2)
add_para(doc, "多粒度建模的目的不是简单地把时间窗口切成两段，而是让两条表示承担不同的预测职责。Fine branch 关注局部变化、短时波动和突变，适合识别阵风、尾流扰动和功率爬坡；Coarse branch 关注平滑趋势、宏观演化和较长时间依赖，适合提供稳定的背景状态。")
add_para(doc, "原稿曾考虑根据波动率动态调整两条分支的权重，但最终正式模型不再使用动态 gate，而是固定为 Fine gate = 0.5、Coarse gate = 0.5。这样做并不意味着固定权重在任何场景下都更优，而是避免中等波动样本导致门控不稳定，同时确保两类信息始终被保留。Dynamic VADSP 保留为机制诊断和消融对照，不作为正式模型的定义。")
add_equation(doc, "X_dual = 0.5 · X_fine + 0.5 · X_coarse。")
add_para(doc, "在实现上，双分支仍分别进入后续的空间图与时空编码器；上式用于说明其正式的融合协议，而不是把两条分支提前压成一条、再丢失粒度身份。")

add_heading(doc, "3.2 趋势先验自适应双图", 2)
add_para(doc, "风机之间的空间关系受到地理位置、风向、风速传播、尾流效应、功率趋势一致性和上下游方向等因素影响。完全固定的图无法适应状态变化，完全自由学习的图又可能把高频噪声产生的偶然相关当成稳定连接。因此，本文采用 Trend Prior + Adaptive Graph 的组合。")
add_para(doc, "首先，只利用历史信息提取低频趋势，例如 causal moving average 或 causal EMA。由此得到风机之间的趋势相似性矩阵 A_trend；距离关系矩阵 A_dist 则由风机空间位置计算。二者按 alpha = 0.7 融合，并通过 top-k 稀疏化保留主要连接，默认 top-k = 5。该过程只在训练数据上估计，避免把验证集或测试集的信息带入图构建。")
add_equation(doc, "A_prior = Normalize(α A_trend + (1 − α) A_dist)，其中 α = 0.7。")
add_para(doc, "随后，以可学习节点嵌入 E1 和 E2 生成自适应邻接关系：")
add_equation(doc, "A_adp = Softmax(ReLU(E1 E2ᵀ))。")
add_para(doc, "最终图关系由先验和自适应关系共同决定。本文的默认实现使用 Hadamard Product 进行门控，使趋势先验能够筛除缺少物理或趋势支持的高频伪相关，同时保留数据驱动图对未建模关系的补充能力。")
add_equation(doc, "A_final = Normalize(A_adp ⊙ A_prior)。")
add_para(doc, "为服务于不同粒度，模型保留两类空间图：A_coarse 更强调广域趋势联动，A_fine 更强调局部近邻扰动和尾流传播。它们不是两套互不相关的图，而是对同一风场在不同时间粒度下的互补描述。")

add_heading(doc, "3.3 Bi-Diffusion 时空编码", 2)
add_para(doc, "真实风场中的影响具有方向性。例如，上游风机可能先受到阵风影响，再通过风向传播到下游风机；反向传播或反馈关系也可能存在。为此，图卷积同时使用正向邻接矩阵 A 和转置矩阵 Aᵀ，分别描述两个方向上的信息扩散：")
add_equation(doc, "H_spatial = GCN(X, A) + GCN(X, Aᵀ)。")
add_para(doc, "空间传播之后，模型使用因果时间编码器提取历史依赖。Fine branch 使用较短感受野的因果 TCN，以避免局部突变被过度平滑；Coarse branch 使用更长感受野的扩张因果 TCN，或在相同因果约束下使用轻量时间注意力，以提取较长周期趋势。由于每一层只访问当前及过去的时间步，该结构不会使用未来信息。")

add_heading(doc, "3.4 Macro-Trend Prompt 与 ST Prompt", 2)
add_para(doc, "Prompt 在本文中不是额外的装饰性向量，而是把不同粒度的表示转换成解码器能够直接使用的条件信息。其关键点有两个。")
add_para(doc, "第一，Macro-Trend Prompt 只由空间增强后的 h_coarse 生成。h_coarse 先经过时间池化，再经过线性投影和 LayerNorm，压缩为一组宏观趋势 token。它携带的是已经结合空间联动后的宏观状态，例如风场整体处于上升还是下降阶段、功率变化的主要方向以及跨风机共享的背景趋势。")
add_para(doc, "第二，ST Prompt 显式编码风机、未来步和粒度三类身份信息：")
add_equation(doc, "ST-Prompt[h,n] = Embedding(node n) + Embedding(horizon h) + Embedding(granularity)。")
add_para(doc, "这样，解码器接收到的不再只是“一个通用的未来查询”，而是带有具体任务条件的查询：哪一台风机、未来哪一步、需要关注哪一种粒度。该异步对齐不要求把粗粒度序列强行拉长，也不需要在空缺位置填入 0，因此不会人为破坏时间连续性。")

add_heading(doc, "3.5 ST-MG Coupling Block 与对称交叉融合", 2)
add_para(doc, "ST-MG Coupling Block 是全文的核心耦合单元。空间增强、Prompt 生成和 Fine/Coarse 交互都在该单元内完成。这样安排的原因是：空间关系会影响不同粒度的解释，而粒度信息也会影响哪些空间关系值得保留；将二者完全拆开，会丢失这种相互约束。")
add_para(doc, "分支 A 为 Coarse → Fine。把未来细粒度查询作为 Query，把由 h_coarse 生成的 Macro-Trend Prompt 作为 Key 和 Value。该分支用宏观空间趋势约束局部高频预测，减少模型被短期噪声牵引的风险。分支 B 为 Fine → Coarse。把粗粒度查询作为 Query，把近期的空间增强细粒度特征 h_fine 作为 Key 和 Value。该分支利用局部突变和近邻扰动反向修正宏观趋势，使长期状态不会脱离当前风场的实际变化。")
add_equation(doc, "Z_A = CrossAttention(Q_fine, K_macro, V_macro)。")
add_equation(doc, "Z_B = CrossAttention(Q_coarse, K_fine, V_fine)。")
add_para(doc, "两个分支分别经过残差连接、LayerNorm 和前馈网络，再由动态融合门控合成为最终表示。与 concat 或直接相加相比，这种对称交叉融合允许模型在每个预测样本上决定两类信息的相对作用；与单向引导相比，它还保留了局部细节对宏观状态的反向修正。")

add_heading(doc, "3.6 ST Prompt 多步解码", 2)
add_para(doc, "融合后的表示进入多步预测解码器。解码器以 ST Prompt 作为条件，将当前上下文映射到未来每个时间步和每台风机的预测值。输出形状为 [B, H, N]，其中 H = 10。与直接使用一个线性层输出所有未来步相比，ST Prompt 使不同预测视界可以使用不同的条件信息；同时，所有条件信息都来自历史窗口及其编码结果，满足因果推理约束。")

add_heading(doc, "3.7 MS-MG-DWU 动态联合损失", 2)
add_para(doc, "不同 horizon 和不同风机的预测难度通常并不相同。MS-MG-DWU 从粒度和站点两个维度处理这种不平衡：粒度权重模式为 difficulty_rate，站点权重模式为 dynamic，正式损失名称为 msmg_dwu_loss。")
add_para(doc, "对每个预测视界 h，先在 valid_target_mask 下计算掩码损失 L_g[h]。为避免某一视界长期被忽略，模型为每个视界维护可学习的不确定性参数 log_sigma_g，并采用不确定性加权：")
add_equation(doc, "L_granularity = Σ_h [exp(−log_sigma_g[h]) · L_g[h] + log_sigma_g[h]]。")
add_para(doc, "对每个风机 n，计算该风机在所有有效预测位置上的平均误差，并用指数移动平均维护难度估计。站点权重按误差与平均误差的比值归一化，再裁剪到 [0.5, 3.0]，且权重统计与梯度分离，避免网络通过改变权重统计本身来降低损失。默认 ema_alpha = 0.9，站点损失权重 lambda_site = 0.2。")
add_equation(doc, "L_site = mean_n [w_site[n] · L_node[n]]；L_total = mean(L_granularity) + λ_site · L_site。")
add_para(doc, "所有基础损失均采用掩码形式，分母使用 mask.sum().clamp_min(1)，以保证整批目标无效时训练不会崩溃，也不会产生 NaN 或 Inf。去掉 MS-MG-DWU 时，使用 masked_score_aligned_hybrid 作为对照，而不是把损失项完全删除。")

add_heading(doc, "4 实验设计与评价协议", 1)
add_heading(doc, "4.1 数据与预处理", 2)
add_para(doc, "实验使用原稿所述的真实多站点风电场集群数据，研究对象为分散式海上风电场景。每个样本包含 134 台风机在过去 144 个 10 min 时间步上的 16 维输入特征，并预测未来 10 个时间步的功率。数据按时间顺序划分训练、验证和测试集，不进行随机打乱式的时间切分；标准化器只在训练集上拟合。")
add_para(doc, "功率变量严格区分为 Patv_raw 和 Patv_clean_for_input。前者用于保留原始预测目标，后者用于构造模型输入；valid_target_mask 用于标记目标是否有效。评价指标、损失和缺失数据实验都必须使用同一掩码协议，以避免不同实验使用不同有效样本集合。")
add_heading(doc, "4.2 训练、选择与评价", 2)
add_para(doc, "所有主要对照使用相同的数据划分、输入输出形状、训练预算和早停协议。正式实例的随机种子为 2026。最大训练轮数为 20 epochs，监控指标为 val_official_score_h10，指标方向为越低越好；patience = 6，min_delta = 0.01。每个 epoch 结束后，如果验证集 H10 Score 至少改善 0.01，则保存 best checkpoint 并重置等待计数；连续 6 个 epoch 没有达到该改善幅度时提前停止。")
add_para(doc, "在 H3、H6 和 H10 上分别截取预测结果，报告 MAE、RMSE 和 R²；若项目已有官方 Score，则同时报告该指标。所有指标都在 valid_target_mask 下计算，并对每个 horizon 单独记录，以保证短视界与长视界之间的差异能够被看见。")
add_heading(doc, "4.3 基线方法", 2)
add_para(doc, "基线设置按“是否建模多任务、是否建模空间关系、是否支持多粒度协同”逐步展开。单任务基线包括 LSTM 和 TCN，用于检验多任务多粒度建模的必要性；经典时空图模型包括 Graph WaveNet 和 STGCN，用于比较纯数据驱动图与本文趋势先验自适应图的差异；多粒度相关基线包括 MGCNet 和 STCADFN，用于比较固定聚合、分层掩码或串行融合与本文 Prompt 对齐和层内对称耦合的差异。")
add_heading(doc, "4.4 消融实验", 2)
add_para(doc, "为避免不同表格对同一模型使用不同名称，本文统一规定 CANONICAL_FULL = P0 = A0。P0 和 A0 指向同一次训练、同一个 checkpoint 和同一组评价指标，不重复训练。最终正式模型的核心消融如下。")
add_table(doc, ["编号", "变体", "所检验的机制"], [
    ("A0 / P0", "Full Fixed Dual + Trend Prior + Adaptive Graph + Bi-Diffusion + Macro Prompt + Cross Fusion + ST Prompt + MS-MG-DWU", "完整正式模型"),
    ("A1", "w/o Spatial Graph", "不执行节点间空间传播，检验空间图的必要性"),
    ("A2", "w/o Adaptive Graph", "只保留 prior graph，检验自适应图的增益"),
    ("A3", "w/o Diffusion", "用 Simple Graph Operator 替代 Bi-Diffusion，检验方向性传播"),
    ("A4", "Mean-Pooling Macro Prompt", "保持 4 个 Macro Prompt token，仅将 attention temporal pooling 改为等权 mean pooling"),
    ("A5", "Short-context Reverse Cross", "保留双向 Cross，将 Fine→Coarse 近期上下文从 24 steps 缩短为 6 steps"),
    ("A6", "Additive Cross Fusion", "保留双向 cross-attention，以 additive fusion 替代 adaptive gated fusion"),
    ("A7", "Temporal-Granularity Prompt", "保留 STPromptDirectDecoder，仅移除 node identity embedding，保留 future-step 与 granularity embedding"),
    ("A8", "w/o MS-MG-DWU", "使用 masked_score_aligned_hybrid 训练"),
], [1100, 4660, 3900])
add_para(doc, "原稿中还记录了 w/o VADSP、Fine-only、Coarse-only、Fixed Dual、Random Gate 和 Shuffled Volatility 等机制诊断。由于最终正式模型采用固定双分支，Dynamic VADSP 不再是 Full 模型的一部分；这些变体应作为额外的分支机制实验，用来回答“动态波动率门控是否真的利用了波动条件信息”，而不能与 A0 的最终定义混写。")
add_heading(doc, "4.5 鲁棒性与可解释性实验", 2)
add_para(doc, "缺失数据实验在测试集高频细粒度输入上随机遮蔽 10%、20% 和 30% 的观测，用于模拟传感器掉线。比较重点不是简单补零后的平均误差，而是 Macro-Trend Prompt 能否利用仍然可用的宏观趋势信息，为缺失的局部高频预测提供条件约束。")
add_para(doc, "机制解释主要包含三部分。第一，按历史窗口波动程度将样本划分为 Low、Medium 和 High volatility，并比较动态分块或固定双分支在高波动、功率爬坡和突变时段的差异。第二，分析空间图在平稳天气与强天气过程中的连接变化，观察 Fine 图是否更集中于局部近邻、Coarse 图是否更体现广域联动。第三，使用 SHAP 或等价的归因方法比较不同 horizon 的驱动因素：短视界应更多依赖目标站点近期风速、风向和功率惯性，长视界则应更多依赖气象协变量、粗粒度趋势和上游站点的传播信息。上述分析用于检验模型是否学习到与风场机理一致的模式，而不是将可视化结果直接当作因果证明。")

add_heading(doc, "5 结果解读与讨论", 1)
add_heading(doc, "5.1 多粒度协同的作用", 2)
add_para(doc, "本文的多粒度设计遵循“粗粒度提供背景约束、细粒度保留局部修正”的分工。对短视界预测，近期风速、风向和历史功率的细粒度变化通常更重要；随着预测视界延长，单个站点近期观测的直接贡献会减弱，气象趋势、空间传播和粗粒度状态的重要性会相对上升。因而，Fine/Coarse 的价值不应只用一个总体平均分数判断，还应结合 H3、H6、H10 以及高波动样本分别观察。")
add_heading(doc, "5.2 空间先验与自适应图的互补性", 2)
add_para(doc, "趋势先验的作用不是替代自适应图，而是把学习空间限制在更容易解释的范围内。对于长期或低频趋势，空间上较远的风机也可能因共同天气系统而具有联系；对于短期扰动，近邻风机和上下游关系往往更关键。双图设计因此不是简单复制同一张邻接矩阵，而是让不同粒度使用不同的空间关注范围。Bi-Diffusion 则进一步保留传播方向信息，使图结构能够表达上游到下游的非对称影响。")
add_heading(doc, "5.3 Prompt 对齐与对称融合的解释", 2)
add_para(doc, "Macro-Trend Prompt 可以理解为由 Coarse branch 提供的一张“背景提示卡”：它不直接替代细粒度序列，而是告诉细粒度预测当前风场处于怎样的宏观状态。ST Prompt 则像解码器的任务标签，明确指定风机、未来步和粒度。两者结合后，模型在预测每个未来位置时既有共享的宏观背景，也有与具体任务对应的条件信息。")
add_para(doc, "对称 Cross-Fusion 的必要性在于，粗、细粒度之间的关系不是单向的。宏观趋势可以抑制局部噪声，但局部突变也可能暴露宏观趋势的滞后。允许 Fine → Coarse 的反向修正，可以减少长视界预测对过时趋势的依赖；允许 Coarse → Fine 的正向约束，则可以减少短视界预测对偶然噪声的过度响应。")
add_heading(doc, "5.4 关于 VADSP 与固定双分支的边界", 2)
add_para(doc, "原稿曾把波动感知动态语义分块作为重要方向。该思路在机制上具有直观意义：平稳阶段可以扩大感受野，突变阶段可以缩小窗口、保留局部细节。但从正式实验协议看，最终模型采用固定 0.5/0.5 双分支，说明本文当前更重视跨 horizon 的稳定性和可复现实验边界。VADSP 的合理定位应是诊断性机制或单独消融，而不是在未统一训练协议的情况下与正式模型混合表述。若未来结果显示 VADSP 在 High-volatility 和功率爬坡样本上稳定优于 Fixed Dual，再进一步讨论将其纳入正式版本会更稳妥。")
add_heading(doc, "5.5 结果报告的完整性要求", 2)
add_para(doc, "为了让审稿人能够复核结论，最终结果表应至少报告各模型在 H3、H6、H10 上的 MAE、RMSE、R² 和官方 Score，并同时给出参数量、训练时间、推理时间和 best epoch。消融表需要确保每一行只改变一个机制；缺失数据实验需要明确遮蔽位置、遮蔽比例和掩码处理方式；解释性分析需要区分“模型关注了什么”和“变量具有因果作用”这两个不同命题。")
add_para(doc, "当前初稿没有提供可核验的完整数值表，因此本文保留了原稿已有的定性判断和实验设计，但没有补写具体的性能提升百分比。正式投稿前，应将实际运行得到的 metrics.csv、消融结果和可视化证据填入相应位置，并以同一 checkpoint 和同一评价协议为准。")

add_heading(doc, "6 结论", 1)
add_para(doc, "本文围绕高维非平稳风电功率预测中的三个实际难点展开：高频局部波动与低频长期趋势并存，风机之间存在动态且具有方向性的空间传播，以及不同预测视界和站点的学习难度不均衡。为此，本文构建 ST-MGPrompt，并将其正式模型固定为 FULL_FIXED_DUAL_DYNAMIC_MSMGDWU。该模型以 Fine/Coarse 双粒度表示为起点，以趋势先验和自适应图共同约束空间传播，以 Bi-Diffusion 表达方向性影响，再通过 Macro-Trend Prompt、ST Prompt 和对称交叉融合完成跨粒度的层内协同，最后使用 MS-MG-DWU 对不同 horizon 和不同站点进行动态优化。")
add_para(doc, "从方法逻辑上看，本文的核心不是增加模块数量，而是让每个模块承担清楚且互相衔接的职责：双粒度负责分离趋势与扰动，双图负责区分宏观联动与局部传播，Prompt 负责把空间增强后的状态对齐到具体预测任务，Cross-Fusion 负责在粒度之间建立双向修正，动态损失负责把训练资源分配给更难的预测位置。这样的组织方式有助于降低模型解释和实验复现的难度。")
add_para(doc, "本文仍有两点需要在最终版本中进一步完善。第一，需要补充完整的数值实验和显著性或重复运行分析，避免仅依靠定性描述评价模型。第二，需要在不同风场、不同缺失模式和不同预测视界下检验趋势先验与固定双分支的稳定性，并谨慎区分相关性解释与因果结论。")

add_heading(doc, "参考文献", 1)
add_para(doc, "原稿未提供完整的参考文献条目。正式投稿前，请依据 Applied Energy 的引用格式补齐 MGCNet、STCADFN、Graph WaveNet、STGCN、TCN、LSTM、SHAP 以及相关时空图和多粒度预测工作的完整书目信息，并在正文中逐一对应引用。", first_line=False)

# Keep the title and headings together with following text, and set document metadata.
doc.core_properties.title = "基于时空多粒度协同对齐的高维非平稳风电功率预测"
doc.core_properties.subject = "ST-MGPrompt revised manuscript"
doc.core_properties.author = ""
doc.core_properties.comments = "Rewritten for logical coherence and readability; canonical model protocol aligned with the model-detail document."

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(OUT)
