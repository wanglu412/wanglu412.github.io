// ========================
// 1. 配置：优点文本 & 颜色
// ========================

// 这里可以随便加删改，50~90 条都没问题
const virtueTexts = [
  "你很耐心",
  "你很温柔",
  "你很体贴",
  "你很善良",
  "你很健谈",
  "你很真诚",
  "你很会倾听",
  "你充满活力",
  "你很会照顾别人",
  "你很有责任感",
  "你很幽默",
  "你值得被珍惜",
  "你很勇敢",
  "你很有创意",
  "你很温暖",
  "你很细心",
  "你很自律",
  "你很有想法",
  "你很懂我",
  "你总在为别人着想",
  "你给人安全感",
  "你很值得信赖",
  "你笑起来超好看",
  "你有自己的坚持",
  "你很聪明",
  "你也会温柔地对自己",
  "你认真对待每一段关系",
  "你不轻易放弃",
  "你总能发现生活的小快乐",
  "你会安慰人",
  "你很真挚",
  "你很可爱",
  "你很可靠",
  "你愿意为别人付出",
  "你有治愈别人的能力",
  "你总能把气氛变得轻松",
  "你对世界保持好奇",
  "你认真生活",
  "你给人很多勇气",
  "你是独一无二的存在",
  "你比自己想象中更重要",
  "和你在一起很安心",
  "你值得被好好爱着",
  "你本身就是一个很了不起的故事",
  "你愿意相信别人",
  "你能温柔地接住别人的情绪",
  "你在慢慢学会爱自己",
  "你比昨天又成长了一点",
  "你一直在默默努力",
  "你带来很多光",
  "有你在，世界好像都变得好了点"
];

// 高对比度颜色列表（深色系，和白色对比度强）
const highContrastColors = [
  // 深色系：保证对比度
  "#111827", // 深灰蓝
  "#1f2937",
  "#0f172a",
  "#1e293b",
  "#4b5563",

  // 鲜艳红橙
  "#b91c1c", // 深红
  "#ef4444", // 鲜红
  "#f97316", // 橙色
  "#ea580c", // 深橙

  // 粉色 & 玫红
  "#db2777", // 玫红
  "#be185d",
  "#ec4899",

  // 亮紫 & 深紫
  "#7c3aed",
  "#6d28d9",
  "#4c1d95",

  // 亮蓝 & 深蓝
  "#2563eb",
  "#1d4ed8",
  "#0f766e", // 蓝绿

  // 绿色系
  "#15803d",
  "#16a34a",
  "#22c55e",

  // 金黄偏深（亮但还看得清）
  "#ca8a04",
  "#d97706"
];


// 从颜色列表中轮流取色
function pickColor(index) {
  return highContrastColors[index % highContrastColors.length];
}

// ========================
// 2. 使用心形公式生成坐标
// ========================
//
// 经典心形参数方程：
// x = 16 sin^3 t
// y = 13cos t − 5cos 2t − 2cos 3t − cos 4t
//
// 我们：
// - 在 0~2π 中按点数平均采样 t
// - 计算 (x, y)
// - 归一化到 [0, 100]% 范围
// - 把 y 轴翻转以适配屏幕坐标
// ========================

function generateHeartPositions(numPoints) {
  const points = [];

  // 采样 t
  for (let i = 0; i < numPoints; i++) {
    const t = (Math.PI * 2 * i) / numPoints;
    const x = 16 * Math.pow(Math.sin(t), 3);
    const y =
      13 * Math.cos(t) -
      5 * Math.cos(2 * t) -
      2 * Math.cos(3 * t) -
      Math.cos(4 * t);

    points.push({ x, y });
  }

  // 找出 x,y 的最大最小值，用来归一化
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;

  points.forEach((p) => {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  });

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  // 映射到 0~100%（稍微缩一点，避免贴边）
  const margin = 8; // 8% 留白
  const scaleX = 100 - margin * 2;
  const scaleY = 100 - margin * 2;

  return points.map((p) => {
    const nx = ((p.x - minX) / rangeX) * scaleX + margin;
    const ny = ((p.y - minY) / rangeY) * scaleY + margin;

    // 屏幕坐标 y 轴向下，心形原公式 y 轴向上，所以这里反转一下
    const screenY = 100 - ny;

    return { x: nx, y: screenY };
  });
}

// ========================
// 3. DOM 加载完成后初始化
// ========================

window.addEventListener("DOMContentLoaded", () => {
  const popup = document.getElementById("popup");
  const confirmBtn = document.getElementById("confirmBtn");
  const heartArea = document.getElementById("heartArea");
  const finalMessage = document.getElementById("finalMessage");

  // 1）动态创建所有优点标签
  const virtues = virtueTexts.map((text, index) => {
    const el = document.createElement("div");
    el.className = "virtue";
    el.textContent = text;

    const color = pickColor(index);
    el.style.color = color;
    el.style.borderColor = "#555555"; // 深灰固定边框

    heartArea.appendChild(el);
    return el;
  });

  // 2）生成与优点数量相同的心形坐标
  const positions = generateHeartPositions(virtues.length);

  // 3）设定每个优点的位置 + 动画延迟（更顺滑的控制）
  virtues.forEach((el, index) => {
    const pos = positions[index];
    el.style.left = pos.x + "%";
    el.style.top = pos.y + "%";

    // 每条延迟 0.25s，你可以微调
    const delay = index * 0.25;
    el.style.animationDelay = `${delay}s`;
  });

  // 4）点击“确定”后，开始播放动画
  confirmBtn.addEventListener("click", () => {
    // 关闭弹窗
    popup.style.display = "none";

    // 依次让标签播放动画（show 类启用 animation）
    virtues.forEach((el) => {
      el.classList.add("show");
    });

    // 计算最后一个标签出现结束的时间，之后显示最终白云
    const lastDelay = (virtues.length - 1) * 0.25;
    const appearDuration = 0.7; // virtue-appear 动画时长（秒）
    const extraBuffer = 0.8; // 再多等一小会儿

    const totalTimeMs = (lastDelay + appearDuration + extraBuffer) * 1000;

    setTimeout(() => {
      finalMessage.classList.add("show");
    }, totalTimeMs);
  });
});
