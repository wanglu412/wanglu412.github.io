/***********************
 * 1) 配置区
 ***********************/

// URL 上带 ?to=名字
const RECIPIENT = new URLSearchParams(location.search).get("to") || "你";

// 开场独白
const prologueLines = [
  `${RECIPIENT}，有些话，今天想慢慢告诉你。`,
  `不是惊天动地的故事，却是日复一日的在意。`,
  `接下来，我会把我看到的你，一点一点说出来。`
];

// 优点文本（可增删，50~90都行）
const virtueTexts = [
  "你很耐心","你很温柔","你很体贴","你很善良","你很真诚","你很会倾听","你充满活力",
  "你很会照顾别人","你很有责任感","你很幽默","你值得被珍惜","你很勇敢","你很有创意","你很温暖",
  "你很细心","你很有想法","你很懂我","你总在为别人着想","你给人安全感","你很值得信赖",
  "你笑起来超好看","你有自己的坚持","你很聪明","你也会温柔地对自己","你认真对待每一段关系",
  "你不轻易放弃","你总能发现生活的小快乐","你会安慰人","你很真挚","你很可爱","你很可靠",
  "你愿意为别人付出","你有治愈别人的能力","你总能把气氛变得轻松","你对世界保持好奇","你认真生活",
  "你给人很多勇气","你是独一无二的存在","你比自己想象中更重要","和你在一起很安心","你值得被好好爱着",
  "你本身就是一个很了不起的故事","你愿意相信别人","你能温柔地接住别人的情绪","你在慢慢学会爱自己",
  "你比昨天又成长了一点","你一直在默默努力","你带来很多光","虽然有时爱逃避","虽然有时爱把事情闷在心里","但是你永远都是最好的你","你值得一切美好"
];

// 照片列表（把你的 10~20 张图片放到 assets 目录，填入文件名）
const photoFiles = [
  // 示例：把这些替换成你自己的图片名称
  "p01.jpg","p02.jpg","p03.jpg","p04.jpg","p05.jpg",
  "p06.jpg","p07.jpg","p08.jpg","p09.jpg","p10.jpg",
  "p11.jpg","p12.jpg","p13.jpg","p14.jpg","p15.jpg",
  "p16.jpg","p17.jpg","p18.jpg","p19.jpg","p20.jpg",
  "p21.jpg"
].map(n => `assets/${n}`);

// 颜色（更鲜艳）
const colors = [
  "#7c3aed","#6d28d9","#4c1d95","#2563eb","#db2777",
  "#b91c1c","#ef4444","#f97316","#ea580c","#db2777","#be185d","#ec4899",
  "#7c3aed","#6d28d9","#4c1d95","#2563eb","#1d4ed8","#0f766e",
  "#15803d","#16a34a","#22c55e","#ca8a04","#d97706"
];
const pickColor = i => colors[i % colors.length];

// 节奏控制
const PER_ITEM_DELAY = 0.25;   // 每条优点的延迟（秒）
const APPEAR_DURATION = 0.5;   // 优点动画时长（秒）
const EXTRA_BUFFER = 0.6;      // 结尾前缓冲（秒）

// 照片舞台轮播节奏（毫秒）
const PHOTO_FADE_MS = 700;       // 淡入/淡出时长（与 CSS 对齐）
const PHOTO_STAY_MS = 1100;      // 每张停留时间
const PHOTO_TOTAL_PER = PHOTO_FADE_MS + PHOTO_STAY_MS; // 约 1.8s/张

/***********************
 * 2) 工具函数
 ***********************/
function generateHeartPositions(numPoints){
  // 心形参数方程
  const pts = [];
  for (let i=0;i<numPoints;i++){
    const t = (Math.PI*2*i)/numPoints;
    const x = 16*Math.pow(Math.sin(t),3);
    const y = 13*Math.cos(t)-5*Math.cos(2*t)-2*Math.cos(3*t)-Math.cos(4*t);
    pts.push({x,y});
  }
  // 归一化到 0~100%
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  pts.forEach(p=>{minX=Math.min(minX,p.x);maxX=Math.max(maxX,p.x);minY=Math.min(minY,p.y);maxY=Math.max(maxY,p.y);});
  const rx=maxX-minX||1, ry=maxY-minY||1;
  const margin=8, sx=100-margin*2, sy=100-margin*2;
  return pts.map(p=>{
    const nx=((p.x-minX)/rx)*sx+margin;
    const ny=((p.y-minY)/ry)*sy+margin;
    return {x:nx, y:100-ny}; // 翻转 Y
  });
}

function typewriter(el, text, speed=38){
  return new Promise(resolve=>{
    el.textContent=""; let i=0;
    const reduced=window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const tick=()=>{
      const step = reduced ? text.length : Math.max(1, Math.round(Math.random()*2)+1);
      el.textContent += text.slice(i,i+step); i+=step;
      if(i<text.length){ setTimeout(tick, reduced?0:(speed+Math.random()*40)); } else { resolve(); }
    };
    tick();
  });
}

function preload(src){
  return new Promise((res,rej)=>{
    const img = new Image(); img.onload=()=>res(img); img.onerror=rej; img.src=src;
  });
}

/***********************
 * 3) 主流程
 ***********************/
window.addEventListener("DOMContentLoaded", async ()=>{
  const popup=document.getElementById("popup");
  const confirmBtn=document.getElementById("confirmBtn");
  const heartArea=document.getElementById("heartArea");
  const finalMessage=document.getElementById("finalMessage");
  const prologue=document.getElementById("prologue");
  const prologueLine=document.getElementById("prologueLine");
  const vignette=document.querySelector(".vignette");
  const bgm=document.getElementById("bgm");

  const photoStage=document.getElementById("photoStage");
  const photoMosaic=document.getElementById("photoMosaic");

  // 1) 动态创建优点
  const virtues = virtueTexts.map((text,i)=>{
    const el=document.createElement("div");
    el.className="virtue"; el.textContent=text;
    el.style.color=pickColor(i);
    el.style.borderColor="#555";
    heartArea.appendChild(el);
    return el;
  });

  // 2) 优点布局到心形
  const posVirtues = generateHeartPositions(virtues.length);
  virtues.forEach((el,i)=>{
    const p=posVirtues[i];
    el.style.left=p.x+"%"; el.style.top=p.y+"%";
    el.style.animationDelay = `${i*PER_ITEM_DELAY}s`;
  });

  // 3) 预加载照片
  let loadedPhotos=[];
  try{
    loadedPhotos = await Promise.all(photoFiles.map(preload));
  }catch(e){
    // 有单张失败也没关系，忽略
    loadedPhotos = loadedPhotos.filter(Boolean);
  }

  // 4) 点击开始：音乐 + 暗角 + 独白 → 优点 → 照片舞台 → 照片爱心 → 结尾云朵
  confirmBtn.addEventListener("click", async ()=>{
    popup.style.display="none";
    vignette.classList.add("on");

    // 音乐（尊重减少动态）
    try{
      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      bgm.volume = reduced ? 0 : 0.35;
      await bgm.play();
    }catch(e){}

    // 开场独白
    prologue.classList.add("show");
    for(const line of prologueLines){
      await typewriter(prologueLine, line);
      await new Promise(r=>setTimeout(r,650));
    }
    prologue.classList.remove("show");

    // 优点逐条出现
    virtues.forEach(el=>el.classList.add("show"));

    // 优点全部结束的时间点
    const totalVirtueMs = ((virtues.length-1)*PER_ITEM_DELAY + APPEAR_DURATION + EXTRA_BUFFER)*1000;

    // 到点后展示“照片舞台”
    setTimeout(async ()=>{
      if(loadedPhotos.length===0){
        // 没图就直接结尾
        finalMessage.textContent = `${RECIPIENT}，谢谢你。和你在一起，世界真的变好了 💗`;
        finalMessage.classList.add("show");
        return;
      }

      // 中央大图轮播
      photoStage.classList.add("show");
      const stageImgs = loadedPhotos.map(img=>{
        const el = document.createElement("img");
        el.src = img.src; el.alt="相册照片"; el.className="photo";
        photoStage.appendChild(el);
        return el;
      });

      // 逐张播放
      for(let i=0;i<stageImgs.length;i++){
        stageImgs.forEach((el,idx)=>el.classList.toggle("active", idx===i));
        await new Promise(r=>setTimeout(r, PHOTO_TOTAL_PER));
      }

      // 过渡到“照片爱心马赛克”
      photoStage.classList.remove("show");
      photoMosaic.classList.add("show");

      // 生成与照片数量匹配的心形坐标
      const posPhotos = generateHeartPositions(stageImgs.length);

      // 把同一批 img 变成缩略图，定位到心形
      stageImgs.forEach((el,i)=>{
        // 先移动到 mosaic 容器
        photoMosaic.appendChild(el);
        el.classList.remove("photo");
        el.classList.add("thumb");
        // 先居中（由 CSS translate(-50%,-50%) 控制）
        el.style.left = "50%";
        el.style.top  = "50%";
        // 再异步触发飞入动画
        setTimeout(()=>{
          el.style.left = posPhotos[i].x+"%";
          el.style.top  = posPhotos[i].y+"%";
          el.classList.add("in");
        }, 40 + i*30); // 轻微错峰，层次更好看
      });

      // 等缩略图入位后，显示云朵结尾
      const mosaicTotalMs = 1200 + stageImgs.length*30 + 600;
      setTimeout(()=>{
        finalMessage.textContent = `${RECIPIENT}，谢谢你！把点点滴滴放在一起，恰好是一颗心~`;
        finalMessage.classList.add("show");
      }, mosaicTotalMs);

    }, totalVirtueMs);
  });
});
