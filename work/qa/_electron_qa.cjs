// 真窗口验收本地网页工具：用本机 Electron 起一个可见窗口，驱动界面并截图。
// 用法（参数顺序：脚本、--user-data-dir、--out、--url 放最后）：
//   electron.exe work/qa/_electron_qa.cjs --user-data-dir=... --out=... --url=http://127.0.0.1:8765/
const { app, BrowserWindow, screen } = require("electron");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

// Windows 下 Chromium 会按原生窗口遮挡关系把被盖住的页面判成 hidden，
// 于是 document.visibilityState 忽而变 hidden、sendInputEvent 静默失效。关掉这个计算。
app.commandLine.appendSwitch("disable-features", "CalculateNativeWinOcclusion");

const argv = process.argv.slice();
const pick = (prefix) => {
  const hit = argv.find((a) => a.startsWith(prefix + "="));
  return hit ? hit.slice(prefix.length + 1) : null;
};
const OUT = pick("--out") || path.join(process.cwd(), "_electron_qa.json");
const URL_ARG = pick("--url") || "http://127.0.0.1:8765/";
const UPLOAD_FILE = pick("--upload") || "";
const UPLOAD2 = pick("--upload2") || "";
const UPLOAD_VIDEO = pick("--uploadVideo") || "";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const report = { steps: [], failed: [], uploadedItemId: null, secondItemId: null, testOutputDir: null, testJobId: null, motionOutputDir: null, motionJobId: null, videoItemId: null, cancelJobId: null, cancelOutputDir: null };
const LAST_TOAST = `(() => {
  const stack = document.getElementById('toastStack');
  const node = stack ? stack.lastElementChild : null;
  return node ? node.textContent.trim() : null;
})()`;
function check(name, ok, detail) {
  report.steps.push({ name, ok: Boolean(ok), detail: detail === undefined ? "" : String(detail) });
  if (!ok) report.failed.push(name);
  console.log(`${ok ? "PASS" : "FAIL"} ${name} ${detail === undefined ? "" : detail}`);
}

async function drag(win, x0, y0, x1, y1) {
  const wc = win.webContents;
  wc.sendInputEvent({ type: "mouseMove", x: Math.round(x0), y: Math.round(y0) });
  wc.sendInputEvent({ type: "mouseDown", x: Math.round(x0), y: Math.round(y0), clickCount: 1 });
  const steps = 6;
  for (let i = 1; i <= steps; i += 1) {
    wc.sendInputEvent({
      type: "mouseMove",
      x: Math.round(x0 + ((x1 - x0) * i) / steps),
      y: Math.round(y0 + ((y1 - y0) * i) / steps),
    });
    await sleep(16);
  }
  wc.sendInputEvent({ type: "mouseUp", x: Math.round(x1), y: Math.round(y1), clickCount: 1 });
  await sleep(90);
}

async function ensureVisible(win) {
  // 后台窗口没有合成器，sendInputEvent 与 scrollIntoView 都会失效（实测抛 UnknownVizError）
  let state = "hidden";
  for (let attempt = 0; attempt < 4; attempt += 1) {
    win.setAlwaysOnTop(true);
    if (win.isMinimized()) win.restore();
    win.show();
    win.moveTop();
    win.focus();
    app.focus({ steal: true });
    await sleep(300 + attempt * 250);
    state = await win.webContents.executeJavaScript("document.visibilityState");
    if (state === "visible") return;
  }
  throw new Error("窗口不可见：" + state);
}

async function switchPoint(win, id = "videoFastMode") {
  const wc = win.webContents;
  await wc.executeJavaScript(`document.getElementById('${id}').closest('.video-switch').scrollIntoView({block:'center'})`);
  await sleep(250);
  const point = await wc.executeJavaScript(`(() => {
    const row = document.getElementById('${id}').closest('.video-switch');
    const r = row.getBoundingClientRect();
    return { x: Math.round(r.left + 20), y: Math.round(r.top + r.height / 2),
             inView: r.top >= 0 && r.bottom <= innerHeight && r.left >= 0 && r.right <= innerWidth };
  })()`);
  if (!point.inView) throw new Error("开关不在视口内：" + JSON.stringify(point));
  return point;
}

async function shot(win, file) {
  const image = await win.webContents.capturePage();
  fs.writeFileSync(file, image.toPNG());
  return file;
}

async function setViewport(win, width, height) {
  // 本机屏幕是 1260×840 / 200% 缩放的小屏（探针实测）：内容高度只要加上标题栏超过工作区，
  // Windows 会把**整次**改尺寸拒掉（宽度也一起不动），所以高度必须留出标题栏余量。
  // 另外 setBounds 量的是外框、断言读 innerWidth，所以用 setContentSize 直接定网页可视区。
  const work = (screen.getPrimaryDisplay() || {}).workAreaSize || { width: 1280, height: 800 };
  const targetWidth = Math.min(width, work.width - 40);
  const targetHeight = Math.min(height, work.height - 120);
  let inner = 0;
  for (let attempt = 0; attempt < 4; attempt += 1) {
    // 实测到过窗口在这一刻是被最小化的（getContentBounds 返回 -16000×-16000/0×0），
    // 那种状态下任何改尺寸调用都不生效，必须先 restore + show。
    if (win.isMinimized()) win.restore();
    win.show();
    if (win.isMaximized()) win.unmaximize();
    win.setContentSize(targetWidth, targetHeight);
    await sleep(400 + attempt * 200);
    inner = await win.webContents.executeJavaScript("innerWidth");
    if (inner <= targetWidth + 4 && inner >= targetWidth - 80) return inner;
  }
  console.log(`     setViewport 没定住：innerWidth=${inner} 目标内容=${targetWidth}×${targetHeight}`
    + ` bounds=${JSON.stringify(win.getBounds())} content=${JSON.stringify(win.getContentBounds())}`
    + ` minimized=${win.isMinimized()} maximized=${win.isMaximized()} 屏幕工作区=${JSON.stringify(work)}`);
  return inner;
}

async function run(win) {
  const wc = win.webContents;
  await wc.loadURL(URL_ARG);
  await sleep(1200);

  // ---- 1. 视频开关：存在、默认可见且默认关 ----
  const switchInfo = await wc.executeJavaScript(`(() => {
    const input = document.getElementById('videoFastMode');
    const row = input ? input.closest('.video-switch') : null;
    const rect = row ? row.getBoundingClientRect() : null;
    const track = row ? row.querySelector('.switch-track').getBoundingClientRect() : null;
    return {
      exists: Boolean(input),
      rowRect: rect ? { w: Math.round(rect.width), h: Math.round(rect.height) } : null,
      trackRect: track ? { w: Math.round(track.width), h: Math.round(track.height) } : null,
      checked: input ? input.checked : null,
      isOn: row ? row.classList.contains('is-on') : null,
      copy: row ? row.querySelector('small').textContent.trim().slice(0, 24) : null,
      nativeHidden: input ? getComputedStyle(input).display : null,
    };
  })()`);
  check("开关存在", switchInfo.exists, JSON.stringify(switchInfo.rowRect));
  check("开关可见且未选中", switchInfo.checked === false && switchInfo.isOn === false, JSON.stringify(switchInfo));
  check("轨道尺寸正常", switchInfo.trackRect && switchInfo.trackRect.w >= 28 && switchInfo.trackRect.h >= 14,
    JSON.stringify(switchInfo.trackRect));

  // ---- 2. 真点击：状态、样式、localStorage、toast ----
  await ensureVisible(win);
  const firstPoint = await switchPoint(win);
  wc.sendInputEvent({ type: "mouseMove", x: firstPoint.x, y: firstPoint.y });
  wc.sendInputEvent({ type: "mouseDown", x: firstPoint.x, y: firstPoint.y, clickCount: 1 });
  wc.sendInputEvent({ type: "mouseUp", x: firstPoint.x, y: firstPoint.y, clickCount: 1 });
  await sleep(450);
  const afterClick = await wc.executeJavaScript(`(() => {
    const input = document.getElementById('videoFastMode');
    const row = input.closest('.video-switch');
    const knob = row.querySelector('.switch-knob').getBoundingClientRect();
    const track = row.querySelector('.switch-track').getBoundingClientRect();
    const toast = document.getElementById('toastStack')?.lastElementChild;
    return {
      checked: input.checked,
      isOn: row.classList.contains('is-on'),
      stored: localStorage.getItem('jingyin.videoMode'),
      knobShift: Math.round(knob.left - track.left),
      toast: toast ? toast.textContent.trim() : null,
      trackColor: getComputedStyle(row.querySelector('.switch-track')).backgroundColor,
    };
  })()`);
  check("点一下即切到速度优先", afterClick.checked && afterClick.isOn, JSON.stringify(afterClick));
  check("选择已记住", afterClick.stored === "fast", afterClick.stored);
  check("切换有 toast 提示", /速度优先/.test(afterClick.toast || ""), afterClick.toast);
  check("滑块真的滑过去了", afterClick.knobShift >= 10, `位移 ${afterClick.knobShift}px`);
  await shot(win, path.join(path.dirname(OUT), "_ui_switch_on.png"));

  // ---- 3. 刷新后仍然是速度优先 ----
  await wc.loadURL(URL_ARG);
  await sleep(1400);
  const afterReload = await wc.executeJavaScript(`(() => {
    const input = document.getElementById('videoFastMode');
    const row = input.closest('.video-switch');
    const toast = document.getElementById('toastStack')?.lastElementChild;
    return { checked: input.checked, isOn: row.classList.contains('is-on'),
             stored: localStorage.getItem('jingyin.videoMode'), toast: toast ? toast.textContent.trim() : null };
  })()`);
  check("刷新后保持速度优先", afterReload.checked && afterReload.isOn && afterReload.stored === "fast",
    JSON.stringify(afterReload));
  check("刷新后不再弹切换提示", !/优先/.test(String(afterReload.toast)), String(afterReload.toast));

  // ---- 4. 再点回画质优先 ----
  await ensureVisible(win);
  const back = await switchPoint(win);
  wc.sendInputEvent({ type: "mouseMove", x: back.x, y: back.y });
  wc.sendInputEvent({ type: "mouseDown", x: back.x, y: back.y, clickCount: 1 });
  wc.sendInputEvent({ type: "mouseUp", x: back.x, y: back.y, clickCount: 1 });
  await sleep(400);
  const afterSecond = await wc.executeJavaScript(`(() => {
    const input = document.getElementById('videoFastMode');
    const toast = document.getElementById('toastStack')?.lastElementChild;
    return { checked: input.checked, stored: localStorage.getItem('jingyin.videoMode'),
             toast: toast ? toast.textContent.trim() : null };
  })()`);
  check("可切回画质优先", afterSecond.checked === false && afterSecond.stored === "quality", JSON.stringify(afterSecond));
  check("画质优先也有提示", /画质优先/.test(afterSecond.toast || ""), afterSecond.toast);

  // ---- 4b. 动态水印开关：默认关，打开时把速度优先锁住 ----
  await ensureVisible(win);
  const motionStart = await wc.executeJavaScript(`(() => {
    const input = document.getElementById('videoMotion');
    const row = input ? input.closest('.video-switch') : null;
    const fast = document.getElementById('videoFastMode');
    return { exists: Boolean(input), checked: input ? input.checked : null,
             isOn: row ? row.classList.contains('is-on') : null,
             stored: localStorage.getItem('jingyin.videoMotion'),
             copy: row ? row.querySelector('small').textContent.trim() : null,
             fastDisabled: fast ? fast.disabled : null };
  })()`);
  check("动态水印开关存在且默认关", motionStart.exists && motionStart.checked === false && motionStart.isOn === false,
    JSON.stringify(motionStart));
  check("动态水印说明写清取舍", /每帧/.test(String(motionStart.copy)) && /速度优先/.test(String(motionStart.copy)),
    motionStart.copy);

  const motionPoint = await switchPoint(win, "videoMotion");
  wc.sendInputEvent({ type: "mouseMove", x: motionPoint.x, y: motionPoint.y });
  wc.sendInputEvent({ type: "mouseDown", x: motionPoint.x, y: motionPoint.y, clickCount: 1 });
  wc.sendInputEvent({ type: "mouseUp", x: motionPoint.x, y: motionPoint.y, clickCount: 1 });
  await sleep(450);
  const motionOn = await wc.executeJavaScript(`(() => {
    const input = document.getElementById('videoMotion');
    const row = input.closest('.video-switch');
    const fast = document.getElementById('videoFastMode');
    const fastRow = fast.closest('.video-switch');
    return { checked: input.checked, isOn: row.classList.contains('is-on'),
             stored: localStorage.getItem('jingyin.videoMotion'),
             fastDisabled: fast.disabled, fastMuted: fastRow.classList.contains('is-muted'),
             toast: (() => { const s = document.getElementById('toastStack'); return s && s.lastElementChild ? s.lastElementChild.textContent.trim() : null; })() };
  })()`);
  check("点一下打开动态水印", motionOn.checked && motionOn.isOn && motionOn.stored === "1", JSON.stringify(motionOn));
  check("动态水印打开时锁住速度优先", motionOn.fastDisabled === true && motionOn.fastMuted === true,
    `disabled=${motionOn.fastDisabled} muted=${motionOn.fastMuted}`);
  check("动态水印切换有提示", /动态水印/.test(String(motionOn.toast)), motionOn.toast);
  await shot(win, path.join(path.dirname(OUT), "_ui_motion_on.png"));

  const motionPoint2 = await switchPoint(win, "videoMotion");
  wc.sendInputEvent({ type: "mouseMove", x: motionPoint2.x, y: motionPoint2.y });
  wc.sendInputEvent({ type: "mouseDown", x: motionPoint2.x, y: motionPoint2.y, clickCount: 1 });
  wc.sendInputEvent({ type: "mouseUp", x: motionPoint2.x, y: motionPoint2.y, clickCount: 1 });
  await sleep(400);
  const motionOff = await wc.executeJavaScript(`(() => {
    const input = document.getElementById('videoMotion');
    const fast = document.getElementById('videoFastMode');
    return { checked: input.checked, stored: localStorage.getItem('jingyin.videoMotion'),
             fastDisabled: fast.disabled };
  })()`);
  check("可以再关掉动态水印", motionOff.checked === false && motionOff.stored === "0", JSON.stringify(motionOff));
  check("关掉后速度优先解锁", motionOff.fastDisabled === false, String(motionOff.fastDisabled));

  // ---- 4c. 照片重绘卡片：可用、能选中、真发出预热请求 ----
  const photoCard = await wc.executeJavaScript(`(() => {
    const card = document.getElementById('photoMethodCard');
    const input = card ? card.querySelector('input') : null;
    return { exists: Boolean(card), hidden: card ? card.hidden : null,
             ready: (document.querySelector('meta[name="photo-ready"]') || {}).content,
             disabled: input ? input.disabled : null,
             copy: card ? card.querySelector('small').textContent.trim() : null };
  })()`);
  check("照片重绘卡片存在且可用", photoCard.exists && photoCard.hidden === false && photoCard.ready === "1"
    && photoCard.disabled === false, JSON.stringify(photoCard));
  check("照片重绘文案提示文字页面改用智能修复", /文字/.test(String(photoCard.copy)), photoCard.copy);

  await wc.executeJavaScript(`(() => {
    const original = window.fetch;
    window.__warmup = 0;
    window.fetch = function (url) {
      if (String(url).indexOf('/api/photo/warmup') >= 0) window.__warmup += 1;
      return original.apply(this, arguments);
    };
    return true;
  })()`);
  await wc.executeJavaScript("document.querySelector('#photoMethodCard input').click()");
  await sleep(600);
  const photoSelected = await wc.executeJavaScript(`(() => ({
    stored: localStorage.getItem('jingyin.method'),
    cardSelected: document.getElementById('photoMethodCard').classList.contains('is-selected'),
    warmup: window.__warmup,
    toast: (() => { const s = document.getElementById('toastStack'); return s && s.lastElementChild ? s.lastElementChild.textContent.trim() : null; })(),
  }))()`);
  check("选中照片重绘并记住", photoSelected.stored === "photo" && photoSelected.cardSelected, JSON.stringify(photoSelected));
  check("选中照片重绘时后台预热模型", photoSelected.warmup >= 1, `warmup 请求 ${photoSelected.warmup} 次`);
  check("照片重绘提示首次加载较慢", /模型/.test(String(photoSelected.toast)), photoSelected.toast);
  await shot(win, path.join(path.dirname(OUT), "_ui_photo_mode.png"));
  await wc.executeJavaScript("document.querySelector('.method-card[data-method=\"smart\"] input').click()");
  await sleep(300);
  const backToSmart = await wc.executeJavaScript("localStorage.getItem('jingyin.method')");
  check("可以切回智能纹理修复", backToSmart === "smart", String(backToSmart));

  // ---- 5. 上传一份合成素材，真拖拽画满 16 框，第 17 框必须被拒 ----
  if (UPLOAD_FILE && fs.existsSync(UPLOAD_FILE)) {
    await wc.executeJavaScript(`(() => { const i = document.getElementById('fileInput'); i.hidden = false; i.style.display='block'; return true; })()`);
    const dbg = wc.debugger;
    dbg.attach("1.3");
    await dbg.sendCommand("DOM.enable");
    const doc = await dbg.sendCommand("DOM.getDocument");
    const node = await dbg.sendCommand("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#fileInput" });
    await dbg.sendCommand("DOM.setFileInputFiles", { files: [UPLOAD_FILE], nodeId: node.nodeId });
    await sleep(2500);
    const uploaded = await wc.executeJavaScript(`(() => ({
      items: document.querySelectorAll('#fileList li').length,
      id: document.querySelector('#fileList li') ? document.querySelector('#fileList li').dataset.id : null,
    }))()`);
    report.uploadedItemId = uploaded.id;
    check("合成素材已进批量列表", uploaded.items >= 1 && uploaded.id, `${uploaded.items} 份 item=${uploaded.id}`);

    await ensureVisible(win);
    const canvasRect = await wc.executeJavaScript(`(async () => {
      const li = document.querySelector('#fileList li');
      if (li) li.click();
      await new Promise((r) => setTimeout(r, 600));
      const r = document.getElementById('overlayCanvas').getBoundingClientRect();
      return { left: r.left, top: r.top, width: r.width, height: r.height };
    })()`);
    check("画布有实际尺寸", canvasRect.width > 120 && canvasRect.height > 80, JSON.stringify(canvasRect));

    const perRow = 6;
    const counts = [];
    for (let i = 0; i < 17; i += 1) {
      const col = i % perRow;
      const row = Math.floor(i / perRow);
      const x0 = canvasRect.left + 8 + col * (canvasRect.width - 20) / perRow;
      const y0 = canvasRect.top + 8 + row * 42;
      await drag(win, x0, y0, x0 + 22, y0 + 20);
      counts.push(await wc.executeJavaScript("document.querySelectorAll('#regionList li').length"));
    }
    check("每一次拖拽都新增一个框", counts.every((n, idx) => n === Math.min(idx + 1, 16)), counts.join(","));
    const gate = await wc.executeJavaScript(`(() => ({
      boxes: document.querySelectorAll('#regionList li').length,
      label: document.getElementById('regionCount').textContent.trim(),
      toast: (() => { const s = document.getElementById('toastStack'); return s && s.lastElementChild ? s.lastElementChild.textContent.trim() : null; })(),
    }))()`);
    check("画到第 17 框时只保留 16 个", gate.boxes === 16, `${gate.boxes} 个（目标 16）`);
    check("计数显示 16/16", gate.label.replace(/\s/g, "") === "16/16", gate.label);
    check("第 17 框弹出上限提示", /最多 16 个选框|已经画满/.test(String(gate.toast)), gate.toast);
    await shot(win, path.join(path.dirname(OUT), "_ui_region_limit.png"));

    // ---- 5b. 选框模板：存 16 框 → 清空 → 套用还原 → 不重复 → 刷新后还在 → 删除 ----
    const savedTemplate = await wc.executeJavaScript(`(async () => {
      document.getElementById('saveTemplateBtn').click();
      await new Promise((r) => setTimeout(r, 200));
      const items = Array.from(document.querySelectorAll('#templateList li'));
      const stored = JSON.parse(localStorage.getItem('jingyin.templates.v1') || '[]');
      const stack = document.getElementById('toastStack');
      return {
        blockHidden: document.getElementById('templateBlock').hidden,
        countLabel: document.getElementById('templateCount').textContent.trim(),
        listCount: items.length,
        text: items[0] ? items[0].textContent.replace(/\\s+/g, ' ').trim() : '',
        stored: stored.length ? stored[0].regions.length : 0,
        storedName: stored[0] ? stored[0].name : null,
        toast: stack && stack.lastElementChild ? stack.lastElementChild.textContent.trim() : null,
      };
    })()`);
    check("存为模板后模板区展开", savedTemplate.blockHidden === false && savedTemplate.listCount === 1,
      JSON.stringify(savedTemplate));
    check("模板记住 16 个选框", savedTemplate.stored === 16 && /16 个选框/.test(savedTemplate.text),
      `${savedTemplate.stored} 个 · ${savedTemplate.text}`);
    check("模板有名字与计数", /模板 1/.test(String(savedTemplate.storedName)) && savedTemplate.countLabel === "1",
      `${savedTemplate.storedName} · ${savedTemplate.countLabel}`);
    check("存模板有提示", /已存为/.test(String(savedTemplate.toast)), savedTemplate.toast);
    await shot(win, path.join(path.dirname(OUT), "_ui_template.png"));

    await wc.executeJavaScript("document.getElementById('clearRegionsBtn').click()");
    await sleep(300);
    const cleared = await wc.executeJavaScript(`(() => ({
      boxes: document.querySelectorAll('#regionList li').length,
      label: document.getElementById('regionCount').textContent.trim(),
      blockHidden: document.getElementById('templateBlock').hidden,
    }))()`);
    check("清空选框后模板还在", cleared.boxes === 0 && cleared.blockHidden === false, JSON.stringify(cleared));

    const applied = await wc.executeJavaScript(`(async () => {
      document.querySelector('#templateList .template-apply').click();
      await new Promise((r) => setTimeout(r, 250));
      const stack = document.getElementById('toastStack');
      return {
        boxes: document.querySelectorAll('#regionList li').length,
        label: document.getElementById('regionCount').textContent.trim(),
        toast: stack && stack.lastElementChild ? stack.lastElementChild.textContent.trim() : null,
      };
    })()`);
    check("套用模板把 16 个框加回来", applied.boxes === 16 && applied.label.replace(/\s/g, "") === "16/16",
      `${applied.boxes} 个 · ${applied.label}`);
    check("套用模板汇总了新增数量", /新增 16 个选框/.test(String(applied.toast)), applied.toast);

    const dedupe = await wc.executeJavaScript(`(async () => {
      document.querySelector('#templateList .template-apply').click();
      await new Promise((r) => setTimeout(r, 250));
      const stack = document.getElementById('toastStack');
      return {
        boxes: document.querySelectorAll('#regionList li').length,
        toast: stack && stack.lastElementChild ? stack.lastElementChild.textContent.trim() : null,
      };
    })()`);
    check("重复套用不会叠加选框", dedupe.boxes === 16 && /已经有|没有重复添加/.test(String(dedupe.toast)),
      `${dedupe.boxes} 个 · ${dedupe.toast}`);

    await sleep(900);
    await wc.loadURL(URL_ARG);
    await sleep(1800);
    const templateBack = await wc.executeJavaScript(`(() => {
      const rows = document.querySelectorAll('#fileList li');
      if (rows[0]) rows[0].click();
      return {
        listCount: document.querySelectorAll('#templateList li').length,
        countLabel: document.getElementById('templateCount').textContent.trim(),
        stored: (JSON.parse(localStorage.getItem('jingyin.templates.v1') || '[]')).length,
      };
    })()`);
    await sleep(600);
    const templateBackBoxes = await wc.executeJavaScript("document.querySelectorAll('#regionList li').length");
    check("刷新后模板还在", templateBack.listCount === 1 && templateBack.stored === 1
      && templateBack.countLabel === "1", JSON.stringify(templateBack));
    check("刷新后套用出来的 16 个框也在", templateBackBoxes === 16, `${templateBackBoxes} 个`);

    await wc.executeJavaScript("document.querySelector('#templateList .template-delete').click()");
    await sleep(350);
    const deleteDialog = await wc.executeJavaScript(`(() => ({
      hidden: document.getElementById('confirmModal').hidden,
      title: document.getElementById('confirmTitle').textContent.trim(),
    }))()`);
    check("删除模板先要确认", deleteDialog.hidden === false && /删除模板/.test(deleteDialog.title),
      JSON.stringify(deleteDialog));
    await wc.executeJavaScript("document.getElementById('confirmOkBtn').click()");
    await sleep(400);
    const deleted = await wc.executeJavaScript(`(() => ({
      listCount: document.querySelectorAll('#templateList li').length,
      stored: (JSON.parse(localStorage.getItem('jingyin.templates.v1') || '[]')).length,
      boxes: document.querySelectorAll('#regionList li').length,
      toast: (() => { const s = document.getElementById('toastStack'); return s && s.lastElementChild ? s.lastElementChild.textContent.trim() : null; })(),
    }))()`);
    check("删除模板后列表与存储都清空", deleted.listCount === 0 && deleted.stored === 0, JSON.stringify(deleted));
    check("删模板不动已套用的选框", deleted.boxes === 16, `${deleted.boxes} 个`);
    dbg.detach();
  } else {
    console.log("SKIP 素材上传/16 框部分：没有 --upload 指定的合成素材");
  }

  // ---- 6. 开关真的会改变浏览器发出去的请求（UI → 服务端口）----
  if (UPLOAD_FILE && fs.existsSync(UPLOAD_FILE)) {
    await ensureVisible(win);
    const onPoint = await switchPoint(win);
    wc.sendInputEvent({ type: "mouseMove", x: onPoint.x, y: onPoint.y });
    wc.sendInputEvent({ type: "mouseDown", x: onPoint.x, y: onPoint.y, clickCount: 1 });
    wc.sendInputEvent({ type: "mouseUp", x: onPoint.x, y: onPoint.y, clickCount: 1 });
    await sleep(400);
    await wc.executeJavaScript(`(() => {
      const original = window.fetch;
      window.__captured = null;
      window.fetch = async function (url, options) {
        const response = await original.apply(this, arguments);
        if (String(url).indexOf('/api/process') >= 0) {
          try {
            window.__captured = { body: options && options.body, job: await response.clone().json() };
          } catch (error) { window.__captured = { error: String(error) }; }
        }
        return response;
      };
      return true;
    })()`);
    await wc.executeJavaScript("document.getElementById('processBtn').click()");
    await sleep(700);
    const diag = await wc.executeJavaScript(`(() => {
      const btn = document.getElementById('processBtn');
      const stack = document.getElementById('toastStack');
      return {
        disabled: btn.disabled,
        jobModalHidden: document.getElementById('jobModal').hidden,
        checked: document.querySelectorAll('#fileList li input:checked').length,
        boxes: document.querySelectorAll('#regionList li').length,
        toast: stack && stack.lastElementChild ? stack.lastElementChild.textContent.trim() : null,
        captured: window.__captured ? Object.keys(window.__captured) : null,
      };
    })()`);
    console.log("     提交诊断：" + JSON.stringify(diag));
    let captured = null;
    for (let i = 0; i < 40; i += 1) {
      await sleep(500);
      captured = await wc.executeJavaScript("window.__captured ? { body: window.__captured.body, id: window.__captured.job && window.__captured.job.id, mode: window.__captured.job && window.__captured.job.video_mode } : null");
      if (captured && captured.body) break;
    }
    let sentMode = null;
    let sentBoxes = null;
    if (captured && captured.body) {
      const parsed = JSON.parse(captured.body);
      sentMode = parsed.video_mode;
      sentBoxes = parsed.items[0].regions.length;
      report.testJobId = captured.id;
      for (let i = 0; i < 60; i += 1) {
        const status = await wc.executeJavaScript(`fetch('/api/jobs/${captured.id}').then((r) => r.json())`);
        if (status.status === "done" || status.status === "failed") {
          report.testOutputDir = status.output_dir;
          report.testJobMode = status.video_mode;
          report.testSuccess = status.success_count;
          report.testNote = status.results && status.results[0] ? status.results[0].note : null;
          break;
        }
        await sleep(500);
      }
    }
    check("浏览器把 video_mode=fast 发给了服务", sentMode === "fast", `实际发出 ${sentMode}，随请求带出 ${sentBoxes} 个框；诊断 ${JSON.stringify(diag)}`);
    check("服务端回显 job.video_mode=fast", report.testJobMode === "fast", report.testJobMode);
    check("这份 UI 任务真的跑完了", report.testSuccess === 1, `成功 ${report.testSuccess} 项`);
    console.log(`     成品说明文案：${report.testNote || "（未取到）"}`);
  }

  // ---- 6b. 动态水印字段真的发给服务，并由服务回显 ----
  if (UPLOAD_FILE && fs.existsSync(UPLOAD_FILE)) {
    await sleep(900);
    // 上一份任务提交后 jobModal 会盖在页面上，原生点击会打到遮罩层
    await wc.executeJavaScript("(() => { const close = document.getElementById('jobCloseBtn'); if (close) close.click(); return true; })()");
    await sleep(400);
    await ensureVisible(win);
    const motionPoint3 = await switchPoint(win, "videoMotion");
    wc.sendInputEvent({ type: "mouseMove", x: motionPoint3.x, y: motionPoint3.y });
    wc.sendInputEvent({ type: "mouseDown", x: motionPoint3.x, y: motionPoint3.y, clickCount: 1 });
    wc.sendInputEvent({ type: "mouseUp", x: motionPoint3.x, y: motionPoint3.y, clickCount: 1 });
    await sleep(450);
    const motionLocked = await wc.executeJavaScript(`(() => ({
      motion: document.getElementById('videoMotion').checked,
      fastDisabled: document.getElementById('videoFastMode').disabled,
      fastChecked: document.getElementById('videoFastMode').checked,
    }))()`);
    check("打开动态水印后速度优先被锁住", motionLocked.motion === true && motionLocked.fastDisabled === true,
      JSON.stringify(motionLocked));

    await wc.executeJavaScript(`(() => {
      const original = window.fetch;
      window.__captured = null;
      window.fetch = async function (url, options) {
        const response = await original.apply(this, arguments);
        if (String(url).indexOf('/api/process') >= 0) {
          try {
            window.__captured = { body: options && options.body, job: await response.clone().json() };
          } catch (error) { window.__captured = { error: String(error) }; }
        }
        return response;
      };
      return true;
    })()`);
    await wc.executeJavaScript("document.getElementById('processBtn').click()");
    let capturedMotion = null;
    for (let i = 0; i < 40; i += 1) {
      await sleep(500);
      capturedMotion = await wc.executeJavaScript("window.__captured ? { body: window.__captured.body, id: window.__captured.job && window.__captured.job.id, motion: window.__captured.job && window.__captured.job.motion } : null");
      if (capturedMotion && capturedMotion.body) break;
    }
    check("浏览器把 motion=true 发给了服务", Boolean(capturedMotion && JSON.parse(capturedMotion.body).motion === true),
      capturedMotion ? capturedMotion.body.slice(0, 80) : "未捕获");
    check("服务端回显 job.motion=true", Boolean(capturedMotion && capturedMotion.motion === true),
      String(capturedMotion && capturedMotion.motion));
    if (capturedMotion && capturedMotion.id) {
      for (let i = 0; i < 60; i += 1) {
        const status = await wc.executeJavaScript(`fetch('/api/jobs/${capturedMotion.id}').then((r) => r.json())`);
        if (status.status === "done" || status.status === "failed") {
          // 这一趟也真写了一个 成品/ 批次，不在这里记下目录，跑完就会留在交付目录里
          report.motionOutputDir = status.output_dir;
          report.motionJobId = capturedMotion.id;
          break;
        }
        await sleep(500);
      }
    }
    await wc.executeJavaScript("(() => { const close = document.getElementById('jobCloseBtn'); if (close) close.click(); return true; })()");
    await wc.executeJavaScript("document.getElementById('videoMotion').click()");
    await sleep(300);
    const motionReset = await wc.executeJavaScript(`(() => ({
      motion: document.getElementById('videoMotion').checked,
      fastDisabled: document.getElementById('videoFastMode').disabled,
      stored: localStorage.getItem('jingyin.videoMotion'),
    }))()`);
    check("验完把动态水印关回默认", motionReset.motion === false && motionReset.fastDisabled === false
      && motionReset.stored === "0", JSON.stringify(motionReset));
  }

  // ---- 7. 重新加载页面：素材与手动画的 16 个框必须原样回来 ----
  if (UPLOAD_FILE && fs.existsSync(UPLOAD_FILE)) {
    await sleep(800); // 选框写 localStorage 是防抖的，等一下再走
    await wc.loadURL(URL_ARG);
    await sleep(1700);
    const back = await wc.executeJavaScript(`(() => ({
      rows: document.querySelectorAll('#fileList li').length,
      boxes: document.querySelectorAll('#regionList li').length,
      label: document.getElementById('regionCount').textContent.trim(),
      canvas: (document.getElementById('previewImage') || {}).src || '',
      toast: (() => { const s = document.getElementById('toastStack'); return s && s.lastElementChild ? s.lastElementChild.textContent.trim() : null; })(),
    }))()`);
    check("重新打开后素材还在列表", back.rows === 1, `${back.rows} 份`);
    check("手动画的 16 个框原样回来", back.boxes === 16 && back.label.replace(/\s/g, "") === "16/16", `${back.boxes} 个 · ${back.label}`);
    check("恢复提示告诉用户框回来了", /已恢复上次框选/.test(String(back.toast)), String(back.toast));
    check("画布用的还是本机那一份素材", /\/api\/items\/[0-9a-f]+\/display/.test(String(back.canvas)), String(back.canvas).slice(0, 90));
  }

  // ---- 8. 低对比疑似提示：虚线候选 → 采纳 → 不写 localStorage ----
  if (UPLOAD2 && fs.existsSync(UPLOAD2) && report.uploadedItemId) {
    await wc.executeJavaScript(`(() => { const i = document.getElementById('fileInput'); i.hidden = false; i.style.display='block'; return true; })()`);
    const dbg = wc.debugger;
    dbg.attach("1.3");
    await dbg.sendCommand("DOM.enable");
    const doc = await dbg.sendCommand("DOM.getDocument");
    const node = await dbg.sendCommand("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#fileInput" });
    await dbg.sendCommand("DOM.setFileInputFiles", { files: [UPLOAD2], nodeId: node.nodeId });
    await sleep(2800);
    const listState = await wc.executeJavaScript(`(() => {
      const lis = Array.from(document.querySelectorAll('#fileList li'));
      return { rows: lis.length, ids: lis.map((li) => li.dataset.id) };
    })()`);
    report.secondItemId = listState.ids.find((id) => id && id !== report.uploadedItemId) || null;
    check("低对比夹具也进了列表", listState.rows === 2 && report.secondItemId, `${listState.rows} 份`);

    await ensureVisible(win);
    await wc.executeJavaScript(`(() => {
      const rows = document.querySelectorAll('#fileList li');
      if (rows[1]) rows[1].click();
      return true;
    })()`);
    await sleep(700);
    await wc.executeJavaScript("document.getElementById('autoDetectBtn').click()");
    await sleep(3000);
    const sug = await wc.executeJavaScript(`(() => {
      const block = document.getElementById('suggestionBlock');
      const items = Array.from(document.querySelectorAll('#suggestionList li'));
      const stack = document.getElementById('toastStack');
      return {
        visible: Boolean(block) && !block.hidden,
        count: items.length,
        countLabel: document.getElementById('suggestionCount').textContent.trim(),
        badges: items.map((li) => (li.querySelector('.suggestion-badge') || {}).textContent || ''),
        note: block ? (block.querySelector('.suggestion-note') || {}).textContent || '' : '',
        regions: document.querySelectorAll('#regionList li').length,
        regionLabel: document.getElementById('regionCount').textContent.trim(),
        toast: stack && stack.lastElementChild ? stack.lastElementChild.textContent.trim() : null,
      };
    })()`);
    check("自动找水印后出现「疑似水印」区", sug.visible && sug.count >= 2, `${sug.count} 条 · ${sug.countLabel}`);
    check("每条都带「疑似」徽标", sug.badges.length > 0 && sug.badges.every((t) => t.trim() === "疑似"), sug.badges.join(","));
    check("说明文案明确要求核对", /疑似，请核对/.test(sug.note) && /低对比/.test(sug.note), sug.note.slice(0, 40));
    check("疑似不自动变成选框", sug.regions === 0 && sug.regionLabel.replace(/\s/g, "") === "0/16", sug.regionLabel);
    check("提示也告诉用户有几处疑似", /疑似/.test(String(sug.toast)), sug.toast);
    await shot(win, path.join(path.dirname(OUT), "_ui_suggestion.png"));

    await wc.executeJavaScript("document.querySelector('#suggestionList li .suggestion-adopt').click()");
    await sleep(700);
    const adopted = await wc.executeJavaScript(`(() => {
      const stack = document.getElementById('toastStack');
      return {
        regions: document.querySelectorAll('#regionList li').length,
        regionLabel: document.getElementById('regionCount').textContent.trim(),
        count: document.querySelectorAll('#suggestionList li').length,
        toast: stack && stack.lastElementChild ? stack.lastElementChild.textContent.trim() : null,
      };
    })()`);
    check("采纳一条后变成正式选框", adopted.regions === 1 && adopted.regionLabel.replace(/\s/g, "") === "1/16", adopted.regionLabel);
    check("采纳后疑似条目减一", adopted.count === sug.count - 1, `${sug.count} → ${adopted.count}`);
    check("采纳后仍提醒核对选框", /核对/.test(String(adopted.toast)), adopted.toast);

    await sleep(900);
    await wc.loadURL(URL_ARG);
    await sleep(1800);
    const restored = await wc.executeJavaScript(`(() => ({
      rows: document.querySelectorAll('#fileList li').length,
      blockHidden: document.getElementById('suggestionBlock').hidden,
    }))()`);
    check("刷新后疑似提示不复活", restored.blockHidden === true && restored.rows === 2, JSON.stringify(restored));
    await wc.executeJavaScript(`(() => {
      const rows = document.querySelectorAll('#fileList li');
      if (rows[1]) rows[1].click();
      return true;
    })()`);
    await sleep(700);
    const adoptedBack = await wc.executeJavaScript(`(() => ({
      regionLabel: document.getElementById('regionCount').textContent.trim(),
      blockHidden: document.getElementById('suggestionBlock').hidden,
    }))()`);
    check("采纳的选框刷新后还在", adoptedBack.regionLabel.replace(/\s/g, "") === "1/16", adoptedBack.regionLabel);

    await wc.executeJavaScript("document.getElementById('autoDetectBtn').click()");
    await sleep(3000);
    const again = await wc.executeJavaScript(`(() => ({
      visible: !document.getElementById('suggestionBlock').hidden,
      count: document.querySelectorAll('#suggestionList li').length,
    }))()`);
    check("再次自动识别能重新出疑似区", again.visible && again.count >= 2, `${again.count} 条`);
    await wc.executeJavaScript("document.getElementById('suggestionClearBtn').click()");
    await sleep(500);
    const cleared = await wc.executeJavaScript(`(() => ({
      hidden: document.getElementById('suggestionBlock').hidden,
      items: document.querySelectorAll('#suggestionList li').length,
    }))()`);
    check("全部忽略后疑似区收起", cleared.hidden === true && cleared.items === 0, JSON.stringify(cleared));
    dbg.detach();
  } else {
    console.log("SKIP 低对比疑似提示部分：没有 --upload2 指定的合成夹具");
  }

  // ---- 8b. 停止这一批：运行中出现按钮 → 真点击 → 服务端判 cancelled → 界面渲染「已停止」 ----
  // 需要一段比图片慢得多的素材才抓得住「正在跑」这个窗口，用 work/qa/_real_oceans.mp4
  // （公开正片，画质优先掩膜路线实测 15–16s/段）。不读用户作品。
  if (UPLOAD_VIDEO && fs.existsSync(UPLOAD_VIDEO)) {
    await sleep(600);
    await wc.executeJavaScript("(() => { const close = document.getElementById('jobCloseBtn'); if (close) close.click(); return true; })()");
    await sleep(400);
    // 前面几步把开关拨过（速度优先、动态水印都点过），这里要的是「默认档」下的真实耗时，
    // 所以先把 localStorage 复位到出厂状态并重新加载页面，再断言它确实回到了默认档。
    await wc.executeJavaScript("(() => { localStorage.setItem('jingyin.videoMode', 'quality'); localStorage.setItem('jingyin.videoMotion', '0'); return true; })()");
    await wc.loadURL(URL_ARG);
    await sleep(1800);
    await ensureVisible(win);
    const dbgV = wc.debugger;
    try { dbgV.attach("1.3"); } catch (error) { /* 上一段还挂着，直接用 */ }
    await dbgV.sendCommand("DOM.enable");
    const docV = await dbgV.sendCommand("DOM.getDocument");
    await wc.executeJavaScript(`(() => { const i = document.getElementById('fileInput'); i.hidden = false; i.style.display='block'; return true; })()`);
    const nodeV = await dbgV.sendCommand("DOM.querySelector", { nodeId: docV.root.nodeId, selector: "#fileInput" });
    await dbgV.sendCommand("DOM.setFileInputFiles", { files: [UPLOAD_VIDEO], nodeId: nodeV.nodeId });
    await sleep(3500);
    const videoRow = await wc.executeJavaScript(`(() => {
      const rows = Array.from(document.querySelectorAll('#fileList li'));
      return { count: rows.length, last: rows.length ? rows[rows.length - 1].dataset.id : null,
               kinds: rows.map((li) => (li.querySelector('.region-tag') || {}).textContent || '') };
    })()`);
    report.videoItemId = videoRow.last;
    check("视频素材进了批量列表", videoRow.count >= 3 && Boolean(videoRow.last), `${videoRow.count} 份 · item=${videoRow.last}`);

    const videoCanvas = await wc.executeJavaScript(`(async () => {
      const li = document.querySelector('#fileList li:last-child');
      if (li) li.click();
      await new Promise((r) => setTimeout(r, 900));
      const r = document.getElementById('overlayCanvas').getBoundingClientRect();
      const fast = document.getElementById('videoFastMode');
      const motion = document.getElementById('videoMotion');
      return { left: r.left, top: r.top, width: r.width, height: r.height,
               fastChecked: fast.checked, motionChecked: motion.checked,
               stored: localStorage.getItem('jingyin.videoMode') };
    })()`);
    check("视频按默认画质优先（速度优先与动态水印都关着）",
      videoCanvas.fastChecked === false && videoCanvas.motionChecked === false,
      JSON.stringify({ fast: videoCanvas.fastChecked, motion: videoCanvas.motionChecked, stored: videoCanvas.stored }));
    check("视频画布有实际尺寸", videoCanvas.width > 120 && videoCanvas.height > 60, JSON.stringify(videoCanvas));

    await drag(win,
      videoCanvas.left + videoCanvas.width * 0.72, videoCanvas.top + videoCanvas.height * 0.72,
      videoCanvas.left + videoCanvas.width * 0.96, videoCanvas.top + videoCanvas.height * 0.95);
    const videoBoxes = await wc.executeJavaScript("document.querySelectorAll('#regionList li').length");
    check("能在视频上画出一个选框", videoBoxes === 1, `${videoBoxes} 个`);

    await wc.executeJavaScript(`(() => {
      const original = window.fetch;
      window.__cancelCalls = [];
      window.__jobId = null;
      window.fetch = async function (url, options) {
        const response = await original.apply(this, arguments);
        if (String(url).indexOf('/cancel') >= 0) {
          window.__cancelCalls.push({ url: String(url), method: (options && options.method) || 'GET' });
        }
        if (String(url).indexOf('/api/process') >= 0) {
          try { window.__jobId = (await response.clone().json()).id; } catch (error) { window.__jobId = null; }
        }
        return response;
      };
      return true;
    })()`);
    await wc.executeJavaScript("document.getElementById('processBtn').click()");

    let cancelRow = null;
    for (let i = 0; i < 40; i += 1) {
      await sleep(250);
      cancelRow = await wc.executeJavaScript(`(() => {
        const modal = document.getElementById('jobModal');
        const row = document.getElementById('jobCancelRow');
        const btn = document.getElementById('jobCancelBtn');
        const r = btn.getBoundingClientRect();
        return { modalHidden: modal.hidden, rowHidden: row.hidden,
                 visible: !modal.hidden && !row.hidden && r.width > 20 && r.top >= 0 && r.bottom <= innerHeight,
                 x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2),
                 title: document.getElementById('jobTitle').textContent.trim(),
                 stats: document.getElementById('jobStats').textContent.trim(),
                 tasks: document.querySelectorAll('#taskProgressList .task-row').length };
      })()`);
      if (cancelRow.visible) break;
    }
    check("运行中的任务弹窗显示「停止这一批」", cancelRow.visible,
      JSON.stringify({ title: cancelRow.title, stats: cancelRow.stats, tasks: cancelRow.tasks }));

    wc.sendInputEvent({ type: "mouseMove", x: cancelRow.x, y: cancelRow.y });
    wc.sendInputEvent({ type: "mouseDown", x: cancelRow.x, y: cancelRow.y, clickCount: 1 });
    wc.sendInputEvent({ type: "mouseUp", x: cancelRow.x, y: cancelRow.y, clickCount: 1 });
    await sleep(250);
    const afterCancel = await wc.executeJavaScript(`(() => ({
      disabled: document.getElementById('jobCancelBtn').disabled,
      rowHidden: document.getElementById('jobCancelRow').hidden,
      title: document.getElementById('jobTitle').textContent.trim(),
      message: document.getElementById('jobMessage').textContent.trim(),
      calls: window.__cancelCalls,
      jobId: window.__jobId,
      stopped: document.querySelectorAll('#taskProgressList .task-row.is-cancelled').length,
    }))()`);
    check("点一下真的发出 POST /api/jobs/<id>/cancel",
      afterCancel.calls.length === 1 && afterCancel.calls[0].method === "POST"
        && /\/api\/jobs\/[0-9a-f-]+\/cancel$/.test(afterCancel.calls[0].url),
      JSON.stringify(afterCancel.calls));
    check("发出后按钮进入防重复状态（禁用或整行收起）",
      afterCancel.disabled === true || afterCancel.rowHidden === true,
      JSON.stringify({ disabled: afterCancel.disabled, rowHidden: afterCancel.rowHidden }));
    check("标题按实测状态切到「正在停止」或「已停止这一批」",
      /正在停止|已停止这一批/.test(afterCancel.title), afterCancel.title);

    let jobIdForCancel = afterCancel.jobId;
    for (let i = 0; i < 40 && !jobIdForCancel; i += 1) {
      await sleep(250);
      jobIdForCancel = await wc.executeJavaScript("window.__jobId");
    }
    check("停止请求打的正是这一批的任务 id", Boolean(jobIdForCancel)
      && afterCancel.calls.length > 0 && String(afterCancel.calls[0].url).indexOf(jobIdForCancel) >= 0,
      `${jobIdForCancel} ← ${afterCancel.calls.length ? afterCancel.calls[0].url : "未发出"}`);

    let finalJob = null;
    for (let i = 0; jobIdForCancel && i < 120; i += 1) {
      finalJob = await wc.executeJavaScript(`fetch('/api/jobs/${jobIdForCancel}').then((r) => r.json())`);
      if (finalJob && finalJob.status !== "running" && finalJob.status !== "queued") break;
      await sleep(500);
    }
    // 界面自己每 800ms 轮一次；服务端刚进终态时 DOM 还是上一轮的「正在停止」，
    // 要等它追上再读——否则量到的是脚本的读取时机，不是用户看到的界面。
    let uiAtEnd = null;
    for (let i = 0; i < 30; i += 1) {
      uiAtEnd = await wc.executeJavaScript(`(() => ({
        title: document.getElementById('jobTitle').textContent.trim(),
        message: document.getElementById('jobMessage').textContent.trim(),
        percent: document.getElementById('jobPercent').textContent.trim(),
        stats: document.getElementById('jobStats').textContent.trim(),
        resultStats: document.getElementById('resultStats').textContent.trim(),
        progressHidden: document.getElementById('jobProgressSection').hidden,
        resultsHidden: document.getElementById('resultsSection').hidden,
        zipHidden: document.getElementById('zipDownloadBtn').hidden,
        cancelledRows: document.querySelectorAll('#taskProgressList .task-row.is-cancelled').length,
        errorRows: document.querySelectorAll('#taskProgressList .task-row.is-error').length,
        stoppedHidden: document.getElementById('jobStoppedSection').hidden,
        stoppedTitle: document.querySelector('#jobStoppedSection h3').textContent.trim(),
        stoppedItems: Array.from(document.querySelectorAll('#jobStoppedList li')).map((li) => li.textContent.trim()),
        stoppedNote: document.querySelector('.job-stopped-note').textContent.trim(),
        stoppedBeforeFailed: Boolean(
          document.getElementById('jobStoppedSection').compareDocumentPosition(
            document.getElementById('jobFailedSection')) & Node.DOCUMENT_POSITION_FOLLOWING),
      }))()`);
      const caught = /已停止这一批|处理完成|这次没有成功/.test(uiAtEnd.title) && uiAtEnd.percent === "100%";
      if (caught) break;
      await sleep(400);
    }
    if (finalJob) {
      report.cancelJobId = finalJob.id;
      report.cancelOutputDir = finalJob.output_dir;
    }
    check("服务端把这批判成 cancelled（不是 failed）", Boolean(finalJob) && finalJob.status === "cancelled",
      finalJob ? `${finalJob.status} · ${uiAtEnd.title}` : "未取到任务");
    check("计数口径不变：completed==total 且三终态相加==total",
      Boolean(finalJob) && finalJob.completed === finalJob.total
        && finalJob.success_count + finalJob.failed_count + finalJob.cancelled_count === finalJob.total,
      finalJob ? `${finalJob.completed}/${finalJob.total} · 成功 ${finalJob.success_count} 失败 ${finalJob.failed_count} 已停止 ${finalJob.cancelled_count}` : "");
    check("确实有素材被停下来", Boolean(finalJob) && finalJob.cancelled_count >= 1,
      finalJob ? `cancelled_count=${finalJob.cancelled_count}` : "");
    check("界面渲染出「已停止」的任务行（不是红色未完成）",
      uiAtEnd.cancelledRows >= 1 && uiAtEnd.errorRows === 0,
      JSON.stringify({ cancelledRows: uiAtEnd.cancelledRows, errorRows: uiAtEnd.errorRows, progressHidden: uiAtEnd.progressHidden }));
    check("标题与说明如实写「已停止」",
      /已停止这一批/.test(uiAtEnd.title) && /已停止/.test(uiAtEnd.message),
      `${uiAtEnd.title}｜${uiAtEnd.message.slice(0, 60)}`);
    check("进度收在 100%（停止也算走完）", uiAtEnd.percent === "100%", `${uiAtEnd.percent} · ${uiAtEnd.stats}`);
    check("停下来后已完成的成品仍然给到下载入口",
      uiAtEnd.resultsHidden === false && uiAtEnd.zipHidden === false && /已停止 \d+ 项/.test(uiAtEnd.resultStats),
      `结果区${uiAtEnd.resultsHidden ? "收起" : "展开"} · ZIP${uiAtEnd.zipHidden ? "隐藏" : "可点"} · ${uiAtEnd.resultStats}`);

    // 进度区在终态整块收起，逐份证据只剩这一张清单：它必须自己说清「哪几份没出成品、是没开工还是跑一半」
    const cancelledTasks = finalJob ? finalJob.tasks.filter((t) => t.status === "cancelled") : [];
    const halfStopped = cancelledTasks.filter((t) => t.started).length;
    check("停下来后界面逐份列出「还没做完的文件」（不再只有统计行）",
      uiAtEnd.stoppedHidden === false && uiAtEnd.stoppedItems.length === cancelledTasks.length
        && cancelledTasks.length >= 1,
      JSON.stringify({ hidden: uiAtEnd.stoppedHidden, items: uiAtEnd.stoppedItems.length,
        cancelled: cancelledTasks.length }));
    check("清单里的文件名与被停止的素材一一对应",
      uiAtEnd.stoppedItems.length > 0 && [...new Set(uiAtEnd.stoppedItems.map((s) => s.split("：")[0]))].join("|")
        === [...new Set(cancelledTasks.map((t) => t.name))].join("|"),
      uiAtEnd.stoppedItems.join(" / ").slice(0, 120));
    check("清单如实分开「处理到一半被停止」与「还没开始就被停止」",
      uiAtEnd.stoppedItems.filter((s) => /处理到一半被停止$/.test(s)).length === halfStopped
        && uiAtEnd.stoppedItems.filter((s) => /还没开始就被停止$/.test(s)).length === cancelledTasks.length - halfStopped,
      JSON.stringify({ 跑一半: halfStopped, 未开工: cancelledTasks.length - halfStopped, items: uiAtEnd.stoppedItems }));
    check("这份清单是独立的琥珀色区块、排在红色失败区块之前，并指向续跑按钮",
      uiAtEnd.stoppedTitle === "还没做完的文件" && uiAtEnd.stoppedBeforeFailed === true
        && /继续处理剩下的/.test(uiAtEnd.stoppedNote),
      `${uiAtEnd.stoppedTitle} · ${uiAtEnd.stoppedNote.slice(0, 40)}`);

    // ---- 8c. 续跑：停止后点「继续处理剩下的 N 项」→ 同一个任务、同一个文件夹，已完成的成品不重做 ----
    // 复用 8b 那一批：停下时已完成的是合成图片，剩下的是画质优先要跑十几秒的视频。
    const snapshotDir = (dir) => {
      const entries = {};
      if (!dir || !fs.existsSync(dir)) return entries;
      for (const name of fs.readdirSync(dir)) {
        if (name.startsWith(".preview_") || name.startsWith(".")) continue;
        const file = path.join(dir, name);
        try {
          if (!fs.statSync(file).isFile()) continue;
          entries[name] = crypto.createHash("md5").update(fs.readFileSync(file)).digest("hex");
        } catch (error) {
          // 正在写的文件读不到就跳过，不能让验收脚本自己崩
        }
      }
      return entries;
    };
    let resumeRow = null;
    for (let i = 0; i < 30; i += 1) {
      resumeRow = await wc.executeJavaScript(`(() => {
        const modal = document.getElementById('jobModal');
        const row = document.getElementById('jobResumeRow');
        const btn = document.getElementById('jobResumeBtn');
        const r = btn.getBoundingClientRect();
        const note = row.querySelector('.job-cancel-note');
        return { rowHidden: row.hidden,
                 visible: !modal.hidden && !row.hidden && r.width > 20 && r.top >= 0 && r.bottom <= innerHeight,
                 x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2),
                 label: btn.textContent.trim(), disabled: btn.disabled,
                 note: note ? note.textContent.trim() : '' };
      })()`);
      if (resumeRow.visible) break;
      await sleep(300);
    }
    check("停下来后界面给出「继续处理剩下的 N 项」",
      resumeRow.visible && /继续处理剩下的 \d+ 项/.test(resumeRow.label) && resumeRow.disabled === false,
      JSON.stringify({ label: resumeRow.label, hidden: resumeRow.rowHidden, note: resumeRow.note.slice(0, 30) }));

    const beforeResume = snapshotDir(finalJob ? finalJob.output_dir : null);
    report.resumeBeforeCount = Object.keys(beforeResume).length;
    check("续跑前磁盘上已有停止时做完成的成品",
      report.resumeBeforeCount >= 1, `${report.resumeBeforeCount} 个文件`);

    await wc.executeJavaScript(`(() => {
      const previous = window.fetch;
      window.__resumeCalls = [];
      window.fetch = async function (url, options) {
        if (String(url).indexOf('/resume') >= 0) {
          window.__resumeCalls.push({ url: String(url), method: (options && options.method) || 'GET' });
        }
        return previous.apply(this, arguments);
      };
      return true;
    })()`);
    wc.sendInputEvent({ type: "mouseMove", x: resumeRow.x, y: resumeRow.y });
    wc.sendInputEvent({ type: "mouseDown", x: resumeRow.x, y: resumeRow.y, clickCount: 1 });
    wc.sendInputEvent({ type: "mouseUp", x: resumeRow.x, y: resumeRow.y, clickCount: 1 });
    await sleep(600);
    const afterResume = await wc.executeJavaScript(`(() => ({
      calls: window.__resumeCalls,
      title: document.getElementById('jobTitle').textContent.trim(),
      progressHidden: document.getElementById('jobProgressSection').hidden,
      resultsHidden: document.getElementById('resultsSection').hidden,
      resumeHidden: document.getElementById('jobResumeRow').hidden,
      stoppedHidden: document.getElementById('jobStoppedSection').hidden,
      cancelCalls: window.__cancelCalls.length,
    }))()`);
    check("点一下真的发出 POST /api/jobs/<id>/resume（且只发一次）",
      afterResume.calls.length === 1 && afterResume.calls[0].method === "POST"
        && /\/api\/jobs\/[0-9a-f-]+\/resume$/.test(afterResume.calls[0].url),
      JSON.stringify(afterResume.calls));
    check("续跑打的正是停下来的那一批",
      Boolean(jobIdForCancel) && afterResume.calls.length > 0
        && String(afterResume.calls[0].url).indexOf(jobIdForCancel) >= 0,
      `${jobIdForCancel} ← ${afterResume.calls.length ? afterResume.calls[0].url : "未发出"}`);
    check("续跑期间不会重复发停止请求", afterResume.cancelCalls === 1, `cancel 调用 ${afterResume.cancelCalls} 次`);
    check("续跑一开工就收起「还没做完的文件」，不把旧状态留在界面上",
      afterResume.stoppedHidden === true, `stoppedHidden=${afterResume.stoppedHidden}`);
    check("续跑后界面退回处理中（进度展开、结果与续跑入口收起）",
      /正在准备处理|正在去水印/.test(afterResume.title) && afterResume.progressHidden === false
        && afterResume.resultsHidden === true && afterResume.resumeHidden === true,
      JSON.stringify({ title: afterResume.title, progressHidden: afterResume.progressHidden,
        resultsHidden: afterResume.resultsHidden, resumeHidden: afterResume.resumeHidden }));

    let resumedJob = null;
    for (let i = 0; jobIdForCancel && i < 160; i += 1) {
      resumedJob = await wc.executeJavaScript(`fetch('/api/jobs/${jobIdForCancel}').then((r) => r.json())`);
      if (resumedJob && resumedJob.status !== "running" && resumedJob.status !== "queued") break;
      await sleep(500);
    }
    let uiResumed = null;
    for (let i = 0; i < 40; i += 1) {
      uiResumed = await wc.executeJavaScript(`(() => ({
        title: document.getElementById('jobTitle').textContent.trim(),
        message: document.getElementById('jobMessage').textContent.trim(),
        percent: document.getElementById('jobPercent').textContent.trim(),
        stats: document.getElementById('jobStats').textContent.trim(),
        resultStats: document.getElementById('resultStats').textContent.trim(),
        cards: document.querySelectorAll('#resultGrid .result-card').length,
        zipHidden: document.getElementById('zipDownloadBtn').hidden,
        resumeHidden: document.getElementById('jobResumeRow').hidden,
        stoppedHidden: document.getElementById('jobStoppedSection').hidden,
        stoppedItems: document.querySelectorAll('#jobStoppedList li').length,
        errorRows: document.querySelectorAll('#taskProgressList .task-row.is-error').length,
      }))()`);
      if (/处理完成|这次没有成功/.test(uiResumed.title) && uiResumed.percent === "100%") break;
      await sleep(400);
    }
    const afterSnapshot = snapshotDir(finalJob ? finalJob.output_dir : null);
    const keptSame = Object.keys(beforeResume).every((name) => afterSnapshot[name] === beforeResume[name]);
    check("续跑把这批判成 done：成功数等于份数、没有停止项也没有失败",
      Boolean(resumedJob) && resumedJob.status === "done"
        && resumedJob.success_count === resumedJob.total
        && resumedJob.cancelled_count === 0 && resumedJob.failed_count === 0,
      resumedJob ? `${resumedJob.status} · 成功 ${resumedJob.success_count}/${resumedJob.total} 已停止 ${resumedJob.cancelled_count} 失败 ${resumedJob.failed_count}` : "未取到任务");
    check("续跑记账：resume_count 累加、剩余归零、不再可续跑",
      Boolean(resumedJob) && resumedJob.resume_count >= 1
        && resumedJob.remaining_count === 0 && resumedJob.resumable === false,
      resumedJob ? `resume_count=${resumedJob.resume_count} 剩余=${resumedJob.remaining_count} resumable=${resumedJob.resumable}` : "");
    check("成品仍落在同一个文件夹（没另开一批）",
      Boolean(resumedJob) && Boolean(finalJob) && resumedJob.output_dir === finalJob.output_dir,
      resumedJob ? String(resumedJob.output_dir).slice(-28) : "");
    check("停止时已完成的成品逐字节没变（确实没重做）",
      keptSame && Object.keys(beforeResume).length > 0,
      JSON.stringify({ before: Object.keys(beforeResume), after: Object.keys(afterSnapshot) }));
    check("续跑把剩下的素材补出了新成品",
      Object.keys(afterSnapshot).length > Object.keys(beforeResume).length,
      `${Object.keys(beforeResume).length} → ${Object.keys(afterSnapshot).length} 个文件`);
    check("界面把全部成品给到结果区、ZIP 仍可下载",
      Boolean(resumedJob) && uiResumed.cards === resumedJob.total && uiResumed.zipHidden === false
        && /全部完成/.test(uiResumed.resultStats),
      JSON.stringify({ cards: uiResumed.cards, total: resumedJob ? resumedJob.total : null, zipHidden: uiResumed.zipHidden, resultStats: uiResumed.resultStats }));
    check("续跑完成后标题「处理完成」且不再显示续跑入口",
      /处理完成/.test(uiResumed.title) && uiResumed.resumeHidden === true && uiResumed.errorRows === 0,
      `${uiResumed.title}｜${uiResumed.message.slice(0, 40)} · resumeHidden=${uiResumed.resumeHidden}`);
    check("全部做完后「还没做完的文件」清单收起且不残留条目",
      uiResumed.stoppedHidden === true && uiResumed.stoppedItems === 0,
      `stoppedHidden=${uiResumed.stoppedHidden} items=${uiResumed.stoppedItems}`);
    report.resumeOutputDir = finalJob ? finalJob.output_dir : null;

    await wc.executeJavaScript("(() => { const close = document.getElementById('jobCloseBtn'); if (close) close.click(); return true; })()");
    await sleep(300);
    const leftover = await wc.executeJavaScript(`(() => {
      const stack = document.getElementById('toastStack');
      return stack && stack.lastElementChild ? stack.lastElementChild.textContent.trim() : null;
    })()`);
    console.log(`     停止后的提示/统计：${leftover || "—"} · ${uiAtEnd.resultStats || "—"}`);
    dbgV.detach();
  } else {
    console.log("SKIP 停止这一批部分：没有 --uploadVideo 指定的视频素材");
  }

  // ---- 9. 移动端宽度不裁切 ----
  const mobileWidth = await setViewport(win, 390, 844);
  await ensureVisible(win);
  await sleep(900);
  const mobile = await wc.executeJavaScript(`(() => {
    const row = document.getElementById('videoFastMode').closest('.video-switch');
    const rect = row.getBoundingClientRect();
    return {
      innerW: window.innerWidth,
      scrollW: document.documentElement.scrollWidth,
      clientW: document.documentElement.clientWidth,
      rowLeft: Math.round(rect.left), rowWidth: Math.round(rect.width),
      visible: rect.top < window.innerHeight && rect.bottom > 0,
      canScroll: document.scrollingElement.scrollTop !== undefined,
    };
  })()`);
  check("窗口真的切到了移动宽度", mobileWidth <= 394 && mobile.innerW <= 394, `innerWidth=${mobile.innerW}`);
  check("390 宽不出横向溢出", mobile.scrollW <= mobile.clientW + 1, JSON.stringify(mobile));
  await wc.executeJavaScript("window.scrollTo(0, document.body.scrollHeight)");
  await sleep(300);
  const bottomTop = await wc.executeJavaScript("document.scrollingElement.scrollTop");
  check("移动端整页可滚动", bottomTop > 4, `scrollTop=${bottomTop}`);
  await wc.executeJavaScript("document.getElementById('videoFastMode').closest('.video-switch').scrollIntoView({block:'center'})");
  await sleep(300);
  const rowInView = await wc.executeJavaScript(`(() => {
    const el = document.getElementById('videoFastMode').closest('.video-switch');
    const r = el.getBoundingClientRect();
    return { top: Math.round(r.top), bottom: Math.round(r.bottom), right: Math.round(r.right),
             inView: r.top >= 0 && r.bottom <= innerHeight, withinWidth: r.right <= innerWidth };
  })()`);
  check("移动端滚一下就能拍到开关", rowInView.inView, JSON.stringify(rowInView));
  check("移动端开关不被裁掉", rowInView.withinWidth, `right=${rowInView.right} / 视口宽`);
  await shot(win, path.join(path.dirname(OUT), "_ui_mobile.png"));
}

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    show: true,
    webPreferences: { sandbox: false },
  });
  try {
    await run(win);
  } catch (error) {
    check("脚本异常", false, String((error && (error.stack || error.message)) || error));
  }
  win.setAlwaysOnTop(false);
  fs.writeFileSync(OUT, JSON.stringify(report, null, 2));
  console.log(`结果：${report.failed.length === 0 ? "全部通过" : report.failed.length + " 项不通过 -> " + report.failed.join("、")}`);
  app.exit(report.failed.length === 0 ? 0 : 1);
});
