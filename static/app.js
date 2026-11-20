(() => {
  'use strict';

  const TOKEN = document.querySelector('meta[name="app-token"]')?.getAttribute('content') || '';
  const PHOTO_READY =
    document.querySelector('meta[name="photo-ready"]')?.getAttribute('content') === '1';
  const IMAGE_EXT = new Set(['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff']);
  const VIDEO_EXT = new Set(['mp4', 'm4v', 'mov', 'webm', 'mkv', 'avi']);
  const MAX_ITEMS = 120;
  const MAX_REGIONS = 16;
  const ZOOM_STEPS = [1, 1.25, 1.5, 2, 2.5, 3, 4];
  const REGIONS_KEY = 'jingyin.regions.v1';
  const TEMPLATES_KEY = 'jingyin.templates.v1';
  const MAX_TEMPLATES = 10;
  const UPLOAD_BUSY = '正在上传，请稍等';

  const $ = (id) => document.getElementById(id);

  const el = {
    appShell: document.querySelector('.app-shell'),
    dropZone: $('dropZone'),
    fileInput: $('fileInput'),
    folderInput: $('folderInput'),
    pickFilesBtn: $('pickFilesBtn'),
    pickFolderBtn: $('pickFolderBtn'),
    uploadStatus: $('uploadStatus'),
    uploadText: $('uploadText'),
    uploadPercent: $('uploadPercent'),
    uploadBar: $('uploadBar'),
    checkAll: $('checkAll'),
    clearFilesBtn: $('clearFilesBtn'),
    fileList: $('fileList'),
    fileListEmpty: $('fileListEmpty'),
    fileCount: $('fileCount'),
    checkedCount: $('checkedCount'),
    emptyAddBtn: $('emptyAddBtn'),
    currentKindBadge: $('currentKindBadge'),
    currentName: $('currentName'),
    currentMeta: $('currentMeta'),
    autoDetectBtn: $('autoDetectBtn'),
    undoRegionBtn: $('undoRegionBtn'),
    stageArea: $('stageArea'),
    stageEmpty: $('stageEmpty'),
    stageSpacer: $('stageSpacer'),
    stageLoading: $('stageLoading'),
    mediaFrame: $('mediaFrame'),
    previewImage: $('previewImage'),
    previewVideo: $('previewVideo'),
    videoFallback: $('videoFallback'),
    posterImage: $('posterImage'),
    overlayCanvas: $('overlayCanvas'),
    drawHint: $('drawHint'),
    videoControls: $('videoControls'),
    videoPlayBtn: $('videoPlayBtn'),
    videoTime: $('videoTime'),
    videoSeek: $('videoSeek'),
    videoBackBtn: $('videoBackBtn'),
    videoForwardBtn: $('videoForwardBtn'),
    zoomTools: $('zoomTools'),
    zoomOutBtn: $('zoomOutBtn'),
    zoomLabel: $('zoomLabel'),
    zoomInBtn: $('zoomInBtn'),
    zoomFitBtn: $('zoomFitBtn'),
    videoErrorNote: $('videoErrorNote'),
    methodHelpBtn: $('methodHelpBtn'),
    photoMethodCard: $('photoMethodCard'),
    regionCount: $('regionCount'),
    videoFastMode: $('videoFastMode'),
    videoMotion: $('videoMotion'),
    clearRegionsBtn: $('clearRegionsBtn'),
    regionEmpty: $('regionEmpty'),
    regionList: $('regionList'),
    suggestionBlock: $('suggestionBlock'),
    suggestionCount: $('suggestionCount'),
    suggestionList: $('suggestionList'),
    suggestionClearBtn: $('suggestionClearBtn'),
    templateBlock: $('templateBlock'),
    templateCount: $('templateCount'),
    templateList: $('templateList'),
    saveTemplateBtn: $('saveTemplateBtn'),
    applyAllBtn: $('applyAllBtn'),
    batchReady: $('batchReady'),
    batchSkipped: $('batchSkipped'),
    processBtn: $('processBtn'),
    processBtnLabel: $('processBtnLabel'),
    processHint: $('processHint'),
    helpBtn: $('helpBtn'),
    jobChip: $('jobChip'),
    jobChipSpinner: $('jobChipSpinner'),
    jobChipText: $('jobChipText'),
    shutdownBtn: $('shutdownBtn'),
    helpModal: $('helpModal'),
    methodModal: $('methodModal'),
    jobModal: $('jobModal'),
    jobStatusIcon: $('jobStatusIcon'),
    jobTitle: $('jobTitle'),
    jobMessage: $('jobMessage'),
    jobCloseBtn: $('jobCloseBtn'),
    jobProgressSection: $('jobProgressSection'),
    jobStats: $('jobStats'),
    jobPercent: $('jobPercent'),
    jobBar: $('jobBar'),
    taskProgressList: $('taskProgressList'),
    jobCancelRow: $('jobCancelRow'),
    jobCancelBtn: $('jobCancelBtn'),
    jobResumeRow: $('jobResumeRow'),
    jobResumeBtn: $('jobResumeBtn'),
    resultsSection: $('resultsSection'),
    resultSummary: $('resultSummary'),
    resultStats: $('resultStats'),
    outputPathText: $('outputPathText'),
    copyPathBtn: $('copyPathBtn'),
    resultGrid: $('resultGrid'),
    openFolderBtn: $('openFolderBtn'),
    zipDownloadBtn: $('zipDownloadBtn'),
    closeResultsBtn: $('closeResultsBtn'),
    jobStoppedSection: $('jobStoppedSection'),
    jobStoppedList: $('jobStoppedList'),
    jobFailedSection: $('jobFailedSection'),
    jobFailedList: $('jobFailedList'),
    viewerModal: $('viewerModal'),
    viewerTitle: $('viewerTitle'),
    viewerBody: $('viewerBody'),
    viewerDownloadBtn: $('viewerDownloadBtn'),
    viewerPager: $('viewerPager'),
    viewerPrevBtn: $('viewerPrevBtn'),
    viewerNextBtn: $('viewerNextBtn'),
    viewerPageText: $('viewerPageText'),
    confirmModal: $('confirmModal'),
    confirmIcon: $('confirmIcon'),
    confirmTitle: $('confirmTitle'),
    confirmMessage: $('confirmMessage'),
    confirmCancelBtn: $('confirmCancelBtn'),
    confirmOkBtn: $('confirmOkBtn'),
    dropOverlay: $('dropOverlay'),
    closedScreen: $('closedScreen'),
    toastStack: $('toastStack'),
  };

  const methodCards = Array.from(document.querySelectorAll('.method-card'));
  const methodInputs = Array.from(document.querySelectorAll('input[name="method"]'));

  const ICONS = {
    play: '<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M7.2 5.7v8.6l7.2-4.3-7.2-4.3Z" fill="currentColor"/></svg>',
    close: '<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m5.5 5.5 9 9m0-9-9 9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m6.2 12.4 3.9 3.9 7.7-8.2" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    alert: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.7"/><path d="M12 7.8v5.4M12 16.7v.1" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.7"/><path d="M12 11v5.4M12 7.8v.1" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/></svg>',
  };

  const state = {
    items: [],
    activeId: null,
    activeRegion: -1,
    method: 'smart',
    videoMode: 'quality',
    videoMotion: false,
    uploading: false,
    submitting: false,
    cancelling: false,
    resuming: false,
    detecting: false,
    job: null,
    pollTimer: null,
    pollFailures: 0,
    zoom: 1,
    fitScale: 1,
    drag: null,
    mediaSeq: 0,
    uploadSeq: 0,
    templates: [],
  };

  let lastFocus = null;
  let confirmResolver = null;
  let layoutQueued = false;
  let dragDepth = 0;
  let savedRegions = {};

  function activeItem() {
    return state.items.find((item) => item.id === state.activeId) || null;
  }

  function isJobActive() {
    return !!state.job && (state.job.status === 'queued' || state.job.status === 'running');
  }

  // 停止（cancelled）也是终态：已完成的成品照样能下载，只是这一批不再往下跑
  function isFinalJobStatus(status) {
    return status === 'done' || status === 'failed' || status === 'cancelled';
  }

  function isBusy() {
    return state.uploading || state.submitting || isJobActive();
  }

  async function api(path, options = {}) {
    const method = (options.method || 'GET').toUpperCase();
    const headers = Object.assign({}, options.headers);
    if (method !== 'GET' && method !== 'HEAD') headers['X-App-Token'] = TOKEN;
    let body = options.body;
    if (body !== undefined && body !== null && !(body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(body);
    }
    const response = await fetch(path, { method, headers, body, credentials: 'same-origin' });
    const type = response.headers.get('content-type') || '';
    let data = null;
    if (type.includes('application/json')) {
      try {
        data = await response.json();
      } catch (error) {
        data = null;
      }
    }
    if (!response.ok) {
      let message = (data && data.error) || null;
      if (!message && response.status === 413) message = '上传文件过大，请分批处理';
      if (!message) message = `请求失败（${response.status}）`;
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return data || {};
  }

  function toast(message, type = 'info', duration = 3200) {
    const node = document.createElement('div');
    node.className = `toast is-${type}`;
    const icon = document.createElement('span');
    icon.className = 'toast-icon';
    icon.innerHTML = type === 'success' ? ICONS.check : type === 'error' ? ICONS.alert : ICONS.info;
    const text = document.createElement('span');
    text.className = 'toast-text';
    text.textContent = message;
    node.append(icon, text);
    el.toastStack.append(node);
    while (el.toastStack.children.length > 4) el.toastStack.firstElementChild.remove();
    setTimeout(() => {
      node.classList.add('is-leaving');
      setTimeout(() => node.remove(), 200);
    }, Math.max(1600, duration));
  }

  function formatDuration(seconds) {
    const value = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const secs = String(value % 60).padStart(2, '0');
    return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${secs}` : `${minutes}:${secs}`;
  }

  function formatBytes(bytes) {
    const value = Number(bytes) || 0;
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
    if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
    return `${(value / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  function extensionOf(name) {
    const text = String(name || '');
    const dot = text.lastIndexOf('.');
    return dot > 0 ? text.slice(dot + 1).toLowerCase() : '';
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function roundRegion(region) {
    return {
      x: Number(region.x.toFixed(6)),
      y: Number(region.y.toFixed(6)),
      w: Number(region.w.toFixed(6)),
      h: Number(region.h.toFixed(6)),
    };
  }

  function isValidRegion(region) {
    return (
      region &&
      Number.isFinite(region.x) &&
      Number.isFinite(region.y) &&
      Number.isFinite(region.w) &&
      Number.isFinite(region.h) &&
      region.w > 0.002 &&
      region.h > 0.002
    );
  }

  function sameRegion(a, b) {
    return (
      Math.abs(a.x - b.x) < 0.015 &&
      Math.abs(a.y - b.y) < 0.015 &&
      Math.abs(a.w - b.w) < 0.02 &&
      Math.abs(a.h - b.h) < 0.02
    );
  }

  function hydrateItem(item) {
    return Object.assign({}, item, {
      checked: true,
      regions: (savedRegions[item.id] || []).slice(),
      suggestions: [],
    });
  }

  function saveRegions() {
    try {
      const map = {};
      for (const item of state.items) {
        if (item.regions.length) map[item.id] = item.regions;
      }
      if (Object.keys(map).length) localStorage.setItem(REGIONS_KEY, JSON.stringify(map));
      else localStorage.removeItem(REGIONS_KEY);
    } catch {}
  }

  function loadSavedRegions() {
    try {
      const raw = JSON.parse(localStorage.getItem(REGIONS_KEY) || '{}');
      const valid = {};
      for (const [id, list] of Object.entries(raw && typeof raw === 'object' ? raw : {})) {
        const regions = Array.isArray(list)
          ? list
              .filter(
                (region) =>
                  region &&
                  [region.x, region.y, region.w, region.h].every(
                    (value) => Number.isFinite(value) && value >= 0 && value <= 1
                  )
              )
              .slice(0, MAX_REGIONS)
          : [];
        if (regions.length) valid[id] = regions;
      }
      savedRegions = valid;
    } catch (error) {
      savedRegions = {};
    }
  }

  function syncModalLock() {
    const anyOpen = Array.from(document.querySelectorAll('.modal-backdrop')).some((node) => !node.hidden);
    document.body.classList.toggle('has-modal', anyOpen);
  }

  function openModal(id) {
    const node = $(id);
    if (!node || !node.hidden) return;
    lastFocus = document.activeElement;
    node.hidden = false;
    syncModalLock();
    const focusable = node.querySelector('button:not([disabled]), a[href], input, [tabindex]');
    if (focusable) setTimeout(() => focusable.focus({ preventScroll: true }), 20);
  }

  function closeModal(id) {
    const node = $(id);
    if (!node || node.hidden) return;
    if (id === 'jobModal' && isJobActive()) {
      toast('任务在后台继续，完成后点右上角查看', 'info', 2600);
    }
    if (id === 'viewerModal') clearViewer();
    node.hidden = true;
    syncModalLock();
    if (lastFocus && document.contains(lastFocus) && lastFocus.focus) {
      lastFocus.focus({ preventScroll: true });
    }
    if (id === 'jobModal') renderJobChip(state.job);
  }

  function askConfirm({ title, message, okText, danger } = {}) {
    if (confirmResolver) resolveConfirm(false);
    el.confirmTitle.textContent = title || '确认操作';
    el.confirmMessage.textContent = message || '';
    el.confirmOkBtn.textContent = okText || '确定';
    el.confirmOkBtn.classList.toggle('is-danger', !!danger);
    el.confirmIcon.classList.toggle('is-danger', !!danger);
    el.confirmModal.hidden = false;
    syncModalLock();
    setTimeout(() => el.confirmOkBtn.focus({ preventScroll: true }), 20);
    return new Promise((resolve) => {
      confirmResolver = resolve;
    });
  }

  function resolveConfirm(value) {
    if (!confirmResolver) return;
    const resolver = confirmResolver;
    confirmResolver = null;
    el.confirmModal.hidden = true;
    syncModalLock();
    resolver(value);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      const area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.append(area);
      area.select();
      let ok = false;
      try {
        ok = document.execCommand('copy');
      } catch (innerError) {
        ok = false;
      }
      area.remove();
      return ok;
    }
  }

  function createRegionTag(item) {
    const tag = document.createElement('span');
    tag.className = 'region-tag';
    tag.dataset.role = 'region-tag';
    if (item.regions.length) tag.classList.add('has-regions');
    tag.textContent = item.regions.length ? `已框选 ${item.regions.length}` : '未框选';
    return tag;
  }

  function createFileItem(item) {
    const li = document.createElement('li');
    li.className = 'file-item' + (item.id === state.activeId ? ' is-active' : '');
    li.dataset.id = item.id;
    li.tabIndex = 0;

    const check = document.createElement('label');
    check.className = 'file-check';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = item.checked;
    input.setAttribute('aria-label', `勾选 ${item.name}`);
    const box = document.createElement('span');
    box.className = 'custom-check';
    check.append(input, box);

    const thumb = document.createElement('div');
    thumb.className = 'thumb-wrap';
    const image = document.createElement('img');
    image.className = 'thumb-img';
    image.src = item.thumb_url;
    image.alt = '';
    image.loading = 'lazy';
    image.draggable = false;
    image.addEventListener(
      'error',
      () => {
        const empty = document.createElement('span');
        empty.className = 'thumb-empty';
        empty.textContent = '无法预览';
        if (image.parentNode) image.replaceWith(empty);
      },
      { once: true }
    );
    thumb.append(image);
    if (item.kind === 'video') {
      const chip = document.createElement('span');
      chip.className = 'video-chip';
      chip.innerHTML = ICONS.play;
      const label = document.createElement('span');
      label.textContent = item.duration ? formatDuration(item.duration) : '视频';
      chip.append(label);
      thumb.append(chip);
    }

    const copy = document.createElement('div');
    copy.className = 'file-copy';
    const name = document.createElement('div');
    name.className = 'file-name';
    name.title = item.name;
    name.textContent = item.name;
    const sub = document.createElement('div');
    sub.className = 'file-sub';
    const size = document.createElement('span');
    size.textContent = `${item.width}×${item.height}`;
    sub.append(size, createRegionTag(item));
    copy.append(name, sub);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'file-remove';
    remove.title = '移除此素材';
    remove.setAttribute('aria-label', `移除 ${item.name}`);
    remove.innerHTML = ICONS.close;

    li.append(check, thumb, copy, remove);
    return li;
  }

  function renderFileList() {
    const top = el.fileList.scrollTop;
    el.fileList.replaceChildren(...state.items.map(createFileItem));
    el.fileList.scrollTop = top;
    el.fileListEmpty.hidden = state.items.length > 0;
    el.fileCount.textContent = String(state.items.length);
    updateSelectionUI();
  }

  function updateFileItemRegion(item) {
    if (!item) return;
    const node = Array.from(el.fileList.children).find((li) => li.dataset.id === item.id);
    if (!node) return;
    const tag = node.querySelector('[data-role="region-tag"]');
    if (!tag) return;
    tag.className = 'region-tag' + (item.regions.length ? ' has-regions' : '');
    tag.textContent = item.regions.length ? `已框选 ${item.regions.length}` : '未框选';
  }

  function syncActiveFlags() {
    for (const node of el.fileList.children) {
      node.classList.toggle('is-active', node.dataset.id === state.activeId);
    }
  }

  function updateStageButtons() {
    const item = activeItem();
    el.autoDetectBtn.disabled = !item || state.detecting;
    el.undoRegionBtn.disabled = !item || item.regions.length === 0;
    el.clearRegionsBtn.disabled = !item || item.regions.length === 0;
    el.applyAllBtn.disabled = !item || item.regions.length === 0 || !state.items.some((entry) => entry.checked);
    if (el.saveTemplateBtn) el.saveTemplateBtn.disabled = !item || item.regions.length === 0;
  }

  function renderRegions() {
    const item = activeItem();
    const regions = item ? item.regions : [];
    el.regionCount.textContent = `${regions.length}/${MAX_REGIONS}`;
    el.regionEmpty.hidden = regions.length > 0;
    el.regionList.hidden = regions.length === 0;
    el.regionList.replaceChildren();
    regions.forEach((region, index) => {
      const li = document.createElement('li');
      li.className = 'region-item' + (index === state.activeRegion ? ' is-active' : '');
      li.dataset.index = String(index);

      const number = document.createElement('span');
      number.className = 'region-index';
      number.textContent = String(index + 1);

      const detail = document.createElement('div');
      detail.className = 'region-detail';
      const title = document.createElement('strong');
      title.textContent = `选框 ${index + 1}`;
      detail.append(
        title,
        document.createTextNode(
          `位置 ${Math.round(region.x * 100)}%, ${Math.round(region.y * 100)}% · 大小 ${Math.round(
            region.w * 100
          )}% × ${Math.round(region.h * 100)}%`
        )
      );

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'region-delete';
      remove.title = '删除这个选框';
      remove.setAttribute('aria-label', `删除选框 ${index + 1}`);
      remove.innerHTML = ICONS.close;
      remove.addEventListener('click', (event) => {
        event.stopPropagation();
        removeRegion(index);
      });

      li.append(number, detail, remove);
      li.addEventListener('click', () => {
        state.activeRegion = index;
        renderRegions();
        drawCanvas();
      });
      el.regionList.append(li);
    });

    if (item && state.activeRegion >= regions.length) state.activeRegion = regions.length - 1;
    el.drawHint.hidden = !item || regions.length > 0;
    renderSuggestions();
    renderTemplates();
    updateStageButtons();
  }

  function renderSuggestions() {
    const item = activeItem();
    const suggestions = item && Array.isArray(item.suggestions) ? item.suggestions : [];
    el.suggestionBlock.hidden = suggestions.length === 0;
    el.suggestionCount.textContent = String(suggestions.length);
    el.suggestionList.replaceChildren();
    const full = !!item && item.regions.length >= MAX_REGIONS;
    suggestions.forEach((suggestion, index) => {
      const li = document.createElement('li');
      li.className = 'suggestion-item';

      const badge = document.createElement('span');
      badge.className = 'suggestion-badge';
      badge.textContent = '疑似';

      const detail = document.createElement('div');
      detail.className = 'suggestion-detail';
      const title = document.createElement('strong');
      title.textContent = suggestion.label || '低对比水印';
      detail.append(
        title,
        document.createTextNode(
          `位置 ${Math.round(suggestion.x * 100)}%, ${Math.round(suggestion.y * 100)}% · 大小 ${Math.round(
            suggestion.w * 100
          )}% × ${Math.round(suggestion.h * 100)}%`
        )
      );

      const adopt = document.createElement('button');
      adopt.type = 'button';
      adopt.className = 'suggestion-adopt';
      adopt.textContent = '采纳';
      adopt.disabled = full;
      adopt.title = full ? `选框已到 ${MAX_REGIONS} 个上限` : '把这一处加入选框';
      adopt.addEventListener('click', (event) => {
        event.stopPropagation();
        adoptSuggestion(index);
      });

      const dismiss = document.createElement('button');
      dismiss.type = 'button';
      dismiss.className = 'suggestion-dismiss';
      dismiss.textContent = '忽略';
      dismiss.setAttribute('aria-label', `忽略疑似位置 ${index + 1}`);
      dismiss.addEventListener('click', (event) => {
        event.stopPropagation();
        dismissSuggestion(index);
      });

      li.append(badge, detail, adopt, dismiss);
      el.suggestionList.append(li);
    });
  }

  function adoptSuggestion(index) {
    const item = activeItem();
    const suggestion = item && item.suggestions && item.suggestions[index];
    if (!suggestion) return;
    if (item.regions.length >= MAX_REGIONS) {
      toast(`选框已到 ${MAX_REGIONS} 个上限，删掉几个再采纳`, 'error', 3200);
      return;
    }
    item.regions.push(roundRegion(suggestion));
    item.suggestions.splice(index, 1);
    state.activeRegion = item.regions.length - 1;
    refreshRegionState(item);
    toast('已采纳这处疑似位置，处理前核对一下选框', 'success', 2800);
  }

  function dismissSuggestion(index) {
    const item = activeItem();
    if (!item || !item.suggestions || !item.suggestions[index]) return;
    item.suggestions.splice(index, 1);
    renderSuggestions();
    drawCanvas();
  }

  function clearSuggestions() {
    const item = activeItem();
    if (!item || !item.suggestions || !item.suggestions.length) return;
    item.suggestions = [];
    renderSuggestions();
    drawCanvas();
  }

  function dropStaleSuggestions() {
    const item = activeItem();
    if (!item || item.kind !== 'video' || !item.suggestions || !item.suggestions.length) return;
    item.suggestions = [];
    renderSuggestions();
    drawCanvas();
  }

  function loadTemplates() {
    try {
      const raw = JSON.parse(localStorage.getItem(TEMPLATES_KEY) || '[]');
      state.templates = (Array.isArray(raw) ? raw : [])
        .filter((entry) => entry && Array.isArray(entry.regions))
        .map((entry) => ({
          id: String(entry.id || `t${Date.now()}${Math.random()}`),
          name: String(entry.name || '模板'),
          regions: entry.regions.filter(isValidRegion).slice(0, MAX_REGIONS).map(roundRegion),
        }))
        .filter((entry) => entry.regions.length)
        .slice(0, MAX_TEMPLATES);
    } catch (error) {
      state.templates = [];
    }
  }

  function persistTemplates() {
    try {
      localStorage.setItem(TEMPLATES_KEY, JSON.stringify(state.templates));
    } catch {}
  }

  function nextTemplateName() {
    let max = 0;
    for (const entry of state.templates) {
      const match = /(\d+)\s*$/.exec(entry.name || '');
      if (match) max = Math.max(max, Number(match[1]));
    }
    return `模板 ${max + 1}`;
  }

  function renderTemplates() {
    if (!el.templateBlock) return;
    const item = activeItem();
    const templates = state.templates;
    el.templateBlock.hidden = templates.length === 0 && (!item || item.regions.length === 0);
    el.templateCount.textContent = String(templates.length);
    el.templateList.replaceChildren();
    templates.forEach((template) => {
      const li = document.createElement('li');
      li.className = 'template-item';

      const detail = document.createElement('div');
      detail.className = 'template-detail';
      const title = document.createElement('strong');
      title.textContent = template.name;
      const meta = document.createElement('span');
      meta.textContent = `${template.regions.length} 个选框 · 按画面比例套用`;
      detail.append(title, meta);

      const actions = document.createElement('div');
      actions.className = 'template-actions';

      const applyAll = document.createElement('button');
      applyAll.type = 'button';
      applyAll.className = 'template-apply';
      applyAll.textContent = '套用全批';
      applyAll.addEventListener('click', (event) => {
        event.stopPropagation();
        applyTemplate(template, false);
      });

      const applyChecked = document.createElement('button');
      applyChecked.type = 'button';
      applyChecked.className = 'template-apply is-quiet';
      applyChecked.textContent = '仅勾选';
      applyChecked.addEventListener('click', (event) => {
        event.stopPropagation();
        applyTemplate(template, true);
      });

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'template-delete';
      remove.textContent = '删除';
      remove.addEventListener('click', (event) => {
        event.stopPropagation();
        deleteTemplate(template);
      });

      actions.append(applyAll, applyChecked, remove);
      li.append(detail, actions);
      el.templateList.append(li);
    });
  }

  function saveTemplate() {
    const item = activeItem();
    if (!item || !item.regions.length) {
      toast('先在画面上框出水印，再存成模板', 'error');
      return;
    }
    const template = {
      id: `t${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
      name: nextTemplateName(),
      regions: item.regions.map((region) => roundRegion(region)),
    };
    state.templates.unshift(template);
    if (state.templates.length > MAX_TEMPLATES) state.templates.length = MAX_TEMPLATES;
    persistTemplates();
    renderTemplates();
    toast(`已存为「${template.name}」（${template.regions.length} 个选框），可套用到整批`, 'success', 3200);
  }

  function applyTemplate(template, onlyChecked) {
    if (!template || !template.regions.length) return;
    const targets = state.items.filter((entry) => !onlyChecked || entry.checked);
    if (!targets.length) {
      toast(onlyChecked ? '先勾选要套用的素材' : '还没有素材可套用', 'error');
      return;
    }
    let added = 0;
    let fullSkips = 0;
    let touched = 0;
    targets.forEach((entry) => {
      let local = 0;
      for (const region of template.regions) {
        if (entry.regions.some((existing) => sameRegion(existing, region))) continue;
        if (entry.regions.length >= MAX_REGIONS) {
          fullSkips += 1;
          continue;
        }
        entry.regions.push(Object.assign({}, region));
        local += 1;
      }
      if (local) touched += 1;
      added += local;
    });
    const active = activeItem();
    targets.forEach(updateFileItemRegion);
    if (active) {
      refreshRegionState(active);
    } else {
      updateSelectionUI();
      saveRegions();
      renderTemplates();
    }
    if (!added) {
      toast(
        fullSkips
          ? `套用「${template.name}」时 ${fullSkips} 个选框因素材已达 ${MAX_REGIONS} 个上限未加入`
          : `这些素材上已经有「${template.name}」的选框，没有重复添加`,
        fullSkips ? 'error' : 'info',
        3800
      );
      return;
    }
    toast(
      `已把「${template.name}」套用到 ${touched} 项素材，新增 ${added} 个选框${
        fullSkips ? `；${fullSkips} 个因已满 ${MAX_REGIONS} 个上限未加入` : ''
      }`,
      'success',
      4000
    );
  }

  async function deleteTemplate(template) {
    const ok = await askConfirm({
      title: '删除模板',
      message: `确定删除「${template.name}」？已经套用出去的选框不受影响。`,
      okText: '删除',
      danger: true,
    });
    if (!ok) return;
    state.templates = state.templates.filter((entry) => entry.id !== template.id);
    persistTemplates();
    renderTemplates();
    toast(`已删除「${template.name}」`, 'info', 2200);
  }


  function updateSelectionUI() {
    const checked = state.items.filter((item) => item.checked);
    const ready = checked.filter((item) => item.regions.length);
    const skipped = checked.length - ready.length;

    el.checkedCount.textContent = String(checked.length);
    el.checkAll.checked = state.items.length > 0 && checked.length === state.items.length;
    el.checkAll.indeterminate = checked.length > 0 && checked.length < state.items.length;

    el.batchReady.textContent = String(ready.length);
    el.batchSkipped.textContent = skipped ? `${skipped} 项未框选，将自动跳过` : '';

    const busy = isBusy();
    el.processBtn.disabled = busy || ready.length === 0;
    if (state.submitting || isJobActive()) {
      el.processBtnLabel.textContent = '正在处理…';
      el.processHint.textContent = '任务在后台跑，可以先添加下一批。';
    } else if (state.uploading) {
      el.processBtnLabel.textContent = '正在上传…';
      el.processHint.textContent = '上传完就能开始处理。';
    } else if (ready.length) {
      el.processBtnLabel.textContent = `开始批量处理 · ${ready.length} 项`;
      el.processHint.textContent = '成品在工具目录的「成品」文件夹。';
    } else if (checked.length) {
      el.processBtnLabel.textContent = '开始批量处理';
      el.processHint.textContent = '先在画面上框住水印。';
    } else {
      el.processBtnLabel.textContent = '开始批量处理';
      el.processHint.textContent = state.items.length ? '先勾选要处理的素材。' : '先添加素材。';
    }
    updateStageButtons();
  }

  function stopVideo() {
    const video = el.previewVideo;
    video.onloadeddata = null;
    video.onerror = null;
    if (!video.paused) video.pause();
    if (video.getAttribute('src')) {
      video.removeAttribute('src');
      video.load();
    }
    el.videoControls.hidden = true;
    syncPlayButton();
  }

  function updateVideoTime() {
    const video = el.previewVideo;
    if (video.hidden || el.videoControls.hidden) return;
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    const current = Number.isFinite(video.currentTime) ? video.currentTime : 0;
    el.videoTime.textContent = `${formatDuration(current)} / ${formatDuration(duration)}`;
    if (duration > 0) el.videoSeek.value = String((current / duration) * 100);
  }

  function syncPlayButton() {
    const playing = !el.previewVideo.paused && !el.previewVideo.ended;
    const playIcon = el.videoPlayBtn.querySelector('.play-icon');
    const pauseIcon = el.videoPlayBtn.querySelector('.pause-icon');
    if (playIcon) playIcon.hidden = playing;
    if (pauseIcon) pauseIcon.hidden = !playing;
    el.videoPlayBtn.setAttribute('aria-label', playing ? '暂停' : '播放');
  }

  function togglePlayback() {
    const video = el.previewVideo;
    if (video.hidden) return;
    if (video.paused) {
      video
        .play()
        .then(() => syncPlayButton())
        .catch(() => toast('这个视频暂时无法在浏览器中播放', 'error'));
    } else {
      video.pause();
      syncPlayButton();
    }
  }

  function stepVideo(offset) {
    const video = el.previewVideo;
    if (video.hidden || !Number.isFinite(video.duration)) return;
    video.currentTime = clamp(video.currentTime + offset, 0, Math.max(0, video.duration - 0.04));
    updateVideoTime();
  }

  function clearStageMedia() {
    stopVideo();
    el.previewImage.onloadeddata = null;
    el.previewImage.onload = null;
    el.previewImage.onerror = null;
    el.posterImage.onload = null;
    el.posterImage.onerror = null;
    el.previewImage.hidden = true;
    el.previewImage.removeAttribute('src');
    el.previewVideo.hidden = true;
    el.videoFallback.hidden = true;
    el.posterImage.removeAttribute('src');
    el.videoControls.hidden = true;
    el.videoErrorNote.hidden = true;
    el.drawHint.hidden = true;
  }

  function resetStage() {
    state.mediaSeq += 1;
    state.activeId = null;
    state.activeRegion = -1;
    state.zoom = 1;
    state.drag = null;
    clearStageMedia();
    el.stageEmpty.hidden = false;
    el.stageSpacer.hidden = true;
    el.stageLoading.hidden = true;
    el.zoomTools.hidden = true;
    el.currentKindBadge.hidden = true;
    el.currentName.textContent = '还没有选择文件';
    el.currentName.removeAttribute('title');
    el.currentMeta.textContent = '选中素材后显示尺寸与时长';
    renderRegions();
    drawCanvas();
    updateSelectionUI();
  }

  function updateStageHeader(item) {
    el.currentKindBadge.hidden = false;
    el.currentKindBadge.textContent = item.kind === 'image' ? '图片' : '视频';
    el.currentName.textContent = item.name;
    el.currentName.title = item.name;
    el.currentMeta.textContent = `${item.width}×${item.height}${
      item.duration ? ` · ${formatDuration(item.duration)}` : ''
    }`;
  }

  function showVideoFallback(item) {
    state.mediaSeq += 1;
    stopVideo();
    el.previewVideo.hidden = true;
    el.videoControls.hidden = true;
    el.videoFallback.hidden = false;
    el.posterImage.src = item.thumb_url;
    el.posterImage.onerror = () => {
      if (state.activeId === item.id) toast('无法读取视频封面', 'error');
    };
    el.videoErrorNote.textContent = '这个视频在浏览器里放不了，已改用封面图；框选和处理不受影响。';
    el.videoErrorNote.hidden = false;
    el.stageLoading.hidden = true;
    layoutStage();
  }

  function loadActiveMedia(item) {
    const seq = (state.mediaSeq += 1);
    state.drag = null;
    state.activeRegion = -1;
    state.zoom = 1;
    clearStageMedia();
    updateStageHeader(item);
    el.stageEmpty.hidden = true;
    el.stageSpacer.hidden = false;
    el.stageLoading.hidden = false;
    el.zoomTools.hidden = false;
    layoutStage();

    if (item.kind === 'image') {
      el.previewImage.hidden = false;
      el.previewImage.onload = () => {
        if (seq !== state.mediaSeq || state.activeId !== item.id) return;
        el.stageLoading.hidden = true;
        layoutStage();
      };
      el.previewImage.onerror = () => {
        if (seq !== state.mediaSeq || state.activeId !== item.id) return;
        el.stageLoading.hidden = true;
        toast('无法加载预览图，可以重新添加这个文件', 'error');
      };
      el.previewImage.src = item.display_url;
      return;
    }

    if (!item.playable) {
      showVideoFallback(item);
      return;
    }

    el.previewVideo.hidden = false;
    el.videoControls.hidden = false;
    el.videoErrorNote.hidden = true;
    el.videoTime.textContent = `0:00 / ${formatDuration(item.duration)}`;
    el.videoSeek.value = '0';
    syncPlayButton();
    el.previewVideo.poster = item.thumb_url;
    el.previewVideo.onloadeddata = () => {
      if (seq !== state.mediaSeq || state.activeId !== item.id) return;
      el.stageLoading.hidden = true;
      updateVideoTime();
    };
    el.previewVideo.onerror = () => {
      if (seq !== state.mediaSeq || state.activeId !== item.id) return;
      showVideoFallback(item);
    };
    el.previewVideo.src = item.display_url;
    el.previewVideo.load();
  }

  function selectItem(id, options = {}) {
    const item = state.items.find((entry) => entry.id === id);
    if (!item) return;
    const changed = state.activeId !== id;
    state.activeId = id;
    syncActiveFlags();
    if (changed) {
      state.mediaSeq += 1;
      state.drag = null;
      state.activeRegion = -1;
      loadActiveMedia(item);
    }
    renderRegions();
    drawCanvas();
    updateSelectionUI();
    if (options.scroll !== false) {
      const node = Array.from(el.fileList.children).find((li) => li.dataset.id === id);
      if (node && node.scrollIntoView) node.scrollIntoView({ block: 'nearest' });
    }
  }

  function scheduleLayout() {
    if (layoutQueued) return;
    layoutQueued = true;
    requestAnimationFrame(() => {
      layoutQueued = false;
      layoutStage();
    });
  }

  function layoutStage() {
    const item = activeItem();
    if (!item || el.stageSpacer.hidden) return;
    const areaW = Math.max(80, el.stageArea.clientWidth - 6);
    const areaH = Math.max(60, el.stageArea.clientHeight - 6);
    const style = getComputedStyle(el.stageSpacer);
    const padX = (parseFloat(style.paddingLeft) || 0) + (parseFloat(style.paddingRight) || 0);
    const padY = (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0);
    const availW = Math.max(60, areaW - padX);
    const availH = Math.max(60, areaH - padY);
    const fit = Math.min(availW / item.width, availH / item.height, 1);
    state.fitScale = clamp(fit, 0.02, 1);
    const displayW = Math.max(30, Math.round(item.width * state.fitScale * state.zoom));
    const displayH = Math.max(20, Math.round(item.height * state.fitScale * state.zoom));
    el.mediaFrame.style.width = `${displayW}px`;
    el.mediaFrame.style.height = `${displayH}px`;
    el.stageSpacer.style.width = `${Math.max(areaW, displayW + padX)}px`;
    el.stageSpacer.style.height = `${Math.max(areaH, displayH + padY)}px`;
    el.zoomLabel.textContent = state.zoom === 1 ? '适配' : `${state.zoom}×`;
    const stepIndex = ZOOM_STEPS.indexOf(state.zoom);
    el.zoomOutBtn.disabled = stepIndex <= 0;
    el.zoomInBtn.disabled = stepIndex >= ZOOM_STEPS.length - 1;
    drawCanvas();
  }

  function applyZoom(nextZoom) {
    if (!activeItem() || nextZoom === state.zoom) return;
    const area = el.stageArea;
    const ratioX = area.scrollWidth > 0 ? (area.scrollLeft + area.clientWidth / 2) / area.scrollWidth : 0.5;
    const ratioY = area.scrollHeight > 0 ? (area.scrollTop + area.clientHeight / 2) / area.scrollHeight : 0.5;
    state.zoom = nextZoom;
    layoutStage();
    requestAnimationFrame(() => {
      area.scrollLeft = Math.max(0, ratioX * area.scrollWidth - area.clientWidth / 2);
      area.scrollTop = Math.max(0, ratioY * area.scrollHeight - area.clientHeight / 2);
      drawCanvas();
    });
  }

  function changeZoom(direction) {
    let index = ZOOM_STEPS.indexOf(state.zoom);
    if (index < 0) index = 0;
    const next = ZOOM_STEPS[clamp(index + direction, 0, ZOOM_STEPS.length - 1)];
    applyZoom(next);
  }

  function roundedRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function regionRect(region, width, height) {
    return {
      x: region.x * width,
      y: region.y * height,
      w: region.w * width,
      h: region.h * height,
    };
  }

  function drawRegionBox(ctx, region, width, height, index, active) {
    const rect = regionRect(region, width, height);
    if (rect.w <= 0 || rect.h <= 0) return;
    ctx.save();
    ctx.fillStyle = active ? 'rgba(236,103,72,0.17)' : 'rgba(236,103,72,0.10)';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.lineWidth = active ? 2 : 1.3;
    ctx.strokeStyle = active ? '#ee5c3b' : 'rgba(238,92,59,0.78)';
    ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, Math.max(1, rect.w - 1), Math.max(1, rect.h - 1));

    const labelSize = 15;
    const labelX = rect.x;
    const labelY = rect.y - labelSize - 3 < 0 ? rect.y + 2 : rect.y - labelSize - 3;
    ctx.fillStyle = '#e85b3b';
    roundedRect(ctx, labelX, labelY, labelSize, labelSize, 4);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '600 9px "Microsoft YaHei UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(index + 1), labelX + labelSize / 2, labelY + labelSize / 2 + 0.5);

    if (active) {
      const handles = [
        [rect.x, rect.y],
        [rect.x + rect.w, rect.y],
        [rect.x, rect.y + rect.h],
        [rect.x + rect.w, rect.y + rect.h],
      ];
      ctx.fillStyle = '#fff';
      ctx.strokeStyle = '#e85b3b';
      ctx.lineWidth = 1.5;
      for (const [hx, hy] of handles) {
        ctx.beginPath();
        ctx.rect(hx - 4, hy - 4, 8, 8);
        ctx.fill();
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  function drawCreateBox(ctx, region, width, height) {
    const rect = regionRect(region, width, height);
    if (rect.w < 2 || rect.h < 2) return;
    ctx.save();
    ctx.fillStyle = 'rgba(236,103,72,0.12)';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = '#ee5c3b';
    ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w, rect.h);
    ctx.restore();
  }

  function drawSuggestionBox(ctx, suggestion, width, height) {
    const rect = regionRect(suggestion, width, height);
    if (rect.w < 2 || rect.h < 2) return;
    ctx.save();
    ctx.fillStyle = 'rgba(201,138,30,0.10)';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(201,138,30,0.9)';
    ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, Math.max(1, rect.w - 1), Math.max(1, rect.h - 1));
    ctx.setLineDash([]);

    const label = '疑似';
    ctx.font = '600 10px "Microsoft YaHei UI", sans-serif';
    const labelWidth = ctx.measureText(label).width + 8;
    const labelX = rect.x;
    const labelY = rect.y - 16 < 0 ? rect.y + rect.h + 2 : rect.y - 16;
    ctx.fillStyle = 'rgba(201,138,30,0.92)';
    roundedRect(ctx, labelX, labelY, labelWidth, 15, 3);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, labelX + labelWidth / 2, labelY + 8);
    ctx.restore();
  }

  function drawCanvas() {
    const item = activeItem();
    const canvas = el.overlayCanvas;
    const rect = el.mediaFrame.getBoundingClientRect();
    const cssW = rect.width;
    const cssH = rect.height;
    if (!item || el.stageSpacer.hidden || cssW < 2 || cssH < 2) {
      if (canvas.width > 1 || canvas.height > 1) {
        canvas.width = 1;
        canvas.height = 1;
      }
      return;
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const pxW = Math.max(1, Math.round(cssW * dpr));
    const pxH = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== pxW || canvas.height !== pxH) {
      canvas.width = pxW;
      canvas.height = pxH;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    if (Array.isArray(item.suggestions)) {
      item.suggestions.forEach((suggestion) => drawSuggestionBox(ctx, suggestion, cssW, cssH));
    }

    item.regions.forEach((region, index) => {
      const shown =
        state.drag && state.drag.index === index && state.drag.preview ? state.drag.preview : region;
      const active = index === state.activeRegion;
      drawRegionBox(ctx, shown, cssW, cssH, index, active);
    });

    if (state.drag && state.drag.mode === 'create' && state.drag.preview) {
      drawCreateBox(ctx, state.drag.preview, cssW, cssH);
    }
  }

  function pointOnCanvas(event) {
    const rect = el.overlayCanvas.getBoundingClientRect();
    const px = clamp(event.clientX - rect.left, 0, rect.width);
    const py = clamp(event.clientY - rect.top, 0, rect.height);
    return {
      px,
      py,
      nx: rect.width > 0 ? px / rect.width : 0,
      ny: rect.height > 0 ? py / rect.height : 0,
      rect,
    };
  }

  function handleForPoint(region, point) {
    const rect = regionRect(region, point.rect.width, point.rect.height);
    const corners = [
      ['nw', rect.x, rect.y],
      ['ne', rect.x + rect.w, rect.y],
      ['sw', rect.x, rect.y + rect.h],
      ['se', rect.x + rect.w, rect.y + rect.h],
    ];
    for (const [name, x, y] of corners) {
      if (Math.abs(point.px - x) <= 9 && Math.abs(point.py - y) <= 9) return name;
    }
    return null;
  }

  function regionAtPoint(regions, point) {
    for (let index = regions.length - 1; index >= 0; index -= 1) {
      const rect = regionRect(regions[index], point.rect.width, point.rect.height);
      if (point.px >= rect.x - 3 && point.px <= rect.x + rect.w + 3 && point.py >= rect.y - 3 && point.py <= rect.y + rect.h + 3) {
        return index;
      }
    }
    return -1;
  }

  function updateCanvasCursor(point) {
    const item = activeItem();
    if (!item || state.drag) return;
    const active = item.regions[state.activeRegion];
    const handle = active ? handleForPoint(active, point) : null;
    if (handle) {
      el.overlayCanvas.style.cursor = handle === 'nw' || handle === 'se' ? 'nwse-resize' : 'nesw-resize';
      return;
    }
    el.overlayCanvas.style.cursor = regionAtPoint(item.regions, point) >= 0 ? 'move' : 'crosshair';
  }

  function previewFromMove(event, item) {
    const point = pointOnCanvas(event);
    const drag = state.drag;
    if (!drag) return;
    if (drag.mode === 'create') {
      const x0 = Math.min(drag.start.nx, point.nx);
      const y0 = Math.min(drag.start.ny, point.ny);
      const x1 = Math.max(drag.start.nx, point.nx);
      const y1 = Math.max(drag.start.ny, point.ny);
      drag.preview = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    } else if (drag.mode === 'move') {
      const dx = point.nx - drag.start.nx;
      const dy = point.ny - drag.start.ny;
      const source = drag.origin;
      drag.preview = {
        x: clamp(source.x + dx, 0, 1 - source.w),
        y: clamp(source.y + dy, 0, 1 - source.h),
        w: source.w,
        h: source.h,
      };
    } else if (drag.mode === 'resize') {
      const source = drag.origin;
      const dx = point.nx - drag.start.nx;
      const dy = point.ny - drag.start.ny;
      const minW = Math.max(0.002, 6 / point.rect.width);
      const minH = Math.max(0.002, 6 / point.rect.height);
      let x0 = source.x;
      let y0 = source.y;
      let x1 = source.x + source.w;
      let y1 = source.y + source.h;
      if (drag.handle === 'nw' || drag.handle === 'sw') x0 = clamp(source.x + dx, 0, x1 - minW);
      if (drag.handle === 'ne' || drag.handle === 'se') x1 = clamp(source.x + source.w + dx, x0 + minW, 1);
      if (drag.handle === 'nw' || drag.handle === 'ne') y0 = clamp(source.y + dy, 0, y1 - minH);
      if (drag.handle === 'sw' || drag.handle === 'se') y1 = clamp(source.y + source.h + dy, y0 + minH, 1);
      drag.preview = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    }
    drawCanvas();
  }

  function beginDrag(event) {
    const item = activeItem();
    if (!item || el.stageSpacer.hidden) return;
    if (event.button !== undefined && event.button !== 0) return;
    event.preventDefault();
    const point = pointOnCanvas(event);
    const active = item.regions[state.activeRegion];
    const handle = active ? handleForPoint(active, point) : null;

    if (handle) {
      state.drag = {
        mode: 'resize',
        index: state.activeRegion,
        handle,
        origin: Object.assign({}, active),
        start: point,
        preview: null,
        pointerId: event.pointerId,
      };
    } else {
      const index = regionAtPoint(item.regions, point);
      if (index >= 0) {
        state.activeRegion = index;
        state.drag = {
          mode: 'move',
          index,
          origin: Object.assign({}, item.regions[index]),
          start: point,
          preview: null,
          pointerId: event.pointerId,
        };
      } else {
        if (item.regions.length >= MAX_REGIONS) {
          toast(`一份素材最多 ${MAX_REGIONS} 个选框，已经画满了`, 'error', 3600);
          return;
        }
        state.activeRegion = -1;
        state.drag = {
          mode: 'create',
          index: -1,
          start: point,
          preview: null,
          pointerId: event.pointerId,
        };
      }
      for (const node of el.regionList.children) {
        node.classList.toggle('is-active', Number(node.dataset.index || -1) === state.activeRegion);
      }
    }
    el.drawHint.hidden = true;
    try {
      el.overlayCanvas.setPointerCapture(event.pointerId);
    } catch {}
    drawCanvas();
  }

  function moveDrag(event) {
    if (!state.drag) {
      updateCanvasCursor(pointOnCanvas(event));
      return;
    }
    event.preventDefault();
    const item = activeItem();
    if (!item) return;
    previewFromMove(event, item);
  }

  function finishDrag(event) {
    const drag = state.drag;
    if (!drag) return;
    const item = activeItem();
    state.drag = null;
    try {
      if (el.overlayCanvas.hasPointerCapture(drag.pointerId)) {
        el.overlayCanvas.releasePointerCapture(drag.pointerId);
      }
    } catch {}
    if (!item) {
      drawCanvas();
      return;
    }
    if (event) event.preventDefault();

    if (drag.mode === 'create') {
      const preview = drag.preview;
      if (preview && preview.w * item.width >= 4 && preview.h * item.height >= 4) {
        item.regions.push(roundRegion(preview));
        state.activeRegion = item.regions.length - 1;
      }
    } else if (drag.preview) {
      if (item.regions[drag.index]) {
        item.regions[drag.index] = roundRegion(drag.preview);
        state.activeRegion = drag.index;
      }
    } else {
      state.activeRegion = drag.index;
    }
    refreshRegionState(item);
  }

  function cancelDrag() {
    if (!state.drag) return;
    try {
      if (el.overlayCanvas.hasPointerCapture(state.drag.pointerId)) {
        el.overlayCanvas.releasePointerCapture(state.drag.pointerId);
      }
    } catch {}
    state.drag = null;
    const item = activeItem();
    el.drawHint.hidden = !item || item.regions.length > 0;
    drawCanvas();
  }

  function refreshRegionState(item) {
    if (item && state.activeId === item.id) {
      renderRegions();
      drawCanvas();
    }
    if (item) updateFileItemRegion(item);
    updateSelectionUI();
    saveRegions();
  }

  function removeRegion(index) {
    const item = activeItem();
    if (!item || !item.regions[index]) return;
    item.regions.splice(index, 1);
    if (state.activeRegion >= item.regions.length) state.activeRegion = item.regions.length - 1;
    refreshRegionState(item);
  }

  function undoRegion() {
    const item = activeItem();
    if (!item || !item.regions.length) return;
    item.regions.pop();
    if (state.activeRegion >= item.regions.length) state.activeRegion = item.regions.length - 1;
    refreshRegionState(item);
    toast(`已撤销最后一个选框，还剩 ${item.regions.length} 个`, 'info', 1800);
  }

  function clearRegions() {
    const item = activeItem();
    if (!item || !item.regions.length) return;
    item.regions = [];
    state.activeRegion = -1;
    refreshRegionState(item);
  }

  function applyToChecked() {
    const item = activeItem();
    if (!item || !item.regions.length) return;
    const checked = state.items.filter((entry) => entry.checked);
    if (!checked.length) {
      toast('先在素材列表勾选要套用的文件', 'error');
      return;
    }
    const replaced = checked.filter((entry) => entry.id !== item.id && entry.regions.length).length;
    checked.forEach((entry) => {
      entry.regions = item.regions.map((region) => Object.assign({}, region));
    });
    checked.forEach(updateFileItemRegion);
    updateSelectionUI();
    saveRegions();
    toast(
      `已把 ${item.regions.length} 个选框套用到 ${checked.length} 项素材${replaced ? '（原有选框已替换）' : ''}`,
      'success',
      3400
    );
  }

  async function autoDetect() {
    const item = activeItem();
    if (!item || state.detecting) return;
    state.detecting = true;
    const label = el.autoDetectBtn.querySelector('span');
    if (label) label.textContent = '识别中…';
    updateStageButtons();
    try {
      let time = 0;
      if (item.kind === 'video') {
        if (!el.previewVideo.hidden && Number.isFinite(el.previewVideo.currentTime)) {
          time = el.previewVideo.currentTime;
        } else {
          time = Math.min(5, Math.max(0, (item.duration || 0) * 0.12));
        }
      }
      const data = await api('/api/detect', { method: 'POST', body: { item_id: item.id, time } });
      const target = state.items.find((entry) => entry.id === item.id);
      if (!target) return;
      const found = data && data.found && Array.isArray(data.regions) ? data.regions : [];
      const incoming =
        data && Array.isArray(data.suggestions)
          ? data.suggestions.filter((suggestion) => isValidRegion(suggestion))
          : [];
      target.suggestions = incoming;
      let added = 0;
      let skipped = 0;
      for (const region of found) {
        if (!isValidRegion(region)) continue;
        if (target.regions.some((existing) => sameRegion(existing, region))) continue;
        if (target.regions.length >= MAX_REGIONS) {
          skipped += 1;
          continue;
        }
        target.regions.push(roundRegion(region));
        added += 1;
      }
      if (state.activeId === target.id && added) state.activeRegion = target.regions.length - 1;
      refreshRegionState(target);
      const pending = target.suggestions.length;
      if (added && skipped) {
        toast(
          `选框已到 ${MAX_REGIONS} 个上限，加入了 ${added} 处，另有 ${skipped} 处没能加上，请删掉几个再自动识别`,
          'error',
          4200
        );
      } else if (added) {
        toast(
          pending
            ? `已加入 ${added} 处选框；另有 ${pending} 处低对比疑似位置，请核对虚线框`
            : data.message || '已找到疑似水印，请核对选框位置',
          'success',
          4200
        );
      } else if (pending) {
        toast(
          `没找到明显水印，但有 ${pending} 处低对比疑似位置（虚线框），请核对右侧「疑似水印」`,
          'info',
          4800
        );
      } else if (skipped) {
        toast(`选框已到 ${MAX_REGIONS} 个上限，请删掉几个再自动识别`, 'error', 3800);
      } else {
        toast('自动找到的位置和已有选框相同', 'info');
      }
    } catch (error) {
      toast(error.message || '自动识别失败，请手动框选', 'error', 4200);
    } finally {
      state.detecting = false;
      if (label) label.textContent = '自动找水印';
      updateStageButtons();
    }
  }

  async function loadState() {
    try {
      const data = await api('/api/state');
      state.items = (data.items || []).map(hydrateItem);
      renderFileList();
      if (state.items.length) selectItem(state.items[0].id);
      else resetStage();
      const restored = state.items.filter((item) => item.regions.length);
      if (restored.length) {
        toast(`已恢复上次框选的 ${restored.length} 份素材`, 'info', 3200);
        saveRegions();
      }
      if (data.job && data.job.id) {
        state.job = data.job;
        renderJob(data.job);
        if (data.job.status === 'queued' || data.job.status === 'running') startPolling(data.job.id);
      }
    } catch (error) {
      toast('无法连接本地服务，请重新启动去水印工具', 'error', 6000);
      renderFileList();
      resetStage();
    }
  }

  function uploadFiles(files) {
    state.uploading = true;
    const seq = (state.uploadSeq += 1);
    updateSelectionUI();
    el.uploadStatus.hidden = false;
    el.uploadText.textContent = `正在上传 ${files.length} 份素材…`;
    el.uploadPercent.textContent = '0%';
    el.uploadBar.style.width = '0%';

    const form = new FormData();
    files.forEach((file) => form.append('files', file, file.name));
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');
    xhr.setRequestHeader('X-App-Token', TOKEN);

    xhr.upload.addEventListener('progress', (event) => {
      if (!event.lengthComputable || seq !== state.uploadSeq) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      el.uploadPercent.textContent = `${percent}%`;
      el.uploadBar.style.width = `${percent}%`;
    });
    xhr.upload.addEventListener('load', () => {
      if (seq !== state.uploadSeq) return;
      el.uploadText.textContent = '正在读取素材信息…';
      el.uploadPercent.textContent = '100%';
      el.uploadBar.style.width = '100%';
    });

    const done = () => {
      state.uploading = false;
      updateSelectionUI();
      if (seq === state.uploadSeq) {
        setTimeout(() => {
          if (seq === state.uploadSeq && !state.uploading) el.uploadStatus.hidden = true;
        }, 900);
      }
    };

    xhr.addEventListener('load', () => {
      let data = {};
      try {
        data = JSON.parse(xhr.responseText || '{}');
      } catch (error) {
        data = {};
      }
      const errors = Array.isArray(data.errors) ? data.errors : [];
      if (xhr.status >= 200 && xhr.status < 300) {
        const added = (data.items || []).map(hydrateItem);
        if (added.length) {
          const firstNew = added[0].id;
          state.items.push(...added);
          renderFileList();
          if (!state.activeId) selectItem(firstNew);
          toast(`已添加 ${added.length} 份素材`, 'success', 2600);
        } else {
          toast('没有文件添加成功', 'error');
        }
      } else {
        toast((data && data.error) || `上传失败（${xhr.status}）`, 'error', 5200);
      }
      if (errors.length) {
        const extra = errors.length > 2 ? `；另有 ${errors.length - 2} 个未添加` : '';
        toast(errors.slice(0, 2).join('；') + extra, 'error', 6000);
      }
      done();
    });
    xhr.addEventListener('error', () => {
      toast('上传中断，确认文件还在本机后重试', 'error', 5200);
      done();
    });
    xhr.addEventListener('abort', () => {
      toast('上传已取消', 'info');
      done();
    });
    xhr.send(form);
  }

  async function addFiles(fileListLike) {
    if (state.uploading) {
      toast(UPLOAD_BUSY, 'info');
      return;
    }
    const files = Array.from(fileListLike || []);
    if (!files.length) return;
    const accepted = [];
    let ignored = 0;
    for (const file of files) {
      const ext = extensionOf(file.name);
      if (!file.size || (!IMAGE_EXT.has(ext) && !VIDEO_EXT.has(ext))) {
        ignored += 1;
        continue;
      }
      accepted.push(file);
    }
    if (ignored) toast(`已忽略 ${ignored} 个不支持或空白的文件`, 'info', 3200);
    if (!accepted.length) {
      if (!ignored) toast('没有可添加的文件', 'error');
      return;
    }
    const capacity = MAX_ITEMS - state.items.length;
    if (capacity <= 0) {
      toast(`素材库最多 ${MAX_ITEMS} 份，请先移除一些文件`, 'error');
      return;
    }
    if (accepted.length > capacity) {
      toast(`一次最多 ${MAX_ITEMS} 份素材，多余的已忽略`, 'info', 3600);
      accepted.splice(capacity);
    }
    uploadFiles(accepted);
  }

  function readAllEntries(reader) {
    return new Promise((resolve) => {
      const collected = [];
      const read = () => {
        reader.readEntries(
          (batch) => {
            if (!batch.length) {
              resolve(collected);
            } else {
              collected.push(...batch);
              read();
            }
          },
          () => resolve(collected)
        );
      };
      read();
    });
  }

  async function walkEntry(entry, files) {
    try {
      if (entry.isFile) {
        const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
        if (file) files.push(file);
      } else if (entry.isDirectory) {
        const children = await readAllEntries(entry.createReader());
        for (const child of children) await walkEntry(child, files);
      }
    } catch {}
  }

  async function collectDroppedFiles(dataTransfer) {
    const entries = [];
    if (dataTransfer.items) {
      for (const item of Array.from(dataTransfer.items)) {
        if (item.kind !== 'file') continue;
        const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
        if (entry) entries.push(entry);
      }
    }
    if (!entries.length) return Array.from(dataTransfer.files || []);
    const files = [];
    for (const entry of entries) await walkEntry(entry, files);
    return files.length ? files : Array.from(dataTransfer.files || []);
  }

  async function removeItem(id) {
    if (state.uploading) {
      toast('上传进行中，暂时不能移除素材', 'info');
      return;
    }
    if (isJobActive()) {
      toast('还有文件正在处理，暂时不能移除素材', 'error');
      return;
    }
    const index = state.items.findIndex((item) => item.id === id);
    if (index < 0) return;
    try {
      await api(`/api/items/${id}`, { method: 'DELETE' });
    } catch (error) {
      toast(error.message || '移除失败', 'error');
      return;
    }
    const wasActive = state.activeId === id;
    state.items.splice(index, 1);
    saveRegions();
    if (wasActive) {
      state.mediaSeq += 1;
      state.activeId = null;
      state.activeRegion = -1;
      state.drag = null;
      clearStageMedia();
    }
    renderFileList();
    if (wasActive) {
      const next = state.items[Math.min(index, state.items.length - 1)];
      if (next) selectItem(next.id, { scroll: false });
      else resetStage();
    } else {
      updateSelectionUI();
    }
    toast('已移除 1 份素材', 'info', 2000);
  }

  async function clearAllItems() {
    if (!state.items.length) return;
    if (state.uploading) {
      toast('上传进行中，暂时不能清空素材', 'info');
      return;
    }
    if (isJobActive()) {
      toast('还有文件正在处理，暂时不能清空素材', 'error');
      return;
    }
    const ok = await askConfirm({
      title: '清空全部素材？',
      message: `将移除列表里的 ${state.items.length} 份文件和它们的选框；「成品」文件夹里的结果不动。`,
      okText: '清空素材',
      danger: true,
    });
    if (!ok) return;
    try {
      await api('/api/clear', { method: 'POST' });
    } catch (error) {
      toast(error.message || '清空失败', 'error');
      return;
    }
    state.items = [];
    saveRegions();
    renderFileList();
    resetStage();
    toast('素材库已清空', 'info', 2400);
  }

  function setJobIcon(status) {
    el.jobStatusIcon.classList.remove('is-done', 'is-failed');
    if (status === 'done') {
      el.jobStatusIcon.innerHTML = ICONS.check;
      el.jobStatusIcon.classList.add('is-done');
    } else if (status === 'failed') {
      el.jobStatusIcon.innerHTML = ICONS.alert;
      el.jobStatusIcon.classList.add('is-failed');
    } else if (status === 'cancelled') {
      el.jobStatusIcon.innerHTML = ICONS.info;
    } else {
      el.jobStatusIcon.innerHTML = '<span class="loading-ring"></span>';
    }
  }

  function renderJobChip(job) {
    if (!job) {
      el.jobChip.hidden = true;
      return;
    }
    const active = job.status === 'queued' || job.status === 'running';
    el.jobChip.hidden = false;
    el.jobChip.classList.toggle('is-active', active);
    el.jobChip.classList.toggle('is-done', !active);
    el.jobChipSpinner.hidden = !active;
    el.jobChipText.textContent = active
      ? `处理中 ${job.completed || 0}/${job.total || 0}`
      : (job.results || []).length
        ? '查看处理结果'
        : '查看任务详情';
    el.jobChip.title = active ? '查看处理进度' : '查看上次处理结果';
  }

  function renderTaskList(tasks) {
    const nodes = new Map();
    for (const node of Array.from(el.taskProgressList.children)) {
      if (node.dataset.id) nodes.set(node.dataset.id, node);
    }
    for (const task of tasks || []) {
      let node = nodes.get(task.id);
      if (!node) {
        node = document.createElement('li');
        node.dataset.id = task.id;
        node.className = 'task-row';
        const dot = document.createElement('span');
        dot.className = 'task-status-dot';
        const name = document.createElement('span');
        name.className = 'task-name';
        const percent = document.createElement('span');
        percent.className = 'task-percent';
        node.append(dot, name, percent);
      }
      let statusClass = '';
      let percentText = '等待';
      if (task.status === 'processing') {
        statusClass = ' is-processing';
        percentText = `${Math.max(1, Math.round((task.progress || 0) * 100))}%`;
      } else if (task.status === 'done') {
        statusClass = ' is-done';
        percentText = '完成';
      } else if (task.status === 'error') {
        statusClass = ' is-error';
        percentText = '未完成';
      } else if (task.status === 'cancelled') {
        statusClass = ' is-cancelled';
        percentText = '已停止';
      }
      node.className = `task-row${statusClass}`;
      node.title = task.error || task.name;
      node.querySelector('.task-name').textContent = task.name;
      node.querySelector('.task-percent').textContent = percentText;
      nodes.delete(task.id);
      el.taskProgressList.append(node);
    }
    for (const node of nodes.values()) node.remove();
  }

  function createResultCard(result, index, results) {
    const card = document.createElement('article');
    card.className = 'result-card';

    const media = document.createElement('div');
    media.className = 'result-media';
    media.tabIndex = 0;
    media.setAttribute('role', 'button');
    media.setAttribute('aria-label', `对比查看 ${result.name}`);

    if (result.kind === 'video') {
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.preload = 'metadata';
      video.src = `${result.preview_url}#t=0.1`;
      video.addEventListener(
        'error',
        () => {
          if (!video.parentNode) return;
          const fallback = document.createElement('span');
          fallback.className = 'result-fallback';
          fallback.textContent = '无法预览';
          video.replaceWith(fallback);
        },
        { once: true }
      );
      media.append(video);
      const badge = document.createElement('span');
      badge.className = 'play-badge';
      badge.innerHTML = ICONS.play;
      media.append(badge);
    } else {
      const image = document.createElement('img');
      image.src = result.preview_url;
      image.alt = '';
      image.loading = 'lazy';
      image.addEventListener(
        'error',
        () => {
          if (!image.parentNode) return;
          const fallback = document.createElement('span');
          fallback.className = 'result-fallback';
          fallback.textContent = '无法预览';
          image.replaceWith(fallback);
        },
        { once: true }
      );
      media.append(image);
    }

    const typeBadge = document.createElement('span');
    typeBadge.className = 'result-badge';
    typeBadge.textContent = result.kind === 'video' ? '视频' : '图片';
    media.append(typeBadge);
    const hover = document.createElement('span');
    hover.className = 'result-hover';
    hover.textContent = '点击对比查看';
    media.append(hover);

    const info = document.createElement('div');
    info.className = 'result-info';
    const name = document.createElement('div');
    name.className = 'result-name';
    name.title = result.name;
    name.textContent = result.name;
    const meta = document.createElement('div');
    meta.className = 'result-meta';
    meta.textContent = `${formatBytes(result.size)} · ${result.kind === 'video' ? 'MP4 成品' : '原格式成品'}`;
    info.append(name, meta);
    if (result.note) {
      const note = document.createElement('div');
      note.className = 'result-note';
      note.textContent = result.note;
      info.append(note);
    }

    const actions = document.createElement('div');
    actions.className = 'result-actions';
    const compare = document.createElement('button');
    compare.type = 'button';
    compare.className = 'mini-link';
    compare.textContent = '对比查看';
    compare.addEventListener('click', () => openViewer(result, results));
    const download = document.createElement('a');
    download.className = 'mini-link muted';
    download.href = result.download_url;
    download.textContent = '下载';
    download.setAttribute('download', result.name);
    actions.append(compare, download);
    info.append(actions);

    media.addEventListener('click', () => openViewer(result, results));
    media.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openViewer(result, results);
      }
    });

    card.append(media, info);
    return card;
  }

  function renderResults(job) {
    const results = job.results || [];
    const fingerprint = results.map((result) => result.id).join(',');
    if (el.resultGrid.dataset.fingerprint !== fingerprint) {
      el.resultGrid.dataset.fingerprint = fingerprint;
      el.resultGrid.replaceChildren(...results.map(createResultCard));
    }
    el.resultSummary.textContent = `共生成 ${results.length} 个文件`;
    const stopped = job.cancelled_count || 0;
    const issues = [];
    if (job.failed_count) issues.push(`失败 ${job.failed_count} 项`);
    if (stopped) issues.push(`已停止 ${stopped} 项`);
    el.resultStats.textContent = issues.length
      ? `成功 ${job.success_count} 项 · ${issues.join(' · ')}`
      : `${results.length} 项全部完成`;
    el.outputPathText.textContent = job.output_dir || '—';
    el.outputPathText.title = job.output_dir || '';
    if (results.length && job.zip_ready) {
      el.zipDownloadBtn.hidden = false;
      el.zipDownloadBtn.href = `/api/jobs/${encodeURIComponent(job.id)}/zip`;
    } else {
      el.zipDownloadBtn.hidden = true;
      el.zipDownloadBtn.removeAttribute('href');
    }
    el.openFolderBtn.disabled = !results.length;
    el.copyPathBtn.disabled = !results.length;
    el.resultsSection.hidden = results.length === 0;
  }

  function renderJob(job) {
    if (!job) return;
    const final = isFinalJobStatus(job.status);
    setJobIcon(job.status);
    renderJobChip(job);

    if (job.status === 'queued') el.jobTitle.textContent = '正在准备处理';
    else if (job.status === 'running')
      el.jobTitle.textContent = job.cancel_requested ? '正在停止' : '正在去水印';
    else if (job.status === 'done') el.jobTitle.textContent = '处理完成';
    else if (job.status === 'cancelled') el.jobTitle.textContent = '已停止这一批';
    else el.jobTitle.textContent = '这次没有成功';
    el.jobMessage.textContent = job.message || '正在处理这一批素材。';

    const completed = job.completed || 0;
    const total = job.total || 0;
    const percent = final ? 100 : Math.min(99, Math.round((job.progress || 0) * 100));
    el.jobStats.textContent = `${completed} / ${total} 项`;
    el.jobPercent.textContent = `${percent}%`;
    el.jobBar.style.width = `${percent}%`;
    el.jobProgressSection.hidden = final;
    // 按过一次就不再显示：停止是渐进的，重复点只会让人以为没生效
    el.jobCancelRow.hidden = final || !!job.cancel_requested;
    el.jobCancelBtn.disabled = !!state.cancelling;
    // 只在终态给续跑入口；已完成的成品不重做
    const resumable = !!(final && job.resumable);
    el.jobResumeRow.hidden = !resumable;
    if (resumable) {
      el.jobResumeBtn.textContent = `继续处理剩下的 ${job.remaining_count || 0} 项`;
      el.jobResumeBtn.disabled = !!state.resuming;
    }
    renderTaskList(job.tasks || []);

    // 终态时进度区整块收起，被停止的那几份就没有别的逐份证据了，单独列出来
    const stoppedTasks = (job.tasks || []).filter((task) => task.status === 'cancelled');
    const showStopped = final && stoppedTasks.length > 0;
    el.jobStoppedSection.hidden = !showStopped;
    if (showStopped) {
      el.jobStoppedList.replaceChildren(
        ...stoppedTasks.map((task) => {
          const li = document.createElement('li');
          li.textContent = `${task.name}：${task.started ? '处理到一半被停止' : '还没开始就被停止'}`;
          return li;
        })
      );
    } else {
      el.jobStoppedList.replaceChildren();
    }

    const failedTasks = (job.tasks || []).filter((task) => task.status === 'error');
    el.jobFailedSection.hidden = !(final && failedTasks.length);
    if (final && failedTasks.length) {
      el.jobFailedList.replaceChildren(
        ...failedTasks.map((task) => {
          const li = document.createElement('li');
          li.textContent = `${task.name}：${task.error || '处理失败'}`;
          return li;
        })
      );
    } else {
      el.jobFailedList.replaceChildren();
    }

    if (final) {
      renderResults(job);
    } else {
      el.resultsSection.hidden = true;
    }
    document.title = final ? '净印 · 批量去水印' : `净印 · 处理中 ${completed}/${total}`;
    updateSelectionUI();
  }

  function finishJob(job) {
    stopPolling();
    renderJob(job);
    if (el.jobModal.hidden) {
      const results = (job.results || []).length;
      if (job.status === 'done') {
        toast(`处理完成，共 ${results} 个文件；点右上角查看`, 'success', 5000);
      } else if (job.status === 'cancelled') {
        toast(
          results ? `已停止，已完成的 ${results} 个文件仍可下载` : '已停止，这一批还没有成品',
          'info',
          5000
        );
      } else {
        toast('处理失败，点右上角看错误提示', 'error', 5200);
      }
    }
    updateSelectionUI();
  }

  async function cancelCurrentJob() {
    const job = state.job;
    if (!job || isFinalJobStatus(job.status)) return;
    state.cancelling = true;
    el.jobCancelBtn.disabled = true;
    try {
      const data = await api(`/api/jobs/${job.id}/cancel`, { method: 'POST' });
      state.job = data;
      renderJob(data);
      toast('已请求停止：正在处理的那一份会停下，已完成的成品保留', 'info', 4200);
    } catch (error) {
      state.cancelling = false;
      el.jobCancelBtn.disabled = false;
      toast(error.message || '停止失败，请重试', 'error', 4200);
    }
  }

  async function resumeCurrentJob() {
    const job = state.job;
    if (!job || !job.resumable || state.resuming) return;
    state.resuming = true;
    el.jobResumeBtn.disabled = true;
    try {
      const data = await api(`/api/jobs/${job.id}/resume`, { method: 'POST' });
      state.job = data;
      state.resuming = false;
      state.cancelling = false;
      el.taskProgressList.replaceChildren();
      el.jobFailedSection.hidden = true;
      el.jobStoppedSection.hidden = true;
      el.jobProgressSection.hidden = false;
      renderJob(data);
      startPolling(data.id);
      toast(`继续处理剩下的 ${data.remaining_count || 0} 项，已完成的成品不会重跑`, 'info', 4200);
    } catch (error) {
      state.resuming = false;
      el.jobResumeBtn.disabled = false;
      toast(error.message || '继续处理失败，请重试', 'error', 4200);
    }
  }

  function stopPolling() {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  function startPolling(jobId) {
    stopPolling();
    state.pollFailures = 0;
    const tick = async () => {
      if (!state.job || state.job.id !== jobId) return;
      try {
        const data = await api(`/api/jobs/${jobId}`);
        if (!state.job || state.job.id !== jobId) return;
        state.job = data;
        state.pollFailures = 0;
        renderJob(data);
        if (isFinalJobStatus(data.status)) {
          state.cancelling = false;
          state.resuming = false;
          finishJob(data);
          return;
        }
      } catch (error) {
        state.pollFailures += 1;
        if (state.pollFailures >= 5) {
          state.job = Object.assign({}, state.job, {
            status: 'failed',
            message: '与本地服务的连接断了，进度停更；已完成的文件仍在成品文件夹里。',
          });
          renderJob(state.job);
          toast('无法获取任务进度，请检查工具是否仍在运行', 'error', 6000);
          return;
        }
      }
      state.pollTimer = setTimeout(tick, 800);
    };
    state.pollTimer = setTimeout(tick, 400);
  }

  async function startProcessing() {
    if (state.uploading) {
      toast(UPLOAD_BUSY, 'info');
      return;
    }
    if (isJobActive() || state.submitting) return;
    const specs = state.items
      .filter((item) => item.checked && item.regions.length)
      .map((item) => ({ id: item.id, regions: item.regions.map((region) => Object.assign({}, region)) }));
    if (!specs.length) {
      toast('先勾选素材，并在画面上框住水印', 'error');
      return;
    }
    state.submitting = true;
    updateSelectionUI();
    el.processBtnLabel.textContent = '正在创建任务…';
    try {
      const job = await api('/api/process', {
        method: 'POST',
        body: { method: state.method, video_mode: state.videoMode, motion: state.videoMotion, items: specs },
      });
      state.submitting = false;
      state.cancelling = false;
      state.resuming = false;
      state.job = job;
      el.resultGrid.dataset.fingerprint = '';
      el.taskProgressList.replaceChildren();
      el.resultsSection.hidden = true;
      el.jobFailedSection.hidden = true;
      el.jobStoppedSection.hidden = true;
      el.jobProgressSection.hidden = false;
      renderJob(job);
      openModal('jobModal');
      const card = el.jobModal.querySelector('.modal-card');
      if (card) card.scrollTop = 0;
      startPolling(job.id);
      if (!el.previewVideo.hidden) el.previewVideo.pause();
    } catch (error) {
      state.submitting = false;
      updateSelectionUI();
      toast(error.message || '无法开始处理', 'error', 5000);
    }
  }

  function clearViewer() {
    for (const video of Array.from(el.viewerBody.querySelectorAll('video'))) {
      video.pause();
      video.removeAttribute('src');
      video.load();
    }
    el.viewerBody.replaceChildren();
    el.viewerDownloadBtn.removeAttribute('href');
    el.viewerDownloadBtn.removeAttribute('download');
  }

  function viewerFallback(text) {
    const node = document.createElement('div');
    node.className = 'viewer-fallback';
    node.textContent = text;
    return node;
  }

  function buildViewerPane(label, result, before) {
    const pane = document.createElement('div');
    pane.className = 'viewer-pane';
    const head = document.createElement('div');
    head.className = 'viewer-label';
    const strong = document.createElement('strong');
    strong.textContent = label;
    const tag = document.createElement('span');
    tag.textContent = before ? '原素材' : '已去水印的成品';
    head.append(strong, tag);
    pane.append(head);

    const url = before ? result.before_url : result.preview_url;
    if (!url) {
      pane.append(viewerFallback('无法读取这个版本的预览'));
      return pane;
    }
    if (result.kind === 'video') {
      const video = document.createElement('video');
      video.controls = true;
      video.playsInline = true;
      video.preload = 'metadata';
      video.src = url;
      video.addEventListener(
        'error',
        () => {
          if (!video.parentNode) return;
          video.replaceWith(
            viewerFallback('这种格式浏览器放不了，下载后用播放器看。')
          );
        },
        { once: true }
      );
      pane.append(video);
    } else {
      const image = document.createElement('img');
      image.src = url;
      image.alt = label;
      image.addEventListener(
        'error',
        () => {
          if (!image.parentNode) return;
          image.replaceWith(viewerFallback('无法加载这一版预览。'));
        },
        { once: true }
      );
      pane.append(image);
    }
    return pane;
  }

  let viewerItems = [];
  let viewerIndex = 0;

  function showViewerPage(index) {
    if (!viewerItems.length) return;
    viewerIndex = Math.min(Math.max(index, 0), viewerItems.length - 1);
    const result = viewerItems[viewerIndex];
    clearViewer();
    el.viewerTitle.textContent = result.name;
    el.viewerTitle.title = result.name;
    el.viewerBody.append(buildViewerPane('处理前', result, true), buildViewerPane('处理后', result, false));
    el.viewerDownloadBtn.href = result.download_url;
    el.viewerDownloadBtn.setAttribute('download', result.name);
    el.viewerPageText.textContent = `${viewerIndex + 1} / ${viewerItems.length}`;
    el.viewerPager.hidden = viewerItems.length < 2;
    el.viewerPrevBtn.disabled = viewerIndex <= 0;
    el.viewerNextBtn.disabled = viewerIndex >= viewerItems.length - 1;
  }

  function openViewer(result, results) {
    const list = Array.isArray(results) && results.length ? results : result ? [result] : [];
    if (!list.length) return;
    viewerItems = list;
    const at = list.indexOf(result);
    showViewerPage(at < 0 ? 0 : at);
    openModal('viewerModal');
  }

  function setMethod(method, notify) {
    let next = method === 'inpaint' ? 'inpaint' : 'smart';
    if (method === 'photo') next = PHOTO_READY ? 'photo' : 'smart';
    state.method = next;
    methodInputs.forEach((input) => {
      input.checked = input.value === state.method;
    });
    methodCards.forEach((card) => {
      card.classList.toggle('is-selected', card.dataset.method === state.method);
    });
    try {
      localStorage.setItem('jingyin.method', state.method);
    } catch {}
    if (notify) {
      const labels = {
        smart: '已选择智能纹理修复',
        inpaint: '已选择内容填充',
        photo: '已选择照片重绘：首次使用要等本地模型加载（约 20–30 秒），已在后台准备',
      };
      toast(labels[state.method], 'info', state.method === 'photo' ? 4200 : 1800);
    }
    if (state.method === 'photo') {
      api('/api/photo/warmup', { method: 'POST' }).catch(() => {});
    }
  }

  function setVideoMode(fast, notify) {
    state.videoMode = fast ? 'fast' : 'quality';
    if (el.videoFastMode) {
      el.videoFastMode.checked = fast;
      const row = el.videoFastMode.closest('.video-switch');
      if (row) row.classList.toggle('is-on', fast);
    }
    try {
      localStorage.setItem('jingyin.videoMode', state.videoMode);
    } catch {}
    if (notify) {
      toast(
        fast
          ? '速度优先：贴边台标走插值，快约 3–4 倍'
          : '画质优先：贴边台标走逐帧修复',
        'info',
        2400
      );
    }
  }

  function setVideoMotion(motion, notify) {
    state.videoMotion = !!motion;
    if (el.videoMotion) {
      el.videoMotion.checked = !!motion;
      const row = el.videoMotion.closest('.video-switch');
      if (row) row.classList.toggle('is-on', !!motion);
    }
    if (el.videoFastMode) {
      el.videoFastMode.disabled = !!motion;
      const row = el.videoFastMode.closest('.video-switch');
      if (row) row.classList.toggle('is-muted', !!motion);
    }
    try {
      localStorage.setItem('jingyin.videoMotion', motion ? '1' : '0');
    } catch {}
    if (notify) {
      toast(
        motion
          ? '动态水印已打开：每帧重新定位选框，会慢一些'
          : '动态水印已关闭：按固定选框修复',
        'info',
        2600
      );
    }
  }

  async function openOutputFolder() {
    if (!state.job) return;
    el.openFolderBtn.disabled = true;
    try {
      await api(`/api/jobs/${state.job.id}/open-folder`, { method: 'POST' });
      toast('已在资源管理器中打开成品文件夹', 'success', 2600);
    } catch (error) {
      toast(error.message || '无法打开文件夹', 'error', 5000);
    } finally {
      if ((state.job.results || []).length) el.openFolderBtn.disabled = false;
    }
  }

  async function copyOutputPath() {
    const path = state.job && state.job.output_dir;
    if (!path) return;
    const ok = await copyText(path);
    toast(ok ? '成品路径已复制' : '复制失败，请手动选择路径', ok ? 'success' : 'error', 2600);
  }

  async function shutdownTool() {
    if (isJobActive()) {
      toast('还有文件正在处理，请等任务完成后再关闭', 'error');
      return;
    }
    const ok = await askConfirm({
      title: '关闭去水印工具？',
      message: '本地服务会停，素材列表和任务进度清空；已生成的成品仍留在「成品」文件夹。',
      okText: '关闭工具',
      danger: true,
    });
    if (!ok) return;
    try {
      await api('/api/shutdown', { method: 'POST' });
    } catch (error) {
      toast(error.message || '无法关闭工具', 'error');
      return;
    }
    stopPolling();
    state.job = null;
    document.querySelectorAll('.modal-backdrop').forEach((node) => {
      node.hidden = true;
    });
    syncModalLock();
    el.jobChip.hidden = true;
    el.closedScreen.hidden = false;
    if (el.appShell) el.appShell.hidden = true;
    document.title = '净印 · 已关闭';
  }

  function isTypingTarget(target) {
    if (!target || !target.tagName) return false;
    const tag = target.tagName.toUpperCase();
    return ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(tag) || target.isContentEditable;
  }

  function anyModalOpen() {
    return Array.from(document.querySelectorAll('.modal-backdrop')).some((node) => !node.hidden);
  }

  function openFilePicker() {
    if (state.uploading) {
      toast(UPLOAD_BUSY, 'info');
      return;
    }
    el.fileInput.click();
  }

  function bindEvents() {
    el.dropZone.addEventListener('click', openFilePicker);
    el.dropZone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openFilePicker();
      }
    });
    el.pickFilesBtn.addEventListener('click', openFilePicker);
    el.pickFolderBtn.addEventListener('click', () => {
      if (state.uploading) {
        toast(UPLOAD_BUSY, 'info');
        return;
      }
      el.folderInput.click();
    });
    el.emptyAddBtn.addEventListener('click', openFilePicker);
    el.fileInput.addEventListener('change', () => {
      addFiles(el.fileInput.files);
      el.fileInput.value = '';
    });
    el.folderInput.addEventListener('change', () => {
      addFiles(el.folderInput.files);
      el.folderInput.value = '';
    });

    el.checkAll.addEventListener('change', () => {
      const checked = el.checkAll.checked;
      state.items.forEach((item) => {
        item.checked = checked;
      });
      for (const input of el.fileList.querySelectorAll('.file-check input')) input.checked = checked;
      updateSelectionUI();
    });
    el.clearFilesBtn.addEventListener('click', clearAllItems);

    el.fileList.addEventListener('click', (event) => {
      const node = event.target.closest('.file-item');
      if (!node) return;
      if (event.target.closest('.file-remove')) {
        removeItem(node.dataset.id);
        return;
      }
      if (event.target.closest('.file-check')) return;
      selectItem(node.dataset.id, { scroll: false });
    });
    el.fileList.addEventListener('change', (event) => {
      const input = event.target.closest('.file-check input');
      if (!input) return;
      const node = input.closest('.file-item');
      const item = state.items.find((entry) => entry.id === node.dataset.id);
      if (!item) return;
      item.checked = input.checked;
      updateSelectionUI();
    });
    el.fileList.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      if (event.target.closest('.file-remove, .file-check')) return;
      const node = event.target.closest('.file-item');
      if (!node) return;
      event.preventDefault();
      selectItem(node.dataset.id);
    });

    el.autoDetectBtn.addEventListener('click', autoDetect);
    el.undoRegionBtn.addEventListener('click', undoRegion);
    el.clearRegionsBtn.addEventListener('click', clearRegions);
    el.suggestionClearBtn.addEventListener('click', clearSuggestions);
    el.saveTemplateBtn.addEventListener('click', saveTemplate);
    el.applyAllBtn.addEventListener('click', applyToChecked);
    el.processBtn.addEventListener('click', startProcessing);

    methodInputs.forEach((input) => {
      input.addEventListener('change', () => {
        if (input.checked) setMethod(input.value, true);
      });
    });

    if (el.videoFastMode) {
      el.videoFastMode.addEventListener('change', () => {
        setVideoMode(el.videoFastMode.checked, true);
      });
    }

    if (el.videoMotion) {
      el.videoMotion.addEventListener('change', () => {
        setVideoMotion(el.videoMotion.checked, true);
      });
    }

    el.helpBtn.addEventListener('click', () => openModal('helpModal'));
    el.methodHelpBtn.addEventListener('click', () => openModal('methodModal'));
    el.shutdownBtn.addEventListener('click', shutdownTool);
    el.jobChip.addEventListener('click', () => {
      if (!state.job) return;
      if (el.jobModal.hidden) {
        renderJob(state.job);
        openModal('jobModal');
      }
    });
    el.jobCloseBtn.addEventListener('click', () => closeModal('jobModal'));
    el.jobCancelBtn.addEventListener('click', cancelCurrentJob);
    el.jobResumeBtn.addEventListener('click', resumeCurrentJob);
    el.closeResultsBtn.addEventListener('click', () => closeModal('jobModal'));
    el.viewerPrevBtn.addEventListener('click', () => showViewerPage(viewerIndex - 1));
    el.viewerNextBtn.addEventListener('click', () => showViewerPage(viewerIndex + 1));
    el.openFolderBtn.addEventListener('click', openOutputFolder);
    el.copyPathBtn.addEventListener('click', copyOutputPath);
    el.confirmOkBtn.addEventListener('click', () => resolveConfirm(true));
    el.confirmCancelBtn.addEventListener('click', () => resolveConfirm(false));

    for (const button of document.querySelectorAll('[data-close]')) {
      button.addEventListener('click', () => closeModal(button.dataset.close));
    }
    for (const backdrop of document.querySelectorAll('.modal-backdrop')) {
      backdrop.addEventListener('mousedown', (event) => {
        if (event.target !== backdrop) return;
        if (backdrop.id === 'confirmModal') resolveConfirm(false);
        else if (backdrop.id !== 'jobModal') closeModal(backdrop.id);
      });
    }

    el.zoomInBtn.addEventListener('click', () => changeZoom(1));
    el.zoomOutBtn.addEventListener('click', () => changeZoom(-1));
    el.zoomFitBtn.addEventListener('click', () => applyZoom(1));

    el.videoPlayBtn.addEventListener('click', togglePlayback);
    el.videoBackBtn.addEventListener('click', () => stepVideo(-0.1));
    el.videoForwardBtn.addEventListener('click', () => stepVideo(0.1));
    el.videoSeek.addEventListener('input', () => {
      const video = el.previewVideo;
      if (video.hidden || !Number.isFinite(video.duration) || video.duration <= 0) return;
      video.currentTime = (Number(el.videoSeek.value) / 100) * video.duration;
      updateVideoTime();
    });
    el.videoSeek.addEventListener('change', dropStaleSuggestions);
    el.previewVideo.addEventListener('seeked', dropStaleSuggestions);
    el.previewVideo.addEventListener('play', dropStaleSuggestions);
    el.previewVideo.addEventListener('timeupdate', updateVideoTime);
    el.previewVideo.addEventListener('loadedmetadata', updateVideoTime);
    el.previewVideo.addEventListener('play', syncPlayButton);
    el.previewVideo.addEventListener('pause', syncPlayButton);
    el.previewVideo.addEventListener('ended', syncPlayButton);

    const canvas = el.overlayCanvas;
    canvas.addEventListener('pointerdown', beginDrag);
    canvas.addEventListener('pointermove', moveDrag);
    canvas.addEventListener('pointerup', finishDrag);
    canvas.addEventListener('pointercancel', cancelDrag);
    canvas.addEventListener('pointerleave', () => {
      if (!state.drag) canvas.style.cursor = 'crosshair';
    });

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        if (!el.confirmModal.hidden) {
          resolveConfirm(false);
          return;
        }
        if (state.drag) {
          cancelDrag();
          return;
        }
        const open = ['viewerModal', 'helpModal', 'methodModal', 'jobModal'].find(
          (id) => !$(id).hidden
        );
        if (open) closeModal(open);
        return;
      }
      if (
        (event.key === 'ArrowLeft' || event.key === 'ArrowRight') &&
        !el.viewerModal.hidden &&
        viewerItems.length > 1
      ) {
        event.preventDefault();
        showViewerPage(viewerIndex + (event.key === 'ArrowRight' ? 1 : -1));
        return;
      }
      if (event.code === 'Space' && !event.repeat) {
        if (anyModalOpen() || isTypingTarget(event.target)) return;
        const item = activeItem();
        if (item && item.kind === 'video' && !el.previewVideo.hidden && !el.videoControls.hidden) {
          event.preventDefault();
          togglePlayback();
        }
      }
    });

    window.addEventListener('dragenter', (event) => {
      if (!event.dataTransfer || !Array.from(event.dataTransfer.types).includes('Files')) return;
      event.preventDefault();
      dragDepth += 1;
      el.dropOverlay.hidden = false;
    });
    window.addEventListener('dragover', (event) => {
      if (!event.dataTransfer || !Array.from(event.dataTransfer.types).includes('Files')) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'copy';
    });
    window.addEventListener('dragleave', (event) => {
      if (!event.dataTransfer || !Array.from(event.dataTransfer.types).includes('Files')) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) el.dropOverlay.hidden = true;
    });
    window.addEventListener('drop', async (event) => {
      if (!event.dataTransfer || !Array.from(event.dataTransfer.types).includes('Files')) return;
      event.preventDefault();
      dragDepth = 0;
      el.dropOverlay.hidden = true;
      const files = await collectDroppedFiles(event.dataTransfer);
      addFiles(files);
    });
    window.addEventListener('blur', () => {
      dragDepth = 0;
      el.dropOverlay.hidden = true;
    });
    window.addEventListener('resize', scheduleLayout);
    if (window.ResizeObserver) {
      new ResizeObserver(scheduleLayout).observe(el.stageArea);
    }
  }

  function init() {
    bindEvents();
    if (el.photoMethodCard && !PHOTO_READY) {
      const input = el.photoMethodCard.querySelector('input');
      if (input) input.disabled = true;
      el.photoMethodCard.classList.add('is-disabled');
      el.photoMethodCard.title = '本地照片修复模型未就绪（缺少 onnxruntime 或 models/lama_fp32.onnx）';
    }
    let stored = null;
    try {
      stored = localStorage.getItem('jingyin.method');
    } catch (error) {
      stored = null;
    }
    if (stored === 'photo' && PHOTO_READY) setMethod('photo', false);
    else setMethod(stored === 'inpaint' ? 'inpaint' : 'smart', false);
    let storedVideo = null;
    try {
      storedVideo = localStorage.getItem('jingyin.videoMode');
    } catch (error) {
      storedVideo = null;
    }
    setVideoMode(storedVideo === 'fast', false);
    let storedMotion = null;
    try {
      storedMotion = localStorage.getItem('jingyin.videoMotion');
    } catch (error) {
      storedMotion = null;
    }
    setVideoMotion(storedMotion === '1', false);
    loadSavedRegions();
    loadTemplates();
    renderRegions();
    updateSelectionUI();
    loadState();
  }

  init();
})();
