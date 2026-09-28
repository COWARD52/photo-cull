const state = {
  photos: [],
  filter: 'all',
  strictness: 1,
  viewerIndex: -1,
};

const $ = (id) => document.getElementById(id);
const fileInput = $('fileInput');
const gallery = $('gallery');
const canvas = $('analysisCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

['pickBtn', 'dropzone', 'addMoreBtn'].forEach((id) => $(id).addEventListener('click', () => fileInput.click()));
fileInput.addEventListener('change', (event) => processFiles([...event.target.files]));

const dropzone = $('dropzone');
['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.remove('dragover');
}));
dropzone.addEventListener('drop', (event) => processFiles([...event.dataTransfer.files]));

document.querySelectorAll('.filter').forEach((button) => button.addEventListener('click', () => {
  document.querySelectorAll('.filter').forEach((item) => item.classList.remove('active'));
  button.classList.add('active');
  state.filter = button.dataset.filter;
  render();
}));

$('strictness').addEventListener('input', (event) => {
  state.strictness = Number(event.target.value);
  classifyAll();
  render();
});

$('exportBtn').addEventListener('click', exportManifest);
$('viewerClose').addEventListener('click', () => $('viewer').close());
$('viewerKeep').addEventListener('click', () => decideCurrent('keep'));
$('viewerReject').addEventListener('click', () => decideCurrent('reject'));

document.addEventListener('keydown', (event) => {
  if (!$('viewer').open) return;
  if (event.key.toLowerCase() === 'k') decideCurrent('keep');
  if (event.key.toLowerCase() === 'd') decideCurrent('reject');
  if (event.key === 'ArrowRight') moveViewer(1);
  if (event.key === 'ArrowLeft') moveViewer(-1);
});

async function processFiles(files) {
  const valid = files.filter((file) => /^image\/(jpeg|png|webp)$/.test(file.type));
  if (!valid.length) {
    alert('请选择 JPG、PNG 或 WebP 图片。HEIC 可先在系统相册中导出为 JPG。');
    return;
  }

  $('emptyView').hidden = true;
  $('workspace').hidden = false;
  $('summary').hidden = true;
  $('progressWrap').hidden = false;
  $('viewTitle').textContent = `正在分析 ${valid.length} 张照片…`;

  const existingKeys = new Set(state.photos.map((photo) => `${photo.file.name}-${photo.file.size}-${photo.file.lastModified}`));
  const incoming = valid.filter((file) => !existingKeys.has(`${file.name}-${file.size}-${file.lastModified}`));

  for (let i = 0; i < incoming.length; i += 1) {
    updateProgress(i, incoming.length, `正在看第 ${i + 1} 张：${incoming[i].name}`);
    try {
      const photo = await analyzePhoto(incoming[i]);
      state.photos.push(photo);
    } catch (error) {
      console.warn('无法分析图片', incoming[i].name, error);
    }
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  groupSimilarPhotos();
  classifyAll();
  updateProgress(incoming.length, incoming.length, '分析完成');
  setTimeout(() => { $('progressWrap').hidden = true; $('summary').hidden = false; }, 450);
  render();
}

async function analyzePhoto(file) {
  const url = URL.createObjectURL(file);
  const bitmap = await createImageBitmap(file);
  const maxSize = 420;
  const scale = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(24, Math.round(bitmap.width * scale));
  const height = Math.max(24, Math.round(bitmap.height * scale));
  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const image = ctx.getImageData(0, 0, width, height);
  const metrics = analyzePixels(image.data, width, height);
  return {
    id: crypto.randomUUID(), file, url, width, height,
    ...metrics,
    groupId: null, groupRank: 1, duplicate: false,
    decision: null, category: 'review', score: 0, reason: '待分析',
  };
}

function analyzePixels(data, width, height) {
  const gray = new Float32Array(width * height);
  let sum = 0;
  let dark = 0;
  let bright = 0;

  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const value = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    gray[p] = value;
    sum += value;
    if (value < 18) dark += 1;
    if (value > 238) bright += 1;
  }

  let lapSum = 0;
  let lapSq = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const p = y * width + x;
      const lap = 4 * gray[p] - gray[p - 1] - gray[p + 1] - gray[p - width] - gray[p + width];
      lapSum += lap;
      lapSq += lap * lap;
      count += 1;
    }
  }
  const sharpness = Math.max(0, lapSq / count - (lapSum / count) ** 2);
  const hash = makeHash(gray, width, height);
  return {
    brightness: sum / gray.length,
    darkRatio: dark / gray.length,
    brightRatio: bright / gray.length,
    sharpness,
    hash,
  };
}

function makeHash(gray, width, height) {
  const cells = [];
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 9; x += 1) {
      const px = Math.min(width - 1, Math.floor((x + .5) * width / 9));
      const py = Math.min(height - 1, Math.floor((y + .5) * height / 8));
      cells.push(gray[py * width + px]);
    }
  }
  let bits = '';
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      bits += cells[y * 9 + x] > cells[y * 9 + x + 1] ? '1' : '0';
    }
  }
  return bits;
}

function hamming(a, b) {
  let distance = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) distance += 1;
  return distance;
}

function groupSimilarPhotos() {
  state.photos.forEach((photo) => { photo.groupId = null; photo.groupRank = 1; photo.duplicate = false; });
  let groupId = 0;
  for (let i = 0; i < state.photos.length; i += 1) {
    const base = state.photos[i];
    if (base.groupId !== null) continue;
    const group = [base];
    for (let j = i + 1; j < state.photos.length; j += 1) {
      const candidate = state.photos[j];
      const timeGap = Math.abs(candidate.file.lastModified - base.file.lastModified);
      if (hamming(base.hash, candidate.hash) <= 10 && timeGap < 120000) group.push(candidate);
    }
    if (group.length > 1) {
      groupId += 1;
      group.sort((a, b) => rawQuality(b) - rawQuality(a));
      group.forEach((photo, index) => {
        photo.groupId = groupId;
        photo.groupRank = index + 1;
        photo.duplicate = index > 0;
      });
    }
  }
}

function rawQuality(photo) {
  const sharp = Math.min(100, Math.log10(photo.sharpness + 1) * 28);
  const exposure = Math.max(0, 100 - Math.abs(photo.brightness - 128) * .8 - (photo.darkRatio + photo.brightRatio) * 75);
  return sharp * .68 + exposure * .32;
}

function classifyAll() {
  const settings = [
    { blur: 38, exposure: 50, duplicatePenalty: 18 },
    { blur: 45, exposure: 57, duplicatePenalty: 25 },
    { blur: 53, exposure: 64, duplicatePenalty: 32 },
  ][state.strictness];

  state.photos.forEach((photo) => {
    const sharp = Math.min(100, Math.log10(photo.sharpness + 1) * 28);
    const exposure = Math.max(0, 100 - Math.abs(photo.brightness - 128) * .8 - (photo.darkRatio + photo.brightRatio) * 75);
    let score = Math.round(sharp * .66 + exposure * .34 - (photo.duplicate ? settings.duplicatePenalty : 0));
    score = Math.max(1, Math.min(99, score));
    photo.score = score;

    if (sharp < settings.blur) {
      photo.category = 'reject'; photo.reason = '清晰度偏低';
    } else if (photo.brightness < settings.exposure || photo.darkRatio > .58) {
      photo.category = 'reject'; photo.reason = '画面偏暗';
    } else if (photo.brightness > 205 || photo.brightRatio > .45) {
      photo.category = 'reject'; photo.reason = '高光过曝';
    } else if (photo.duplicate && photo.groupRank > 1) {
      photo.category = photo.groupRank > 2 ? 'reject' : 'review'; photo.reason = `相似组第 ${photo.groupRank} 名`;
    } else if (score >= 68) {
      photo.category = 'best'; photo.reason = photo.groupId ? '同组最优' : '清晰、曝光良好';
    } else {
      photo.category = 'review'; photo.reason = '质量接近阈值';
    }
  });
}

function currentPhotos() {
  if (state.filter === 'all') return [...state.photos].sort((a, b) => b.score - a.score);
  if (state.filter === 'kept') return state.photos.filter((photo) => photo.decision === 'keep');
  return state.photos.filter((photo) => photo.category === state.filter);
}

function render() {
  updateCounts();
  const photos = currentPhotos();
  const labels = {
    all: ['全部照片', `${state.photos.length} 张照片，已按综合质量排序`],
    best: ['推荐保留', '优先看这些，通常是每组里更清晰的一张'],
    review: ['待你确认', '算法拿不准，交给你的审美判断'],
    reject: ['建议淘汰', '模糊、曝光异常或相似组中的较弱照片'],
    kept: ['你已保留', '你亲自确认过的照片'],
  };
  $('viewEyebrow').textContent = labels[state.filter][0];
  $('viewTitle').textContent = labels[state.filter][1];
  $('noResults').hidden = photos.length > 0;

  gallery.replaceChildren(...photos.map((photo) => {
    const card = document.createElement('article');
    card.className = `photo-card${photo.decision ? ' selected' : ''}`;
    card.innerHTML = `
      <div class="thumb-wrap">
        <img src="${photo.url}" alt="${escapeHtml(photo.file.name)}" loading="lazy" />
        ${photo.groupId ? `<span class="rank-badge">相似组 #${photo.groupId} · ${photo.groupRank}</span>` : ''}
        ${photo.decision ? `<span class="decision-badge">${photo.decision === 'keep' ? '已保留' : '已淘汰'}</span>` : ''}
      </div>
      <div class="photo-meta">
        <strong class="photo-name" title="${escapeHtml(photo.file.name)}">${escapeHtml(photo.file.name)}</strong>
        <div class="score-row"><span class="reason">${photo.reason}</span><span class="score">${photo.score}</span></div>
        <div class="quick-actions">
          <button data-action="reject" class="${photo.decision === 'reject' ? 'is-reject' : ''}">淘汰</button>
          <button data-action="keep" class="${photo.decision === 'keep' ? 'is-keep' : ''}">保留</button>
        </div>
      </div>`;
    card.querySelector('.thumb-wrap').addEventListener('click', () => openViewer(photo.id));
    card.querySelectorAll('[data-action]').forEach((button) => button.addEventListener('click', () => setDecision(photo.id, button.dataset.action)));
    return card;
  }));

  $('summaryBest').textContent = state.photos.filter((p) => p.category === 'best').length;
  $('summaryDuplicates').textContent = state.photos.filter((p) => p.duplicate).length;
  $('summaryIssues').textContent = state.photos.filter((p) => p.category === 'reject' && !p.duplicate).length;
  const possibleSaved = state.photos.filter((p) => p.category === 'reject').length;
  $('summaryCopy').textContent = possibleSaved ? `优先复核 ${possibleSaved} 张建议淘汰照片，可明显减少挑选量。` : '照片整体质量不错，可以从推荐保留开始看。';
}

function updateCounts() {
  $('countAll').textContent = state.photos.length;
  ['best', 'review', 'reject'].forEach((key) => { $(`count${key[0].toUpperCase()}${key.slice(1)}`).textContent = state.photos.filter((p) => p.category === key).length; });
  $('countKept').textContent = state.photos.filter((p) => p.decision === 'keep').length;
}

function updateProgress(done, total, message) {
  const percent = total ? Math.round(done / total * 100) : 0;
  $('progressText').textContent = message;
  $('progressPercent').textContent = `${percent}%`;
  $('progressBar').style.width = `${percent}%`;
}

function setDecision(id, decision) {
  const photo = state.photos.find((item) => item.id === id);
  if (!photo) return;
  photo.decision = photo.decision === decision ? null : decision;
  render();
  if ($('viewer').open) updateViewer(photo);
}

function openViewer(id) {
  state.viewerIndex = currentPhotos().findIndex((photo) => photo.id === id);
  updateViewer(currentPhotos()[state.viewerIndex]);
  $('viewer').showModal();
}

function updateViewer(photo) {
  if (!photo) return;
  $('viewerImage').src = photo.url;
  $('viewerName').textContent = photo.file.name;
  $('viewerReason').textContent = `${photo.reason} · 清晰度 ${Math.round(Math.min(100, Math.log10(photo.sharpness + 1) * 28))} · 亮度 ${Math.round(photo.brightness)}`;
  $('viewerScore').textContent = photo.score;
  $('viewerStatus').textContent = photo.decision === 'keep' ? '你已保留' : photo.decision === 'reject' ? '你已淘汰' : ({ best: '推荐保留', review: '待你确认', reject: '建议淘汰' }[photo.category]);
  $('viewerKeep').classList.toggle('is-active', photo.decision === 'keep');
  $('viewerReject').classList.toggle('is-active', photo.decision === 'reject');
}

function decideCurrent(decision) {
  const photo = currentPhotos()[state.viewerIndex];
  if (!photo) return;
  setDecision(photo.id, decision);
  moveViewer(1);
}

function moveViewer(direction) {
  const photos = currentPhotos();
  if (!photos.length) return;
  state.viewerIndex = (state.viewerIndex + direction + photos.length) % photos.length;
  updateViewer(photos[state.viewerIndex]);
}

function exportManifest() {
  if (!state.photos.length) return;
  const rows = [['文件名', '算法建议', '你的选择', '综合分', '原因', '相似组']];
  state.photos.forEach((p) => rows.push([
    p.file.name,
    ({ best: '推荐保留', review: '待确认', reject: '建议淘汰' })[p.category],
    p.decision === 'keep' ? '保留' : p.decision === 'reject' ? '淘汰' : '未决定',
    p.score,
    p.reason,
    p.groupId ? `#${p.groupId}-${p.groupRank}` : '',
  ]));
  const csv = `\uFEFF${rows.map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(',')).join('\n')}`;
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  link.download = `拣片清单-${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
}
