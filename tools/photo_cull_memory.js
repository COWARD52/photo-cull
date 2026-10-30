// Included in the generated report after the main script; no AI request runs on page load.
let memoryState = {};
const memoryMessage = memoryDialog.querySelector('.memory-message');

async function memoryApi(path, payload) {
  const response = await fetch(path, payload === undefined ? {cache: 'no-store'} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  });
  const value = await response.json();
  if (!response.ok || !value.ok) throw new Error(value.error || '操作失败');
  return value;
}

function memoryLine(parent, tag, text) {
  const element = document.createElement(tag);
  element.textContent = text;
  parent.append(element);
  return element;
}

function renderMemory() {
  const snapshot = memoryState.snapshot || {};
  const draft = memoryState.draft || {};
  const rules = memoryState.memory?.rules || [];
  const run = memoryDialog.querySelector('#memory-run');
  const draftBox = memoryDialog.querySelector('#memory-draft');
  const savedBox = memoryDialog.querySelector('#memory-saved');
  const summarize = memoryDialog.querySelector('#memory-summarize');
  const review = memoryDialog.querySelector('#memory-review');
  draftBox.replaceChildren();
  savedBox.replaceChildren();
  if (!snapshot.id) {
    run.textContent = '尚未确认整理；完成整理后，可从最终取舍中挑选代表性案例。';
  } else {
    run.textContent = `已确认 ${snapshot.total} 张；其中 ${snapshot.changed} 张推翻初筛，选出 ${snapshot.samples.length} 张代表性预览图。`;
    if (snapshot.samples.length) {
      const sampleNames = snapshot.samples.map(item => item.name).join('、');
      memoryLine(draftBox, 'small', '候选案例：' + sampleNames);
    }
  }
  const parseError = draft.snapshot_id === snapshot.id && draft.status === 'parse_error';
  summarize.disabled = !snapshot.id || !snapshot.samples?.length || (draft.snapshot_id === snapshot.id && !parseError);
  summarize.textContent = parseError ? '重新请求总结（再次计费）' : '总结本次偏好';
  if (draft.snapshot_id === snapshot.id && draft.status === 'draft') {
    memoryLine(draftBox, 'p', 'AI 草稿尚未写入 Memory。请逐条审核；取消全部勾选即可拒绝本次结论。');
    for (const rule of draft.rules || []) {
      const box = document.createElement('div'); box.className = 'memory-rule';
      const label = document.createElement('label');
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = true;
      const input = document.createElement('input'); input.type = 'text'; input.maxLength = 180; input.value = rule.text;
      label.append(checkbox, input);
      box.append(label);
      memoryLine(box, 'small', `${rule.scene} · ${rule.dimension} · 证据：${rule.evidence.join('、')} · ${rule.evidence_count || 1} 次独立取舍 · ${rule.confidence === 'guess' ? '暂为猜测' : '中等可信'}`);
      box.dataset.scene = rule.scene; box.dataset.dimension = rule.dimension;
      box._evidence = rule.evidence;
      draftBox.append(box);
    }
    review.hidden = false;
    review.textContent = (draft.rules || []).length ? '审核后写入' : '确认无可用偏好';
  } else {
    review.hidden = true;
    if (draft.status === 'reviewed') memoryLine(draftBox, 'p', '本次总结已审核，不会再次自动调用 AI。');
    if (parseError) {
      memoryLine(draftBox, 'p', '模型已经返回内容，但格式无法解析。原回答已本地暂存；重新请求会再次消耗额度。');
      memoryLine(draftBox, 'pre', draft.raw_response || '未保存到回答内容');
    }
  }
  if (!rules.length) memoryLine(savedBox, 'p', '暂无已确认的跨目录偏好。');
  for (const rule of rules) {
    const box = document.createElement('div'); box.className = 'memory-rule';
    const input = document.createElement('input'); input.type = 'text'; input.maxLength = 180; input.value = rule.text;
    box.append(input);
    memoryLine(box, 'small', `${rule.scene} · ${rule.dimension} · ${rule.evidence_count || 1} 次独立取舍 · ${rule.confidence === 'guess' ? '样本较少，暂为猜测' : '中等可信'} · 案例 ${rule.evidence?.join('、') || '—'}`);
    const actions = document.createElement('div'); actions.className = 'memory-actions';
    const save = memoryLine(actions, 'button', '保存修改');
    const remove = memoryLine(actions, 'button', '删除');
    save.onclick = async () => {
      try { const result = await memoryApi('/api/memory/update', {id: rule.id, text: input.value}); memoryState.memory = result.memory; memoryMessage.textContent = '已更新本机偏好。'; renderMemory(); }
      catch (error) { memoryMessage.textContent = error.message; }
    };
    remove.onclick = async () => {
      if (!confirm('删除这条已确认的偏好？')) return;
      try { const result = await memoryApi('/api/memory/delete', {id: rule.id}); memoryState.memory = result.memory; memoryMessage.textContent = '已删除。'; renderMemory(); }
      catch (error) { memoryMessage.textContent = error.message; }
    };
    box.append(actions); savedBox.append(box);
  }
}

async function openMemory() {
  if (location.protocol === 'file:') { alert('请通过本地服务打开报告后使用选片偏好。'); return; }
  memoryState = {}; renderMemory();
  memoryMessage.textContent = '正在读取本机偏好…';
  memoryDialog.showModal();
  try { memoryState = await memoryApi('/api/memory'); memoryMessage.textContent = ''; renderMemory(); }
  catch (error) { memoryMessage.textContent = '读取失败：' + error.message; }
}

document.querySelector('#memory-open').onclick = openMemory;
memoryDialog.querySelector('#memory-close').onclick = () => memoryDialog.close();
memoryDialog.querySelector('#memory-summarize').onclick = async () => {
  const snapshot = memoryState.snapshot || {};
  if (!currentAIProfile()) { memoryMessage.textContent = '请先关闭此窗口，在“配置 AI”中保存并启用一个供应商。'; return; }
  const force = memoryState.draft?.status === 'parse_error';
  if (!confirm(`${force ? '本次已请求过一次；重新请求可能再次计费。\n' : ''}本次将发送 ${snapshot.samples.length} 张代表性预览图给当前 AI 供应商，可能产生费用。确认发送？`)) return;
  const button = memoryDialog.querySelector('#memory-summarize'); button.disabled = true;
  memoryMessage.textContent = '正在生成草稿；关闭窗口不会取消已发出的请求…';
  try {
    const mode = aiMode.value === 'api' || aiMode.value === 'codex' ? aiMode.value : 'compatible';
    const result = await memoryApi('/api/memory/summarize', {mode, model: aiModel.value, base_url: aiBase.value, api_key: aiKey.value, force});
    memoryState.draft = result.draft;
    memoryMessage.textContent = result.reused ? '已复用本次草稿，没有再次调用 AI。' : result.draft.status === 'parse_error' ? 'AI 已返回，但格式无法解析；原回答已暂存。' : '草稿已生成，请审核后写入。';
    renderMemory();
  } catch (error) { memoryMessage.textContent = '总结失败：' + error.message; button.disabled = false; }
};
memoryDialog.querySelector('#memory-review').onclick = async () => {
  const rules = [...memoryDialog.querySelectorAll('#memory-draft .memory-rule')]
    .filter(box => box.querySelector('input[type=checkbox]').checked)
    .map(box => ({text: box.querySelector('input[type=text]').value, scene: box.dataset.scene,
      dimension: box.dataset.dimension, evidence: box._evidence}));
  const button = memoryDialog.querySelector('#memory-review'); button.disabled = true;
  try {
    const result = await memoryApi('/api/memory/review', {snapshot_id: memoryState.snapshot.id, rules});
    memoryState.memory = result.memory; memoryState.draft.status = 'reviewed';
    memoryMessage.textContent = rules.length ? `已将 ${rules.length} 条审核后的偏好写入本机。` : '已拒绝本次草稿，没有写入偏好。';
    renderMemory();
  } catch (error) { memoryMessage.textContent = '保存失败：' + error.message; button.disabled = false; }
};
