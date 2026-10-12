const $ = id => document.getElementById(id);
let session = JSON.parse(sessionStorage.getItem('m3sculptor-session') || '{}');
let busy = false, connected = false, values = {}, lastRender = '', lastSteps = '';
let serverMode = 'normal';
const ready = () => `Ready · ${serverMode} mode`;
const save = () => sessionStorage.setItem('m3sculptor-session', JSON.stringify(session));
function error(text) { $('error').textContent = text || ''; $('error').hidden = !text; }
function lock(value) {
  busy = value;
  $('send').disabled = value || !connected || !!session.running;
  $('new').disabled = value; $('connect').disabled = value;
  document.querySelectorAll('.choose').forEach(b => b.disabled = value || !connected || !!session.running);
}
async function api(path, body) {
  const response = await fetch('/api' + path, body === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error([data.error, data.detail].filter(Boolean).join(' ') || `Request failed (${response.status})`);
  return data;
}
function imageSource(block) {
  let url = block.image_url?.url || (typeof block.image_url === 'string' ? block.image_url : null) || block.url;
  const source = block.source || block;
  if (source.data) url = `data:${source.mime_type || source.media_type || 'image/png'};base64,${source.data}`;
  return typeof url === 'string' && /^(data:image\/(png|jpeg|webp|gif);base64,|https?:\/\/)/i.test(url) ? url : null;
}
function elapsed(step) {
  const end = step.ended ?? Date.now()/1000;
  const seconds = Math.max(0, end - step.started);
  return seconds < 1 ? '' : (seconds < 60 ? `${seconds.toFixed(0)}s` : `${Math.floor(seconds/60)}m${String(Math.round(seconds%60)).padStart(2,'0')}s`);
}
// Progress mirrors what the workflow reported; it never invents a step.
function renderProgress(snapshot) {
  const steps = snapshot.steps || [], running = snapshot.status === 'running';
  const box = $('progress');
  box.hidden = !running && !(steps.length && snapshot.status === 'error');
  if (box.hidden) { $('progress-steps').replaceChildren(); lastSteps=''; return; }
  $('progress-title').textContent = running ? 'Agent is working...' : (snapshot.status === 'error' ? 'Turn interrupted' : 'Turn complete');
  box.querySelector('.spinner').hidden = !running;
  const list = $('progress-steps');
  const signature = JSON.stringify(steps.map(s=>[s.label,s.detail,s.state]));
  if (signature === lastSteps) {  // Same steps: only refresh the elapsed times, keep the spinner spinning.
    steps.forEach((step,i)=>{const cell=list.children[i]?.querySelector('.detail');
      if(cell) cell.textContent=[step.detail, elapsed(step)].filter(Boolean).join(' · ');});
    return;
  }
  lastSteps = signature;
  list.replaceChildren();
  for (const step of steps) {
    const item = document.createElement('li'); item.className = step.state;
    const icon = document.createElement('span'); icon.className = 'icon';
    if (step.state === 'active') { const s = document.createElement('span'); s.className = 'spinner'; icon.append(s); }
    else icon.textContent = step.state === 'failed' ? '×' : '✓';
    item.append(icon);
    const label = document.createElement('span'); label.textContent = step.label; item.append(label);
    const detail = document.createElement('span'); detail.className = 'detail';
    detail.textContent = [step.detail, elapsed(step)].filter(Boolean).join(' · ');
    item.append(detail);
    list.append(item);
  }
  box.scrollTop = box.scrollHeight;
}
function render(state) {
  values = state.values || {};
  renderProgress(state);
  const messages = values.messages || [], signature = JSON.stringify([messages, values.waiting_for_selection]);
  if (signature !== lastRender && messages.length) {
    lastRender = signature; $('messages').replaceChildren();
    let latestAI = -1;
    messages.forEach((m,i) => { if (['ai','assistant'].includes(m.type || m.role)) latestAI = i; });
    messages.forEach((m, messageIndex) => {
      const role = m.type || m.role;
      if (!['ai','assistant','human','user'].includes(role)) return;
      const card = document.createElement('article'); card.className = 'message ' + (['human','user'].includes(role) ? 'human' : 'ai');
      const label = document.createElement('div'); label.className = 'role'; label.textContent = card.classList.contains('human') ? 'YOU' : 'M3SCULPTOR'; card.append(label);
      const content = m.content_blocks || m.content || '';
      for (const block of (typeof content === 'string' ? [{type:'text',text:content}] : content)) {
        if (typeof block === 'string' || block.type === 'text') {
          const p = document.createElement('div'); p.className = 'text'; p.textContent = typeof block === 'string' ? block : block.text; card.append(p);
        } else if (['image','image_url'].includes(block.type)) {
          const src = imageSource(block), figure = document.createElement('div'); figure.className = 'figure';
          if (src) { const img = document.createElement('img'); img.src = src; img.alt = 'Structure image'; img.onerror = () => {img.alt='Image failed to load';}; figure.append(img); }
          const choice = block.metadata?.selection_index;
          // Never guess numbering when some backend image files are unavailable.
          if (messageIndex === latestAI && values.waiting_for_selection && Number.isInteger(choice)) {
            const button = document.createElement('button'); button.className='choose'; button.textContent=`Select image ${choice}`;
            button.onclick=()=>send(String(choice)); figure.append(button);
          }
          card.append(figure);
        }
      }
      $('messages').append(card);
    });
    $('messages').scrollTop=$('messages').scrollHeight;
  }
  $('paper').textContent=values.retrieved_title || 'No paper retrieved yet';
  const doi=values.retrieved_doi; $('doi').hidden=!doi; $('doi').textContent=doi || '';
  if (doi) $('doi').href='https://doi.org/'+encodeURI(doi);
  lock(busy);
}
async function poll() {
  lock(true); error(''); $('recover').hidden=true;
  try {
    for(;;) {
      const snapshot=await api(`/session/${session.id}/state`);
      render(snapshot);
      if(snapshot.status!=='running') {
        session.running=false; save();
        if(snapshot.status==='error') throw new Error(snapshot.error || 'The task failed. Your message will not be resent automatically.');
        break;
      }
      await new Promise(resolve=>setTimeout(resolve,1000));
    }
  } catch(e) {error(e.message); $('recover').hidden=!session.id;}
  finally {lock(false);}
}
async function recover() {
  lock(true); error('');
  try {
    const snapshot=await api(`/session/${session.id}/state`); render(snapshot);
    session.running = snapshot.status==='running'; save();
    if(session.running) await poll();
    else if(snapshot.status==='error') error(snapshot.error || 'The previous turn failed. Review the conversation before retrying.');
  } catch(e) {error(e.message);} finally {lock(false);}
}
async function send(text) {
  if(busy || !connected || !text.trim() || session.running) return;
  lock(true); error('');
  try {
    if(!session.id) {const created=await api('/session',{}); session.id=created.session_id; save();}
    session.running=true; save();
    const snapshot=await api(`/session/${session.id}/message`, {text});
    render(snapshot); $('input').value=''; await poll();
  } catch(e) {session.running=false; save(); error(e.message); $('recover').hidden=!session.id;} finally {lock(false);}
}
async function connect() {
  lock(true); error(''); $('connection').textContent='Connecting...';
  try {
    const config=await(await fetch('/api/config')).json(); serverMode=config.evaluation_mode;
    if(session.id) {
      try {
        const snapshot=await api(`/session/${session.id}/state`);
        connected=true; $('connection').textContent=ready();
        render(snapshot);
        if(snapshot.status==='running') await recover();
        return;
      } catch(e) {
        session={}; save(); values={}; lastRender=''; $('messages').replaceChildren(); render({values:{}});
      }
    }
    // Create the session now so a rejected POST surfaces here, not on first send.
    const created=await api('/session',{}); session.id=created.session_id; save();
    connected=true; $('connection').textContent=ready();
  } catch(e) {connected=false; $('connection').textContent='Not ready'; error(e.message);} finally {lock(false);}
}
// Everything is typed as natural language; the Agent owns parsing, tolerance and matching.
$('composer').onsubmit=e=>{e.preventDefault();send($('input').value);};
$('connect').onclick=connect; $('recover').onclick=recover;
$('new').onclick=()=>{if(session.id && !confirm('Start a new conversation? The current conversation will no longer be displayed.'))return;
  session={};save();values={};lastRender='';$('messages').replaceChildren();error('');$('recover').hidden=true;render({values:{}});lock(false);};
document.querySelectorAll('.examples button').forEach(b=>b.onclick=()=>{$('input').value=b.textContent;$('input').focus();});
connect();
