/* Experiment switching and views for saved formats other than PPO. */
(() => {
  const el = id => document.getElementById(id);
  const make = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const num = x => x == null ? '—' : typeof x === 'number' ? x.toLocaleString(undefined,{maximumFractionDigits:5}) : String(x);
  const selector = make('select'); selector.id = 'experiment'; selector.setAttribute('aria-label','Select experiment');
  selector.style.maxWidth = '100%';
  const label = make('label','Experiment ', 'caption'); label.htmlFor = selector.id;
  const toolbar = make('div'); toolbar.style.marginTop = '12px'; toolbar.append(label, selector);
  el('run-name').after(toolbar);
  const note=make('p',null,'caption');note.id='experiment-note';toolbar.after(note);
  selector.onchange = () => window.changeExperiment(selector.value);
  const panel = make('section',null,'panel'); panel.id = 'experiment-data'; panel.hidden = true;
  document.querySelector('.layout').before(panel);
  const style=make('style');style.textContent='[hidden]{display:none!important}#experiment{max-width:650px;width:100%;font-size:12px}#experiment-data .charts{margin:18px 0}#experiment-data .cameras{margin-top:16px}#experiment-data pre{max-height:400px}';document.head.append(style);
  let catalogAt = 0;
  let lastOtherSignature='', detailOpen=null;
  window.refreshExperiments = async () => {
    if (Date.now() - catalogAt < 5000) return;
    const response = await fetch('/api/runs',{cache:'no-store'});
    if (!response.ok) throw new Error('Could not list experiments');
    const data = await response.json(); catalogAt = Date.now();
    const previous = selector.value;
    const signature = JSON.stringify(data.runs);
    if (selector.dataset.signature !== signature) {
      selector.replaceChildren();
      for (const run of data.runs) {
        const option = make('option', `${run.name} · ${run.kind.replaceAll('_',' ')}${run.running?' · ACTIVE':''}`);
        option.value = run.id; selector.append(option);
      }
      selector.dataset.signature = signature;
      if ([...selector.options].some(o=>o.value===previous)) selector.value = previous;
    }
  };
  window.experimentLoading = () => {
    lastOtherSignature='';delete panel.dataset.run;
    document.querySelector('.layout').hidden = true;
    document.querySelector('.cards').hidden = true;
    const live = el('live-charts')?.closest('section'); if(live)live.hidden = true;
    panel.hidden = false; panel.replaceChildren(make('p','Loading selected experiment…','caption'));
    for(const camera of ['agentview','wrist']) { el(camera).removeAttribute('src'); delete el(camera).dataset.source; }
  };
  window.prepareExperiment = s => {
    selector.value = s.run_id ?? '.';
    note.textContent=s.experiment_note?'Experiment notes · '+s.experiment_note:'';
    const ppo = s.kind === 'ppo';
    panel.hidden = ppo;
    document.querySelector('.layout').hidden = !ppo;
    document.querySelector('.cards').hidden = false;
    const live = el('live-charts')?.closest('section'); if(live)live.hidden = !ppo;
    const names = ['PPO updates','Current batch','Episode success','Run time'];
    document.querySelectorAll('.cards .card h3').forEach((n,i)=>{n.textContent=ppo?names[i]:(s.cards?.[i]?.label||'—')});
    document.querySelectorAll('.cards .bar').forEach(n=>{n.hidden=!ppo});
  };
  function chart(spec) {
    const box=make('article',null,'chart'); box.append(make('h3',spec.label));
    const rows=spec.rows.filter(r=>Number.isFinite(r[spec.key])&&Number.isFinite(r[spec.x]));
    if(!rows.length){box.append(make('p','No recorded values yet','chart-empty'));return box;}
    const last=rows.at(-1);box.append(make('p',num(last[spec.key]),'live-label'));
    const ns='http://www.w3.org/2000/svg', svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 360 125');
    svg.setAttribute('role','img');svg.setAttribute('aria-label',spec.label);
    const add=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!=null)n.textContent=text;svg.append(n);return n;};
    let lo=Math.min(...rows.map(r=>r[spec.key])),hi=Math.max(...rows.map(r=>r[spec.key]));
    if(spec.key==='success_rate'){lo=0;hi=1;}else if(lo===hi){lo-=.001;hi+=.001;}
    const x=v=>48+(v-rows[0][spec.x])/Math.max(1,last[spec.x]-rows[0][spec.x])*300,y=v=>10+(hi-v)/(hi-lo)*85;
    for(const t of [0,.5,1]){const yy=10+t*85;add('line',{x1:48,x2:348,y1:yy,y2:yy,stroke:'#273540'});add('text',{x:0,y:yy+4,fill:'#94a6b4','font-size':9},num(hi-t*(hi-lo)));}
    add('polyline',{points:rows.map(r=>`${x(r[spec.x])},${y(r[spec.key])}`).join(' '),stroke:'#7fe0bc',fill:'none','stroke-width':2});
    for(const r of rows){const dot=add('circle',{cx:x(r[spec.x]),cy:y(r[spec.key]),r:3,fill:'#7fe0bc'});const title=document.createElementNS(ns,'title');title.textContent=`${spec.x} ${r[spec.x]}: ${num(r[spec.key])}`;dot.append(title);}
    add('text',{x:48,y:120,fill:'#94a6b4','font-size':10},`${spec.x} ${rows[0][spec.x]}`);
    add('text',{x:348,y:120,fill:'#94a6b4','font-size':10,'text-anchor':'end'},`${spec.x} ${last[spec.x]}`);
    box.append(svg);return box;
  }
  function details(title, value, open=false) {
    const item=make('details');item.open=detailOpen?detailOpen.has(title):open;item.append(make('summary',title),make('pre',typeof value==='string'?value:JSON.stringify(value,null,2)));return item;
  }
  window.renderOther = s => {
    el('run-name').textContent=`${s.run} · ${s.kind.replaceAll('_',' ')}`;
    el('status').classList.toggle('offline',!s.running);
    el('status').querySelector('span').textContent=s.phase==='complete'?'Complete':s.running?'Process active':s.phase==='error'?'Error':'Not running';
    el('synced').textContent=`Synced ${new Date(s.now*1000).toLocaleTimeString()} · every 2s`;
    el('path').textContent=s.path+' · Read-only';
    const cards=document.querySelectorAll('.cards .card');
    cards.forEach((node,i)=>{const card=s.cards[i];node.querySelector('.value').textContent=num(card?.value);node.querySelector('.caption').textContent=card?.caption||'';});
    const signature=JSON.stringify([s.run_id,s.phase,s.pid,s.cards,s.series,s.records,s.summary,s.progress,s.checkpoints,s.selected,s.frames,s.episode_turns,s.gpus,s.log]);
    if(signature===lastOtherSignature)return;lastOtherSignature=signature;
    detailOpen=panel.dataset.run===s.run_id?new Set([...panel.querySelectorAll('details[open] summary')].map(n=>n.textContent)):null;
    panel.dataset.run=s.run_id;
    panel.replaceChildren(make('h2','Saved experiment data'),make('p',s.description,'caption'));
    panel.append(make('p',`Phase: ${s.phase} · PID: ${s.pid??'not running'} · Last saved activity: ${s.last_activity?new Date(s.last_activity*1000).toLocaleString():'none'}`,'caption'));
    const plots=make('div',null,'charts');for(const spec of s.series)plots.append(chart(spec));panel.append(plots);
    if(s.episodes.length){
      const selection=make('select');selection.setAttribute('aria-label','Select saved episode');
      for(const episode of s.episodes){const option=make('option',episode.id);option.value=episode.id;selection.append(option);}
      selection.value=s.selected;selection.onchange=()=>window.selectArchiveEpisode(selection.value);panel.append(make('p','Saved episode'),selection);
      const cameras=make('div',null,'cameras');
      for(const[name,frame]of Object.entries(s.frames)){const figure=make('figure',null,'camera');const img=make('img');img.alt=name+' saved observation';img.src='/frames/'+frame.path.split('/').map(encodeURIComponent).join('/')+'?run='+encodeURIComponent(s.run_id??'.')+'&v='+frame.modified;figure.append(img,make('figcaption',name));cameras.append(figure);}
      panel.append(cameras);
      if(s.episode_turns.length)panel.append(details('Saved actions for selected episode',s.episode_turns));
    }
    panel.append(details('Saved summary / results',s.summary,true));
    const records=make('div');records.append(make('h3',`${s.record_count} saved records · showing the latest ${s.records.length}`));
    for(const[index,row]of [...s.records].reverse().entries())records.append(details(`Record ${s.record_count-index}${row.epoch!=null?' · epoch '+row.epoch:row.status?' · '+row.status:''}`,row));
    if(!s.records.length)records.append(make('p','No completed records yet. An active model server may still be loading or generating its first reply.','caption'));
    panel.append(records,details('Configuration',s.config),details('Progress record',s.progress),details('Saved checkpoints',s.checkpoints),details('Data files',s.artifacts));
    panel.append(details('Terminal log tail',s.log||'No terminal log file was saved in this folder.'));
    const gpu=make('div');gpu.append(make('h3','Current machine GPUs · not historical run usage'));
    for(const g of s.gpus)gpu.append(make('p',`GPU ${g.index}: ${g.name} · ${(g.used/1024).toFixed(1)} / ${(g.total/1024).toFixed(1)} GB · ${g.utilization}% active`,'caption'));
    panel.append(gpu);
  };
})();
