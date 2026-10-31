/* Live diagnostics from completed actions and existing optimizer log lines. */
(() => {
  const colors = {reward:'#7fe0bc', mean:'#efc282', x:'#7fe0bc', y:'#82b9ff',
                  z:'#efc282', gripper:'#c8a2ff', loss:'#82b9ff'};
  const specs = [
    ['reward', 'Reward per decision', [['reward','Reward',colors.reward], ['mean','Mean of last 10',colors.mean]]],
    ['return', 'Cumulative episode reward', [['return','Reward sum',colors.reward]]],
    ['goal', 'Goal conditions met', [['goal','Goal fraction',colors.reward]]],
    ['targets', 'Commanded position targets · meters', [['x','X',colors.x],['y','Y',colors.y],['z','Z',colors.z]]],
    ['gripper', 'Commanded gripper · 0 open / 1 closed', [['gripper','Gripper',colors.gripper]]],
    ['loss', 'Optimizer value loss · backward pass', [['value_loss','Value loss',colors.loss]]],
  ];
  const byId = id => document.getElementById(id);
  const number = n => n == null ? '—' : Number(n).toLocaleString(undefined, {maximumFractionDigits:4});
  const section = document.createElement('section');
  section.className = 'panel';
  section.innerHTML = '<div class="section-head"><div><h2>Live decision & optimizer curves</h2><p class="caption" id="live-scope">Per decision · selected episode</p></div><span class="tag" id="live-optimizer">Waiting for optimizer</span></div><div class="stats-line" id="live-summary"></div><div class="live-charts" id="live-charts"></div><p class="caption" style="margin-top:12px">Hover points for values; click a decision to inspect its action. Targets show commands, not measured motion. Reward curves describe rollouts; optimizer loss describes training.</p>';
  document.querySelector('.layout').before(section);
  const style = document.createElement('style');
  style.textContent = '.live-charts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:15px}.live-chart{border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}.live-chart svg{width:100%;height:150px;display:block}.live-title{font-size:12px;display:flex;justify-content:space-between;gap:8px}.live-value{color:var(--mint)}.live-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:var(--muted);min-height:20px}.live-legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}.live-point{cursor:pointer}.live-point:focus{outline:none;stroke:white;stroke-width:3}@media(max-width:1100px){.live-charts{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.live-charts{grid-template-columns:1fr}}';
  document.head.append(style);
  for (const [id,label,series] of specs) {
    const panel = document.createElement('article');
    panel.className = 'live-chart';
    const title = document.createElement('div'); title.className = 'live-title';
    const name = document.createElement('span'); name.textContent = label;
    const value = document.createElement('span'); value.className='live-value'; value.id='live-value-'+id;
    title.append(name,value);
    const legend = document.createElement('div'); legend.className='live-legend';
    for (const [,name,color] of series) {
      const item=document.createElement('span'), dot=document.createElement('i');
      dot.style.background=color; item.append(dot,document.createTextNode(name)); legend.append(item);
    }
    const host=document.createElement('div'); host.id='live-plot-'+id;
    panel.append(title,legend,host); byId('live-charts').append(panel);
  }

  function draw(id, rows, series, {percent=false, optimizer=false, steps=false}={}) {
    const host=byId('live-plot-'+id); host.replaceChildren();
    const values=rows.flatMap(r=>series.map(([key])=>r[key])).filter(Number.isFinite);
    if (!values.length) {
      const p=document.createElement('div');p.className='chart-empty';
      p.textContent=optimizer?'Appears during backward passes; no restart of training needed':'No recorded values yet';
      host.append(p);byId('live-value-'+id).textContent='—';return;
    }
    const last=rows.at(-1);
    byId('live-value-'+id).textContent=series.length===1?(percent?number(last[series[0][0]]*100)+'%':number(last[series[0][0]])):'';
    const ns='http://www.w3.org/2000/svg', svg=document.createElementNS(ns,'svg');
    svg.setAttribute('viewBox','0 0 380 150');svg.setAttribute('role','img');
    svg.setAttribute('aria-label',specs.find(s=>s[0]===id)[1]+' across '+rows.length+' recorded points');
    const add=(tag,attrs,text)=>{const el=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);if(text!=null)el.textContent=text;svg.append(el);return el};
    const bounded=percent||id==='gripper';
    let lo=bounded?0:Math.min(...values), hi=bounded?1:Math.max(...values);
    if(lo===hi){const margin=Math.max(Math.abs(lo)*.1,.001);lo-=margin;hi+=margin;}
    else if(!bounded){const margin=(hi-lo)*.12;lo-=margin;hi+=margin;}
    if(id==='loss')lo=Math.max(0,lo);
    const left=48,right=368,top=12,bottom=122;
    const x=n=>left+(n-rows[0].step)/Math.max(1,rows.at(-1).step-rows[0].step)*(right-left);
    const y=n=>top+(hi-n)/(hi-lo)*(bottom-top);
    for(const t of [0,.5,1]){const yy=top+t*(bottom-top);add('line',{x1:left,x2:right,y1:yy,y2:yy,stroke:'#273540','stroke-dasharray':'3 4'});add('text',{x:1,y:yy+4,fill:'#94a6b4','font-size':9},percent?Math.round((hi-t*(hi-lo))*100)+'%':number(hi-t*(hi-lo)));}
    for(const [key,name,color]of series){
      let d='',prev=null;
      for(const r of rows){if(!Number.isFinite(r[key])){prev=null;continue;}
        d+=prev?(steps?`H${x(r.step)}V${y(r[key])}`:`L${x(r.step)},${y(r[key])}`):`M${x(r.step)},${y(r[key])}`;prev=r;
        const dot=add('circle',{cx:x(r.step),cy:y(r[key]),r:2.7,fill:r.invalid&&key==='reward'?'#ff9d9d':color,class:'live-point',tabindex:0});
        const tip=document.createElementNS(ns,'title');
        tip.textContent=`${optimizer?'Backward pass':'Decision'} ${r.step} · ${name}: ${number(r[key])}${r.tool?' · '+r.tool+' · '+r.status:''}`;dot.append(tip);
        if(!optimizer){const select=()=>window.selectLiveDecision?.(r.step);dot.onclick=select;dot.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select()}};}
      }
      // Keep dots above the curve for pointer and keyboard interaction.
      const path=add('path',{d,fill:'none',stroke:color,'stroke-width':key==='mean'?2:1.4,'pointer-events':'none'});
      svg.insertBefore(path,svg.firstChild);
    }
    for(const [value,anchor]of [[rows[0].step,'start'],[rows.at(-1).step,'end']])add('text',{x:x(value),y:145,fill:'#94a6b4','font-size':10,'text-anchor':anchor},(optimizer?'pass ':'decision ')+value);
    host.append(svg);
  }

  let signature='';
  window.renderLive = s => {
    const optim=s.optimizer||{points:[],completed:0,total:0};
    byId('live-optimizer').textContent=optim.total?`Update ${optim.update??(s.phase==='optimizing'?s.update+1:s.update)} · backward ${optim.completed}/${optim.total}`:'Optimizer has not started';
    byId('live-scope').textContent='Selected episode: '+(s.selected||'waiting')+' · refreshes every 2 seconds';
    const next=JSON.stringify([s.selected,s.turns,optim]);
    if(next===signature)return;signature=next;
    let cumulative=0,windowRewards=[],targets={x:null,y:null,z:null,gripper:null};
    const rows=s.turns.map(r=>{
      const reward=Number.isFinite(r.reward)?r.reward:0;cumulative+=reward;
      windowRewards.push(reward);if(windowRewards.length>10)windowRewards.shift();
      const commanded=r.parsed?.[1]?.targets||{};
      for(const axis of Object.keys(targets))if(Number.isFinite(commanded[axis]))targets[axis]=commanded[axis];
      return {step:r.decision,reward,mean:windowRewards.reduce((a,b)=>a+b,0)/windowRewards.length,
        return:cumulative,goal:r.goal_fraction,...targets,tool:r.tool,status:r.status,invalid:r.status==='invalid'};
    });
    const invalid=s.turns.filter(r=>r.status==='invalid').length;
    const summary=byId('live-summary');summary.replaceChildren();
    for(const text of [`${rows.length} decisions`,`Reward sum ${number(cumulative)}`,`Mean reward ${rows.length?number(cumulative/rows.length):'—'}`,`${invalid} invalid calls`]){const el=document.createElement('span');el.textContent=text;summary.append(el)}
    for(const [id,,series]of specs)draw(id,id==='loss'?optim.points:rows,series,
      {percent:id==='goal',optimizer:id==='loss',steps:id==='targets'||id==='gripper'});
  };
})();
