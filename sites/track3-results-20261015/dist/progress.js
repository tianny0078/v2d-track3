(() => {
  'use strict';
  const data = window.V2D_PROGRESS;
  if (!data) return;
  const labels = {completed:'已完成', running:'进行中', pending:'待执行', failed:'修复中'};
  const completed = data.milestones.filter(s => s.state === 'completed').length;
  const total = data.milestones.length;
  const progress = document.getElementById('overall-progress'); progress.max = total; progress.value = completed;
  document.getElementById('overall-count').textContent = completed + ' / ' + total;
  document.getElementById('overall-percent').textContent = Math.round(completed / total * 100) + '%';
  document.getElementById('overview-current-name').textContent = data.current_stage;
  document.getElementById('overview-updated').textContent = data.updated;
  const overview = document.getElementById('overview-steps');
  data.milestones.forEach((item, index) => {
    const step = document.createElement('a'); step.className = 'overview-step ' + item.state; step.href = '#step-' + item.id;
    const number = document.createElement('span'); number.className = 'step-number'; number.textContent = item.state === 'completed' ? '✓' : String(index + 1).padStart(2,'0');
    const name = document.createElement('strong'); name.textContent = item.name;
    const state = document.createElement('small'); state.textContent = labels[item.state] || item.state;
    step.title = item.detail; step.append(number,name,state); overview.append(step);
  });
  document.getElementById('current-stage').textContent = data.current_stage;
  document.getElementById('current-state').textContent = labels[data.state] || data.state;
  document.getElementById('current-detail').textContent = data.detail;
  document.getElementById('progress-time').textContent = '最后更新：' + data.updated;
  document.getElementById('current-strategy').textContent = data.strategy;
  document.getElementById('live-limitations').textContent = data.limitations.join(' ');
  document.getElementById('hourly-update-label').textContent = data.hourly_updates || '步骤完成后更新';
  const experiments = document.getElementById('experiment-list');
  for (const item of data.experiments || []) {
    const card=document.createElement('div');card.className='panel milestone-card';
    const name=document.createElement('h3');name.textContent=item.name+' · '+(labels[item.state]||item.state);
    const detail=document.createElement('p');detail.textContent=item.detail;card.append(name,detail);
    if (item.links && item.links.length) {
      const links=document.createElement('p');links.className='caption';
      for (const itemLink of item.links) {
        const a=document.createElement('a');a.href=itemLink.url;a.textContent=itemLink.label+' ↗';a.target='_blank';a.rel='noopener';
        if (links.childNodes.length) links.append(document.createTextNode(' · '));links.append(a);
      }
      card.append(links);
    }
    experiments.append(card);
  }
  const list = document.getElementById('live-milestones');
  for (const item of data.milestones) {
    const card = document.createElement('div');
    card.className = 'milestone-card';
    card.id = 'step-' + item.id;
    const state = document.createElement('span');
    state.className = 'milestone-state ' + item.state;
    state.textContent = labels[item.state] || item.state;
    const name = document.createElement('h3'); name.textContent = item.name;
    const detail = document.createElement('p'); detail.textContent = item.detail;
    card.append(state, name, detail); list.append(card);
  }
})();
