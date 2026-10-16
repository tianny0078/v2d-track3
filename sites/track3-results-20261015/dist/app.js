"use strict";
(() => {
  const data=window.V2D_RESULTS;
  if(!data){document.getElementById('updated').textContent='数据暂未载入';return;}
  const $=id=>document.getElementById(id),pct=n=>(100*n).toFixed(1)+'%',fmt=n=>n.toLocaleString('en-US');
  if(location.protocol==='file:')document.querySelectorAll('a[href="downloads/track3-results-offline.zip"]').forEach(a=>a.hidden=true);
  $('updated').textContent=data.updated;$('footer-time').textContent='SNAPSHOT · '+data.updated;
  $('stereo-stat').innerHTML=(100*data.stereoValid.learned[1]).toFixed(1)+'<small>%</small>';
  let selectedFrame=68,mode='depth',selectedMesh='blue_cup',renderMode='surface';
  const setPressed=(selector,attribute,value)=>document.querySelectorAll(selector).forEach(b=>{const active=b.dataset[attribute]===String(value);b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
  function imageTile(file,title,note=''){
    const fig=document.createElement('figure');fig.className='image-tile';
    const a=document.createElement('a');a.href=file;a.target='_blank';a.rel='noopener';
    const img=document.createElement('img');img.src=file;img.alt=`第 ${selectedFrame} 帧 · ${title}`;a.append(img);
    const caption=document.createElement('figcaption');const text=document.createElement('b');text.textContent=title;const detail=document.createElement('span');detail.textContent=note;caption.append(text,detail);fig.append(a,caption);return fig;
  }
  function updateFrames(){
    const f=data.keyframes.find(x=>x.frame===selectedFrame),grid=$('image-grid');grid.replaceChildren();grid.classList.toggle('two',mode!=='depth');
    const root='assets/'+f.prefix+'-';grid.append(imageTile(root+'rgb.jpg','RGB 参考','校准后的原图'));
    if(mode==='depth'){grid.append(imageTile(root+'sgbm.jpg','SGBM',`有效比例 ${pct(f.valid.sgbm)}`),imageTile(root+'learned.jpg','C-FoundationStereo',`有效比例 ${pct(f.valid.learned)}`));}
    else grid.append(imageTile(root+mode+'.jpg',mode==='cups'?'双杯时序分割':'独立手部 mask',mode==='cups'?'蓝杯 / 米色杯':'左右身份修复后'));
    $('depth-legend').hidden=mode!=='depth';$('coverage-line').hidden=mode!=='depth';$('coverage-line').replaceChildren();
    for(const [name,label] of [['blue_cup','蓝杯'],['beige_cup','米色杯']]){const span=document.createElement('span');span.textContent=`${label}深度覆盖：${pct(f.cupCoverage[name].sgbm)} → ${pct(f.cupCoverage[name].learned)}`;$('coverage-line').append(span);}
    $('frame-label').textContent='FRAME '+String(selectedFrame).padStart(6,'0');
    $('image-caption').textContent=mode==='depth'?'深度显示范围 0.25–1.20 m；覆盖率提高不等于深度准确率提高。点击图像查看原图。':'分割来自模型预测。展示关键帧用于质量复核，模型置信度不等于 GT 准确率。';
  }
  document.querySelectorAll('[data-frame]').forEach(b=>b.addEventListener('click',()=>{selectedFrame=Number(b.dataset.frame);setPressed('[data-frame]','frame',selectedFrame);updateFrames();}));
  document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;setPressed('[data-mode]','mode',mode);updateFrames();}));updateFrames();
  for(const row of data.tradeoff){const tr=document.createElement('tr');if(row.alpha===.5)tr.className='selected';for(const value of [row.alpha.toFixed(2),(row.absolute_surface_depth_residual_m_p50_p95[0]*100).toFixed(2)+' cm',row.median_2d_reprojection_pixels_p50_p95[0].toFixed(2)+' px',row.median_mask_iou.toFixed(3)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('tradeoff-rows').append(tr);}
  for(const stage of data.stages){const item=document.createElement('div');item.className='stage';const h=document.createElement('h3');h.textContent=stage.name;const p=document.createElement('p');p.textContent=stage.detail;item.append(h,p);$('stage-list').append(item);}
  for(const next of data.next){const li=document.createElement('li');li.textContent=next;$('next-list').append(li);}
  function updateMeshFacts(){const m=data.meshes.find(x=>x.id===selectedMesh);$('mesh-name').textContent=m.label;$('mesh-download').href=m.file;$('mesh-facts').replaceChildren();
    $('mesh-interpretation').textContent=m.kind==='programmatic'?'空心杯壳的材料表面封闭，杯口和内腔保留。3mm壁厚是假设；当前几何与位姿联合质量未通过，不作为最终碰撞资产。':'每只杯融合137帧；未观测背面保留为空。参考系来自首帧可见表面，网格仍有边界和缺口。';
    for(const [label,value] of [['顶点',fmt(m.vertices)],['三角面',fmt(m.faces)],[m.kind==='programmatic'?'候选壁厚':'融合帧数',m.kind==='programmatic'?m.wall_mm+' mm（假设）':m.frames_integrated+' 帧'],['边界边',fmt(m.boundary_edges)],['是否封闭',m.watertight?'是':'否'],['尺寸范围',m.extents_mm.map(x=>Math.round(x)).join(' × ')+' mm']]){const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;row.append(dt,dd);$('mesh-facts').append(row);}}
  let viewer=null;
  document.querySelectorAll('[data-mesh]').forEach(b=>b.addEventListener('click',()=>{selectedMesh=b.dataset.mesh;setPressed('[data-mesh]','mesh',selectedMesh);updateMeshFacts();if(viewer)viewer.setMesh(selectedMesh);}));
  document.querySelectorAll('[data-render]').forEach(b=>b.addEventListener('click',()=>{renderMode=b.dataset.render;setPressed('[data-render]','render',renderMode);if(viewer)viewer.draw();}));
  $('reset-view').addEventListener('click',()=>viewer&&viewer.reset());updateMeshFacts();

  function startViewer(){const script=document.createElement('script');script.src='assets/mesh-data.js';script.onload=()=>{try{viewer=createViewer($('mesh-canvas'),window.V2D_MESHES,()=>renderMode);viewer.setMesh(selectedMesh);$('mesh-loading').hidden=true;}catch(error){$('mesh-loading').textContent='当前浏览器未能启用三维预览，可下载 PLY 查看。';console.warn(error.message);}};script.onerror=()=>{$('mesh-loading').textContent='网格数据载入失败，可使用旁边的 PLY 下载。';};document.head.append(script);}
  if('IntersectionObserver'in window){const observer=new IntersectionObserver(entries=>{if(entries.some(x=>x.isIntersecting)){observer.disconnect();startViewer();}},{rootMargin:'300px'});observer.observe($('mesh-canvas'));}else startViewer();

  function createViewer(canvas,meshes,getMode){
    const gl=canvas.getContext('webgl2',{antialias:true,alpha:true});if(!gl)throw new Error('WebGL2 unavailable');
    function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
    const vs=shader(gl.VERTEX_SHADER,`#version 300 es
      in vec3 position;in vec3 normal;uniform mat4 model;uniform mat4 projection;uniform float pointSize;out vec3 n;
      void main(){vec4 p=model*vec4(position,1.0);n=normalize(mat3(model)*normal);gl_Position=projection*p;gl_PointSize=pointSize;}`);
    const fs=shader(gl.FRAGMENT_SHADER,`#version 300 es
      precision highp float;in vec3 n;uniform vec3 color;uniform bool points;out vec4 outColor;
      void main(){if(points&&length(gl_PointCoord-vec2(.5))>.5)discard;float light=.48+.52*abs(dot(normalize(n),normalize(vec3(-.5,.8,1.0))));outColor=vec4(color*light,1.0);}`);
    const program=gl.createProgram();gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));gl.useProgram(program);
    const locations={};for(const name of ['model','projection','color','pointSize','points'])locations[name]=gl.getUniformLocation(program,name);
    const cache={};let mesh=null,yaw=0,pitch=.14,zoom=1,drag=null;
    function decode(s,Type){const raw=atob(s),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Type(bytes.buffer);}
    function upload(name){if(cache[name])return cache[name];const source=meshes[name],vao=gl.createVertexArray();gl.bindVertexArray(vao);
      for(const [attr,data] of [['position',source.positions],['normal',source.normals]]){const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,decode(data,Float32Array),gl.STATIC_DRAW);const location=gl.getAttribLocation(program,attr);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,3,gl.FLOAT,false,0,0);}
      const index=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,index);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,decode(source.indices,Uint32Array),gl.STATIC_DRAW);return cache[name]={...source,vao};}
    function multiply(a,b){const out=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)out[c*4+r]+=a[k*4+r]*b[c*4+k];return out;}
    function draw(){if(!mesh)return;const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.useProgram(program);gl.bindVertexArray(mesh.vao);
      const c=Math.cos(yaw),s=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),scale=1.8/mesh.extent;
      const center=mesh.center,T=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,-center[0],-center[1],-center[2],1]);
      const S=new Float32Array([scale,0,0,0,0,-scale,0,0,0,0,-scale,0,0,0,0,1]);
      const X=new Float32Array([1,0,0,0,0,cx,sx,0,0,-sx,cx,0,0,0,0,1]);const Y=new Float32Array([c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1]);
      const V=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,-3.6/zoom,1]);const model=multiply(V,multiply(Y,multiply(X,multiply(S,T))));
      const f=1/Math.tan(Math.PI/8),aspect=canvas.width/canvas.height,near=.05,far=30;const P=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
      gl.uniformMatrix4fv(locations.model,false,model);gl.uniformMatrix4fv(locations.projection,false,P);gl.uniform3fv(locations.color,mesh.color);gl.uniform1f(locations.pointSize,2.1*ratio);const points=getMode()==='points';gl.uniform1i(locations.points,points?1:0);if(points)gl.drawArrays(gl.POINTS,0,mesh.vertexCount);else gl.drawElements(gl.TRIANGLES,mesh.indexCount,gl.UNSIGNED_INT,0);
    }
    canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.009;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-drag[1])*.009));drag=[e.clientX,e.clientY];draw();});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
    canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.6,Math.min(2.6,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});canvas.addEventListener('keydown',e=>{let handled=true;if(e.key==='ArrowLeft')yaw-=.12;else if(e.key==='ArrowRight')yaw+=.12;else if(e.key==='ArrowUp')pitch-=.12;else if(e.key==='ArrowDown')pitch+=.12;else if(e.key==='+')zoom=Math.min(2.6,zoom*1.1);else if(e.key==='-')zoom=Math.max(.6,zoom/1.1);else handled=false;if(handled){e.preventDefault();draw();}});
    new ResizeObserver(draw).observe(canvas);return{draw,setMesh(name){mesh=upload(name);draw();},reset(){yaw=0;pitch=.14;zoom=1;draw();}};
  }
})();
