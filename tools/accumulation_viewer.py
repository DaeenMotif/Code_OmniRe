"""Self-contained HTML player for the LiDAR accumulation exports.

`write_viewer(payload, path)` bakes the point clouds of one scene into a single
dependency-free page: a small WebGL point renderer plus a timeline that scrubs the
frame-by-frame accumulation of `DrivingDataset.get_init_objects`.
"""
import base64
import json

import numpy as np


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _pack(payload):
    out = {
        "scene": payload["scene"],
        "startTimestep": payload["start_timestep"],
        "numFrames": payload["num_frames"],
        "maxPts": payload["max_pts"],
        "instances": [],
    }
    for ins in payload["instances"]:
        out["instances"].append({
            "id": ins["id_in_dataset"],
            "uuid": ins["uuid"],
            "className": ins["class_name"],
            "nodeType": ins["node_type"],
            "boxSize": [float(v) for v in ins["box_size"]],
            "numVisibleFrames": ins["num_visible_frames"],
            "totalPoints": ins["total_points"],
            "perFrame": ins["points_per_frame"],
            "cum": ins["cumulative"],
            "saturatedAt": ins["budget_saturated_at_frame"],
            "trajLength": ins["trajectory_length_m"],
            "visible": _b64(ins["visible"].astype(np.uint8)),
            "pts": _b64(ins["pts"].astype(np.float32)),
            "colors": _b64(ins["colors"].astype(np.uint8)),
            "birth": _b64(ins["birth"].astype(np.uint16)),
            "poses": _b64(ins["poses"].astype(np.float32)),
        })
    return out


def write_viewer(payload, path):
    html = TEMPLATE.replace("__DATA__", json.dumps(_pack(payload)))
    with open(path, "w") as f:
        f.write(html)
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiDAR accumulation player</title>
<style>
  :root{
    color-scheme: dark;
    --bg:#121211; --surface-1:#1a1a19; --surface-2:#232321; --line:#33332f;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8a8a80;
    --series-1:#3987e5; --accent:#d95926;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text-primary);
       font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
       height:100vh;display:flex;flex-direction:column;overflow:hidden}
  header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;
         padding:10px 16px;background:var(--surface-1);border-bottom:1px solid var(--line)}
  h1{font-size:13px;font-weight:600;margin:0;letter-spacing:.01em}
  .scene{color:var(--text-muted);font-variant-numeric:tabular-nums}
  .pills{display:flex;gap:6px;flex-wrap:wrap}
  button{font:inherit;color:var(--text-secondary);background:var(--surface-2);
         border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
  button:hover{color:var(--text-primary);border-color:#4a4a45}
  button[aria-pressed="true"]{background:var(--series-1);border-color:var(--series-1);color:#fff}
  .group{display:flex;gap:6px;align-items:center}
  .group>label{color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
  main{flex:1;position:relative;min-height:0}
  canvas#gl{width:100%;height:100%;display:block;cursor:grab}
  canvas#gl:active{cursor:grabbing}
  #hud{position:absolute;top:12px;left:12px;background:rgba(18,18,17,.82);
       border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:210px}
  #hud dl{margin:0;display:grid;grid-template-columns:auto auto;gap:3px 14px}
  #hud dt{color:var(--text-muted);font-size:11px}
  #hud dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
  #ramp{margin-top:9px;display:none}
  #ramp .bar{height:7px;border-radius:3px;
    background:linear-gradient(90deg,#1b3a6b,#3987e5,#9ec9f7,#e8f2fd)}
  #ramp .ends{display:flex;justify-content:space-between;color:var(--text-muted);
              font-size:10px;margin-top:3px;font-variant-numeric:tabular-nums}
  footer{background:var(--surface-1);border-top:1px solid var(--line);padding:8px 16px 10px}
  .transport{display:flex;gap:10px;align-items:center;margin-bottom:4px;flex-wrap:wrap}
  .frame{font-variant-numeric:tabular-nums;color:var(--text-secondary)}
  .frame b{color:var(--text-primary)}
  #chartwrap{position:relative}
  canvas#chart{width:100%;height:76px;display:block;cursor:col-resize}
  #tip{position:absolute;pointer-events:none;display:none;background:var(--surface-2);
       border:1px solid var(--line);border-radius:6px;padding:4px 7px;font-size:11px;
       white-space:nowrap;font-variant-numeric:tabular-nums;transform:translate(-50%,-115%)}
  .hint{color:var(--text-muted);font-size:11px}
  @media (max-width:640px){ #hud{position:static;margin:8px} }
</style>
</head>
<body>
<header>
  <h1>LiDAR accumulation <span class="scene" id="sceneTag"></span></h1>
  <div class="group"><label>Instance</label><div class="pills" id="insPills"></div></div>
  <div class="group"><label>Budget</label><div class="pills">
    <button id="mCapped" aria-pressed="true">Capped</button>
    <button id="mUncapped" aria-pressed="false">Uncapped</button></div></div>
  <div class="group"><label>Color</label><div class="pills">
    <button id="cRgb" aria-pressed="true">LiDAR RGB</button>
    <button id="cAge" aria-pressed="false">Frame order</button></div></div>
  <div class="group"><label>Frame</label><div class="pills">
    <button id="fObj" aria-pressed="true">Object</button>
    <button id="fWorld" aria-pressed="false">World</button></div></div>
  <div class="group"><label>Box</label><div class="pills">
    <button id="tBox" aria-pressed="true">On</button></div></div>
  <div class="group"><label>Size</label>
    <input id="ptSize" type="range" min="1" max="8" step="0.5" value="3"></div>
  <button id="reset">Recenter</button>
</header>

<main>
  <canvas id="gl"></canvas>
  <div id="hud">
    <dl>
      <dt>frame</dt><dd id="hFrame">-</dd>
      <dt>points shown</dt><dd id="hShown">-</dd>
      <dt>added this frame</dt><dd id="hAdded">-</dd>
      <dt>harvested total</dt><dd id="hTotal">-</dd>
      <dt>budget</dt><dd id="hBudget">-</dd>
      <dt>trajectory</dt><dd id="hTraj">-</dd>
    </dl>
    <div id="ramp"><div class="bar"></div>
      <div class="ends"><span id="rampLo"></span><span id="rampHi"></span></div></div>
  </div>
</main>

<footer>
  <div class="transport">
    <button id="play">Play</button>
    <button id="step-">&#9664;</button>
    <button id="step+">&#9654;</button>
    <span class="frame">frame <b id="tFrame">-</b></span>
    <label class="hint">speed
      <input id="speed" type="range" min="1" max="30" step="1" value="8"></label>
    <span class="hint">drag to orbit &middot; wheel to zoom &middot; shift-drag to pan &middot; space to play
      &middot; axes: <b style="color:#e64a47">x</b> <b style="color:#1ab373">y</b> <b style="color:#3987e5">z</b></span>
  </div>
  <div id="chartwrap">
    <canvas id="chart"></canvas>
    <div id="tip"></div>
  </div>
</footer>

<script>
const DATA = __DATA__;

/* ---------- decode ---------- */
function bytes(b64){const s=atob(b64),u=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
function f32(b64){const u=bytes(b64);return new Float32Array(u.buffer,u.byteOffset,u.byteLength/4);}
function u16(b64){const u=bytes(b64);return new Uint16Array(u.buffer,u.byteOffset,u.byteLength/2);}

const F = DATA.numFrames, T0 = DATA.startTimestep;
const INS = DATA.instances.map(o=>{
  const pts=f32(o.pts), birth=u16(o.birth), poses=f32(o.poses);
  const birthF=new Float32Array(birth.length);
  for(let i=0;i<birth.length;i++)birthF[i]=birth[i];
  // world-frame copy: every point placed by the box pose of the frame it came from
  const world=new Float32Array(pts.length);
  for(let i=0;i<birth.length;i++){
    const p=birth[i]*16,x=pts[i*3],y=pts[i*3+1],z=pts[i*3+2];
    world[i*3]  =poses[p   ]*x+poses[p+1 ]*y+poses[p+2 ]*z+poses[p+3 ];
    world[i*3+1]=poses[p+4 ]*x+poses[p+5 ]*y+poses[p+6 ]*z+poses[p+7 ];
    world[i*3+2]=poses[p+8 ]*x+poses[p+9 ]*y+poses[p+10]*z+poses[p+11];
  }
  const col=bytes(o.colors), colF=new Float32Array(col.length);
  for(let i=0;i<col.length;i++)colF[i]=col[i]/255;
  return {...o, pts, world, colF, birthF, poses, vis:bytes(o.visible)};
});

/* ---------- state ---------- */
const S={ins:0, capped:true, ageColor:false, worldFrame:false, showBox:true,
         frame:0, playing:false, size:3};

// the frames that actually contribute points: play loops inside this span, while
// the timeline still scrubs the whole clip so "absent" stretches stay visible
function span(i){
  const o=INS[i===undefined?S.ins:i];
  let a=o.perFrame.findIndex(n=>n>0); if(a<0) return [0,0];
  let b=F-1; while(b>a && o.cum[b]===o.cum[b-1]) b--;
  return [a,b];
}
S.frame = span(0)[1];   // open on the finished aggregate

// optional deep link: #i=<instance idx>&f=<frame>&m=uncapped&c=age&w=1
(function(){
  const h=new URLSearchParams(location.hash.slice(1));
  if(h.has('i')) S.ins=Math.max(0,Math.min(INS.length-1,+h.get('i')));
  S.frame=span()[1];
  if(h.has('f')) S.frame=Math.max(0,Math.min(F-1,+h.get('f')-T0));
  if(h.get('m')==='uncapped') S.capped=false;
  if(h.get('c')==='age') S.ageColor=true;
  if(h.get('w')==='1') S.worldFrame=true;
})();

/* ---------- gl ---------- */
const cv=document.getElementById('gl');
const gl=cv.getContext('webgl',{antialias:true,alpha:false});
if(!gl){document.body.innerHTML='<p style="padding:20px">WebGL is not available in this browser.</p>';}

function sh(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
function prog(vs,fs){const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,vs));
  gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;}

const P_PTS=prog(`
attribute vec3 aPos; attribute vec3 aCol; attribute float aBirth;
uniform mat4 uMVP; uniform float uSize, uAge, uMaxBirth, uFrame;
varying vec3 vCol; varying float vNew;
vec3 ramp(float t){
  vec3 c0=vec3(0.106,0.227,0.420), c1=vec3(0.224,0.529,0.898);
  vec3 c2=vec3(0.620,0.788,0.969), c3=vec3(0.910,0.949,0.992);
  t=clamp(t,0.0,1.0)*3.0;
  if(t<1.0) return mix(c0,c1,t);
  if(t<2.0) return mix(c1,c2,t-1.0);
  return mix(c2,c3,t-2.0);
}
void main(){
  gl_Position = uMVP * vec4(aPos,1.0);
  vNew = abs(aBirth-uFrame) < 0.5 ? 1.0 : 0.0;
  vCol = uAge > 0.5 ? ramp(uMaxBirth>0.0 ? aBirth/uMaxBirth : 0.0) : aCol;
  gl_PointSize = clamp(uSize*(1.0+vNew*0.6)*14.0/max(gl_Position.w,0.05), 1.0, 40.0);
}`,`
precision mediump float; varying vec3 vCol; varying float vNew;
void main(){ vec2 d=gl_PointCoord-0.5; if(dot(d,d)>0.25) discard;
  gl_FragColor=vec4(mix(vCol,vec3(1.0),vNew*0.5),1.0); }`);

const P_LINE=prog(`
attribute vec3 aPos; uniform mat4 uMVP;
void main(){ gl_Position=uMVP*vec4(aPos,1.0); }`,`
precision mediump float; uniform vec4 uColor;
void main(){ gl_FragColor=uColor; }`);

const bPos=gl.createBuffer(), bCol=gl.createBuffer(), bBirth=gl.createBuffer(), bLine=gl.createBuffer();
let uploaded=-1, uploadedFrame=null;
function upload(){
  const o=INS[S.ins], key=S.ins+'|'+S.worldFrame;
  if(uploadedFrame===key) return;
  gl.bindBuffer(gl.ARRAY_BUFFER,bPos);
  gl.bufferData(gl.ARRAY_BUFFER, S.worldFrame?o.world:o.pts, gl.STATIC_DRAW);
  if(uploaded!==S.ins){
    gl.bindBuffer(gl.ARRAY_BUFFER,bCol); gl.bufferData(gl.ARRAY_BUFFER,o.colF,gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER,bBirth); gl.bufferData(gl.ARRAY_BUFFER,o.birthF,gl.STATIC_DRAW);
    uploaded=S.ins;
  }
  uploadedFrame=key;
}

/* ---------- matrices (column-major) ---------- */
function mul(a,b){const o=new Float32Array(16);
  for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;
    for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k]; o[c*4+r]=s;} return o;}
function perspective(fy,asp,n,f){const t=1/Math.tan(fy/2),nf=1/(n-f),o=new Float32Array(16);
  o[0]=t/asp;o[5]=t;o[10]=(f+n)*nf;o[11]=-1;o[14]=2*f*n*nf;return o;}
function lookAt(e,c,up){
  let z=[e[0]-c[0],e[1]-c[1],e[2]-c[2]];let l=Math.hypot(...z);z=z.map(v=>v/l);
  let x=[up[1]*z[2]-up[2]*z[1],up[2]*z[0]-up[0]*z[2],up[0]*z[1]-up[1]*z[0]];
  l=Math.hypot(...x)||1;x=x.map(v=>v/l);
  const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -(x[0]*e[0]+x[1]*e[1]+x[2]*e[2]), -(y[0]*e[0]+y[1]*e[1]+y[2]*e[2]),
    -(z[0]*e[0]+z[1]*e[1]+z[2]*e[2]), 1]);}
function fromPose(p,off){ // row-major 4x4 -> column-major mat4
  return new Float32Array([p[off],p[off+4],p[off+8],0, p[off+1],p[off+5],p[off+9],0,
    p[off+2],p[off+6],p[off+10],0, p[off+3],p[off+7],p[off+11],1]);}

const cam={theta:-0.9,phi:0.45,r:12,t:[0,0,0]};
function recenter(){
  const o=INS[S.ins];
  if(S.worldFrame){
    const p=o.poses, f=curVisibleFrame()*16;
    cam.t=[p[f+3],p[f+7],p[f+11]];
    cam.r=Math.max(...o.boxSize)*3.2;
  }else{ cam.t=[0,0,0]; cam.r=Math.max(...o.boxSize)*2.6; }
  draw();
}
function curVisibleFrame(){
  const o=INS[S.ins]; let f=S.frame;
  while(f>=0 && !o.vis[f]) f--;
  if(f<0){ f=o.vis.findIndex(v=>v); if(f<0)f=0; }
  return f;
}

/* ---------- draw ---------- */
function shownCount(){
  const o=INS[S.ins], n=o.cum[S.frame]||0;
  return S.capped ? Math.min(n, DATA.maxPts) : n;
}
function boxLines(size){
  const [a,b,c]=size.map(v=>v/2);
  const v=[[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]];
  const e=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const out=[]; for(const [i,j] of e) out.push(...v[i],...v[j]);
  return new Float32Array(out);
}
// +x / +y / +z of the object frame -- the axes the box crop tests against
function axisLines(size){
  const [a,b,c]=size.map(v=>v*0.62);   // each axis just clears its own face
  return [[new Float32Array([0,0,0, a,0,0]),[0.90,0.29,0.28,1]],
          [new Float32Array([0,0,0, 0,b,0]),[0.10,0.70,0.45,1]],
          [new Float32Array([0,0,0, 0,0,c]),[0.22,0.53,0.90,1]]];
}
function draw(){
  const dpr=Math.min(devicePixelRatio||1,2);
  const w=cv.clientWidth*dpr|0, h=cv.clientHeight*dpr|0;
  if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
  gl.viewport(0,0,w,h);
  gl.clearColor(0.07,0.07,0.065,1); gl.enable(gl.DEPTH_TEST);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);

  const o=INS[S.ins];
  const eye=[cam.t[0]+cam.r*Math.cos(cam.phi)*Math.cos(cam.theta),
             cam.t[1]+cam.r*Math.cos(cam.phi)*Math.sin(cam.theta),
             cam.t[2]+cam.r*Math.sin(cam.phi)];
  const view=lookAt(eye,cam.t,[0,0,1]);
  const proj=perspective(50*Math.PI/180, w/h||1, 0.05, Math.max(500,cam.r*20));
  const vp=mul(proj,view);

  upload();
  const n=shownCount();
  if(n>0){
    gl.useProgram(P_PTS);
    const aP=gl.getAttribLocation(P_PTS,'aPos'), aC=gl.getAttribLocation(P_PTS,'aCol'),
          aB=gl.getAttribLocation(P_PTS,'aBirth');
    gl.bindBuffer(gl.ARRAY_BUFFER,bPos); gl.enableVertexAttribArray(aP);
    gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,bCol); gl.enableVertexAttribArray(aC);
    gl.vertexAttribPointer(aC,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,bBirth); gl.enableVertexAttribArray(aB);
    gl.vertexAttribPointer(aB,1,gl.FLOAT,false,0,0);
    gl.uniformMatrix4fv(gl.getUniformLocation(P_PTS,'uMVP'),false,vp);
    gl.uniform1f(gl.getUniformLocation(P_PTS,'uSize'),S.size);
    gl.uniform1f(gl.getUniformLocation(P_PTS,'uAge'),S.ageColor?1:0);
    gl.uniform1f(gl.getUniformLocation(P_PTS,'uMaxBirth'),F-1);
    gl.uniform1f(gl.getUniformLocation(P_PTS,'uFrame'),S.frame);
    gl.drawArrays(gl.POINTS,0,n);
  }
  if(S.showBox){
    gl.useProgram(P_LINE);
    const aP=gl.getAttribLocation(P_LINE,'aPos');
    const uM=gl.getUniformLocation(P_LINE,'uMVP'), uC=gl.getUniformLocation(P_LINE,'uColor');
    const m = S.worldFrame ? mul(vp, fromPose(o.poses, curVisibleFrame()*16)) : vp;
    gl.uniformMatrix4fv(uM,false,m);
    const batches=[[boxLines(o.boxSize),[0.85,0.35,0.15,1]], ...axisLines(o.boxSize)];
    for(const [buf,col] of batches){
      gl.bindBuffer(gl.ARRAY_BUFFER,bLine); gl.bufferData(gl.ARRAY_BUFFER,buf,gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(aP); gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
      gl.uniform4f(uC,...col);
      gl.drawArrays(gl.LINES,0,buf.length/3);
    }
  }
  hud(); chart();
}

/* ---------- hud ---------- */
const $=id=>document.getElementById(id);
function fmt(n){return n.toLocaleString('en-US');}
function hud(){
  const o=INS[S.ins], shown=shownCount(), added=o.perFrame[S.frame]||0;
  $('hFrame').textContent = (S.frame+T0) + (o.vis[S.frame]?'':'  (absent)');
  $('hShown').textContent = fmt(shown);
  $('hAdded').textContent = S.capped && o.cum[S.frame]>DATA.maxPts
    ? '0  (budget full)' : '+'+fmt(added);
  $('hTotal').textContent = fmt(o.totalPoints);
  $('hBudget').textContent = o.saturatedAt==null
    ? 'never reached' : 'full at frame '+o.saturatedAt;
  $('hTraj').textContent = o.trajLength.toFixed(2)+' m';
  $('tFrame').textContent = (S.frame+T0);
  $('ramp').style.display = S.ageColor ? 'block':'none';
  $('rampLo').textContent = 'frame '+T0; $('rampHi').textContent = 'frame '+(T0+F-1);
}

/* ---------- timeline chart: cumulative harvest vs frame ---------- */
const ch=$('chart'), c2=ch.getContext('2d');
let chartGeom=null;
function chart(){
  const dpr=Math.min(devicePixelRatio||1,2);
  const w=ch.clientWidth, h=76;
  ch.width=w*dpr; ch.height=h*dpr; c2.setTransform(dpr,0,0,dpr,0,0);
  c2.clearRect(0,0,w,h);
  const o=INS[S.ins];
  const padL=46,padR=12,padT=10,padB=16;
  const iw=w-padL-padR, ih=h-padT-padB;
  // scale to what is actually drawn, so the capped curve does not collapse to a
  // sliver of the uncapped range -- the budget rule always stays on screen
  const seriesMax = S.capped ? Math.min(o.totalPoints, DATA.maxPts) : o.totalPoints;
  const yMax = Math.max(seriesMax, DATA.maxPts)*1.12;
  const X=i=>padL+(F<2?0:i/(F-1))*iw, Y=v=>padT+ih-(v/yMax)*ih;
  chartGeom={padL,iw};

  // baseline + y ticks (recessive)
  c2.strokeStyle='#33332f'; c2.lineWidth=1;
  c2.beginPath(); c2.moveTo(padL,padT+ih+.5); c2.lineTo(padL+iw,padT+ih+.5); c2.stroke();
  c2.fillStyle='#8a8a80'; c2.font='10px ui-sans-serif,system-ui,sans-serif';
  c2.textAlign='right'; c2.textBaseline='middle';
  c2.fillText('0',padL-6,padT+ih);
  // in capped mode the series max IS the budget -- let the budget label say it once
  if(seriesMax!==DATA.maxPts) c2.fillText(fmt(seriesMax),padL-6,Y(seriesMax));

  // budget rule: a threshold, not a series -> muted dashed
  if(DATA.maxPts<yMax){
    const by=Y(DATA.maxPts);
    c2.save(); c2.setLineDash([3,3]); c2.strokeStyle='#6f6f66';
    c2.beginPath(); c2.moveTo(padL,by); c2.lineTo(padL+iw,by); c2.stroke();
    c2.restore();
    c2.fillStyle='#8a8a80'; c2.textAlign='left';
    c2.fillText(fmt(DATA.maxPts)+' budget', padL+4, by<padT+12 ? by+9 : by-7);
  }

  // cumulative curve (single series, no legend needed)
  const cum=S.capped ? o.cum.map(v=>Math.min(v,DATA.maxPts)) : o.cum;
  c2.beginPath(); c2.moveTo(X(0),Y(cum[0]));
  for(let i=1;i<F;i++) c2.lineTo(X(i),Y(cum[i]));
  c2.lineTo(X(F-1),padT+ih); c2.lineTo(X(0),padT+ih); c2.closePath();
  c2.fillStyle='rgba(57,135,229,.16)'; c2.fill();
  c2.beginPath(); c2.moveTo(X(0),Y(cum[0]));
  for(let i=1;i<F;i++) c2.lineTo(X(i),Y(cum[i]));
  c2.strokeStyle='#3987e5'; c2.lineWidth=2; c2.lineJoin='round'; c2.stroke();

  // playhead + direct label of the current value
  c2.strokeStyle='#c3c2b7'; c2.lineWidth=1;
  c2.beginPath(); c2.moveTo(X(S.frame),padT-2); c2.lineTo(X(S.frame),padT+ih); c2.stroke();
  c2.fillStyle='#3987e5'; c2.beginPath();
  c2.arc(X(S.frame),Y(cum[S.frame]),3.5,0,7); c2.fill();
  c2.fillStyle='#ffffff'; c2.textAlign = S.frame>F*0.75?'right':'left';
  c2.fillText(fmt(cum[S.frame])+' pts', X(S.frame)+(S.frame>F*0.75?-8:8), Y(cum[S.frame])-9);

  // x ends
  c2.fillStyle='#8a8a80'; c2.textBaseline='top';
  c2.textAlign='left';  c2.fillText('frame '+T0, padL, padT+ih+4);
  c2.textAlign='right'; c2.fillText('frame '+(T0+F-1), padL+iw, padT+ih+4);
}

/* ---------- interaction ---------- */
function frameFromX(px){
  if(!chartGeom) return S.frame;
  const t=(px-chartGeom.padL)/chartGeom.iw;
  return Math.max(0,Math.min(F-1,Math.round(t*(F-1))));
}
let scrubbing=false;
ch.addEventListener('pointerdown',e=>{scrubbing=true;ch.setPointerCapture(e.pointerId);
  S.frame=frameFromX(e.offsetX);draw();});
ch.addEventListener('pointerup',()=>scrubbing=false);
ch.addEventListener('pointermove',e=>{
  if(scrubbing){S.frame=frameFromX(e.offsetX);draw();}
  const f=frameFromX(e.offsetX), o=INS[S.ins];
  const cum=S.capped?Math.min(o.cum[f],DATA.maxPts):o.cum[f];
  const tip=$('tip'); tip.style.display='block';
  tip.style.left=e.offsetX+'px'; tip.style.top=(e.offsetY-4)+'px';
  tip.textContent='frame '+(f+T0)+' · '+fmt(cum)+' pts · +'+fmt(o.perFrame[f]||0);
});
ch.addEventListener('pointerleave',()=>{$('tip').style.display='none';scrubbing=false;});

let drag=null;
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===1};
  cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointerup',()=>drag=null);
cv.addEventListener('pointermove',e=>{
  if(!drag)return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
  if(drag.pan){
    const s=cam.r*0.0022, ct=Math.cos(cam.theta), st=Math.sin(cam.theta);
    cam.t[0]+= s*(dx*st); cam.t[1]-= s*(dx*ct); cam.t[2]+= s*dy;
  }else{
    cam.theta-=dx*0.008;
    cam.phi=Math.max(-1.5,Math.min(1.5,cam.phi+dy*0.006));
  }
  draw();
});
cv.addEventListener('wheel',e=>{e.preventDefault();
  cam.r=Math.max(0.4,Math.min(400,cam.r*Math.exp(e.deltaY*0.0012))); draw();},{passive:false});

function toggle(id,on){$(id).setAttribute('aria-pressed',on?'true':'false');}
function bindPair(aId,bId,set){
  $(aId).onclick=()=>{set(true);toggle(aId,true);toggle(bId,false);draw();};
  $(bId).onclick=()=>{set(false);toggle(aId,false);toggle(bId,true);draw();};
}
bindPair('mCapped','mUncapped',v=>S.capped=v);
bindPair('cRgb','cAge',v=>S.ageColor=!v);
bindPair('fObj','fWorld',v=>{S.worldFrame=!v;recenter();});
$('tBox').onclick=()=>{S.showBox=!S.showBox;toggle('tBox',S.showBox);
  $('tBox').textContent=S.showBox?'On':'Off';draw();};
$('ptSize').oninput=e=>{S.size=+e.target.value;draw();};
$('reset').onclick=recenter;

const pills=$('insPills');
INS.forEach((o,i)=>{
  const b=document.createElement('button');
  b.textContent='ID '+o.id+' · '+o.className;
  b.setAttribute('aria-pressed', i===0?'true':'false');
  b.onclick=()=>{S.ins=i;
    [...pills.children].forEach((c,j)=>c.setAttribute('aria-pressed',j===i?'true':'false'));
    S.frame=span()[1];
    recenter();};
  pills.appendChild(b);
});
$('sceneTag').textContent=DATA.scene+' · '+F+' frames from '+T0;
[...pills.children].forEach((c,j)=>c.setAttribute('aria-pressed',j===S.ins?'true':'false'));
toggle('mCapped',S.capped); toggle('mUncapped',!S.capped);
toggle('cRgb',!S.ageColor); toggle('cAge',S.ageColor);
toggle('fObj',!S.worldFrame); toggle('fWorld',S.worldFrame);

let timer=null;
function setPlay(on){
  S.playing=on; $('play').textContent=on?'Pause':'Play';
  if(timer){clearInterval(timer);timer=null;}
  if(on){
    const fps=+$('speed').value;
    timer=setInterval(()=>{
      const [a,b]=span();
      S.frame = (S.frame>=b || S.frame<a) ? a : S.frame+1;
      draw();
    },1000/fps);
  }
}
$('play').onclick=()=>setPlay(!S.playing);
$('speed').oninput=()=>{ if(S.playing) setPlay(true); };
$('step-').onclick=()=>{S.frame=(S.frame-1+F)%F;draw();};
$('step+').onclick=()=>{S.frame=(S.frame+1)%F;draw();};
addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();setPlay(!S.playing);}
  if(e.code==='ArrowLeft'){S.frame=(S.frame-1+F)%F;draw();}
  if(e.code==='ArrowRight'){S.frame=(S.frame+1)%F;draw();}
});
addEventListener('resize',draw);
recenter();
</script>
</body>
</html>
"""
