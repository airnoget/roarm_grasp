const fs=require('fs'),path=require('path'),assert=require('assert');
const api=require('./control_protocol.js');
assert.equal(api.makeStep(6,Math.PI,1).joint,6);assert.throws(()=>api.makeStep(7,0,1));assert.throws(()=>api.makeStep(1,0,3));assert.throws(()=>api.makeStep(1,0,1,500));
// Exhaustive direction checks for all five joints and every jog speed.
for(const j of api.joints)for(const mode of Object.values(api.modes)){
 for(const [current,sign] of [[j.min,1],[j.max,-1],[j.min-.12,1],[j.max+.12,-1],[j.min+.05,1],[j.max-.05,-1]]){
  const c=api.makeStep(j.id,current*Math.PI/180,sign*mode.step,mode.spd),target=c.rad*180/Math.PI;
  assert(target>=j.min-1e-8&&target<=j.max+1e-8);assert(Math.abs(target-current)<=2+1e-8);assert.equal(Math.sign(target-current),sign);
  const step=Math.round(c.rad/(2*Math.PI)*4096);
  const ticks=j.id===1||j.id===5?[2048-step]:j.id===2?[2048+step,2048-step]:j.id===3?[1024+step]:j.id===6?[step]:[2048+step];
  assert(ticks.every(x=>x>=0&&x<=4095),`${j.name}: invalid raw servo tick ${ticks}`);
  if(j.id===3)assert(ticks[0]>=1024&&ticks[0]<=3071);
 }
 assert.throws(()=>api.makeStep(j.id,j.min*Math.PI/180,-mode.step,mode.spd),e=>e.code==='limit_lower');
 assert.throws(()=>api.makeStep(j.id,j.max*Math.PI/180,mode.step,mode.spd),e=>e.code==='limit_upper');
 assert.throws(()=>api.makeStep(j.id,(j.max+3)*Math.PI/180,-mode.step,mode.spd),e=>e.code==='feedback_range');
}
const elbow=api.joints.find(j=>j.id===3);
assert.equal(elbow.max,(3071-1024)*360/4096);
for(const deg of [179.8,elbow.max,180,180.15,181.5]){
 const c=api.makeStep(3,deg*Math.PI/180,-.25,40);assert(c.rad*180/Math.PI<deg);
 const tick=Math.round(c.rad/(2*Math.PI)*4096)+1024;assert(tick>=1024&&tick<=3071);
}
// Stale-target regression: feedback has crossed the previous goal.
let directionCases=0;
for(const j of api.joints)for(const mode of Object.keys(api.modes))for(let d=Math.ceil(j.min+3);d<j.max-3;d+=3)for(const offset of [-3,-1,0,1,3])for(const sign of [-1,1]){
 const measured=d*Math.PI/180,previous=(d+offset)*Math.PI/180;
 try{const c=api.makeJog(j.id,measured,previous,sign,mode);assert(sign*(c.rad-measured)>0);directionCases++;}
 catch(e){if(!['lead_limit','limit_lower','limit_upper'].includes(e.code))throw e;}
}
console.log('PASS: signed target invariant for '+directionCases+' angle/speed/lag cases');
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE);
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 try{
  const context=await browser.newContext({viewport:{width:390,height:1000},hasTouch:true});
  await context.addInitScript(()=>{
   window.__requests=[];window.__follow=false;window.__auto=true;window.__defer=false;
   window.__frame={T:-15,b:0,s:0,e:Math.PI/2,t:0,r:0,g:Math.PI,v:1235,servo_ok:[true,true,true,true,true,true,true],Stalltor:false,Stalltep:false,Stallvol:false};
   window.fetch=async url=>{const c=JSON.parse(decodeURIComponent(url.split('json=')[1]));window.__requests.push(c);if(window.__follow)window.__frame[['','b','s','e','t','r','g'][c.joint]]=c.rad;if(window.__defer)await new Promise(r=>window.__resolve=r);return{ok:true,json:async()=>({ok:1})};};
   window.WebSocket=class{constructor(){this.readyState=1;window.__ws=this;setTimeout(()=>this.onopen&&this.onopen(),0);setInterval(()=>{if(window.__auto&&this.readyState===1)this.emit(window.__frame);},100);}emit(data){if(this.onmessage)this.onmessage({data:JSON.stringify(data)});}};
  });
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('file:///'+path.join(__dirname,'index.preview.html').replaceAll('\\','/'));
  await page.waitForTimeout(150);await page.locator('#unlock').click();
  const plus=page.locator('#joints .row').nth(0).locator('button').nth(1);
  const press=async()=>{const b=await plus.boundingBox();await page.mouse.move(b.x+b.width/2,b.y+b.height/2);await page.mouse.down();};
  const count=()=>page.evaluate(()=>__requests.length);
  const noMore=async()=>{const n=await count();await page.waitForTimeout(450);assert.equal(await count(),n);};
  // Physical pointer held down: repeated bounded commands even without target-arrival feedback.
  await press();await page.waitForTimeout(950);assert(await count()>=3);assert(await plus.isEnabled());await page.mouse.up();await noMore();
  assert(await page.evaluate(()=>hold===null&&holdTimer===null));
  // Fast mode moves through many steps with live feedback, not a single tiny click.
  await page.evaluate(()=>{__follow=true;});await page.locator('#speed').selectOption('fast');
  const before=await count();await press();await page.waitForTimeout(950);await page.mouse.up();
  assert(await count()-before>=4);const cmds=await page.evaluate(()=>__requests.slice(-4));assert(cmds.every(c=>c.spd===160&&c.joint===1));await noMore();
  // Moving the captured finger/mouse outside the button cancels repetition.
  await press();await page.waitForTimeout(300);await page.mouse.move(1,1);await noMore();await page.mouse.up();
  // Touch cancel and capture loss both stop.
  for(const name of ['pointercancel','lostpointercapture']){
   await press();await page.waitForTimeout(300);
   await page.evaluate(eventName=>{const target=hold.button,id=hold.pointerId;target.dispatchEvent(new PointerEvent(eventName,{pointerId:id,bubbles:true}));},name);
   await noMore();await page.mouse.up();
  }
  // Release while the HTTP response is delayed: no later auto-resume after ACK.
  await page.evaluate(()=>{__defer=true;});await press();await page.waitForTimeout(100);await page.mouse.up();
  const waiting=await count();await page.evaluate(()=>{__defer=false;__resolve();});await noMore();assert.equal(await count(),waiting);
  // Focus loss, stale feedback and disconnect must stop an active hold.
  await press();await page.waitForTimeout(100);await page.evaluate(()=>window.dispatchEvent(new Event('blur')));await noMore();await page.mouse.up();
  await press();await page.waitForTimeout(100);await page.evaluate(()=>{__auto=false;stamp=performance.now()-4000;});await noMore();await page.mouse.up();
  await page.evaluate(()=>{__auto=true;});await noMore();
  // Simulate lag/overshoot in the UI: '-' must never target above measured.
  for(const mode of ['fine','normal','fast'])for(const sign of [-1,1]){
   await page.locator('#speed').selectOption(mode);
   await page.evaluate(sign=>{__frame.s=0;__ws.emit(__frame);goals.set(2,-sign*3*Math.PI/180);nextClick=0;draw();},sign);
   const c=await page.evaluate(sign=>candidate(api.joints[1],sign),sign);
   assert(sign*c.rad>0,`wrong shoulder direction at ${mode}`);
  }
  // Real DOM button states and inward pointer commands at each joint's limits.
  for(const j of api.joints)for(const mode of ['fine','normal','fast'])for(const upper of [false,true]){
   await page.locator('#speed').selectOption(mode);
   const deg=upper?j.max+.12:j.min-.12;
   await page.evaluate(({key,deg})=>{__frame[key]=deg*Math.PI/180;__ws.emit(__frame);goals.clear();nextClick=0;draw();},{key:j.key,deg});
   const row=page.locator('#joints .row').nth(j.id-1),inward=row.locator('button').nth(upper?0:1),outward=row.locator('button').nth(upper?1:0);
   assert(await inward.isEnabled(),`${j.name} ${mode} inward disabled`);assert.equal(await outward.isEnabled(),false);
   assert((await row.locator('.reason').innerText()).includes('限'));
   await inward.scrollIntoViewIfNeeded();const box=await inward.boundingBox();
   const prior=await count();await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();
   await page.waitForFunction(n=>__requests.length>n,prior);await page.mouse.up();
   const cmd=await page.evaluate(()=>__requests[__requests.length-1]);assert.equal(cmd.joint,j.id);
   assert(upper?cmd.rad<deg*Math.PI/180:cmd.rad>deg*Math.PI/180);
  }
  await page.evaluate(()=>{Object.assign(__frame,{b:0,s:0,e:Math.PI/2,t:0,r:0});__ws.emit(__frame);nextClick=0;draw();});
  await plus.scrollIntoViewIfNeeded();
  // Keyboard press/release also repeats and stops.
  await plus.focus();await page.keyboard.down('Space');await page.waitForTimeout(500);await page.keyboard.up('Space');await noMore();
  fs.mkdirSync(path.join(__dirname,'qa'),{recursive:true});await page.screenshot({path:path.join(__dirname,'qa/mobile.png'),fullPage:true});
  await page.setViewportSize({width:900,height:1050});await page.screenshot({path:path.join(__dirname,'qa/desktop.png'),fullPage:true});
  await page.evaluate(()=>{__frame.servo_ok[3]=false;});await page.waitForTimeout(150);
  assert.equal(await page.locator('#joints .row').nth(2).locator('button:enabled').count(),0);
  await press();await page.waitForTimeout(100);await page.evaluate(()=>{__ws.readyState=3;__ws.onclose();});await noMore();await page.mouse.up();
  assert.equal(await page.locator('#joints button:enabled').count(),0);assert.deepEqual(errors,[]);
  console.log('PASS: all five joints/all speed limit recovery and routing, exact elbow tick bounds, real long press repeat, bounded no-arrival motion, fast mode, pointer release/outside/cancel/capture loss, delayed ACK, blur, stale recovery without resume, keyboard release, missing servo and disconnect');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
