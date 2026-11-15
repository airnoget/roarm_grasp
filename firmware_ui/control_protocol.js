/* Bounded single-joint commands for bench commissioning. No broadcast commands. */
(function(root){
  'use strict';
  const joints=[
    {id:1,key:'b',name:'底座',servo:'11',min:-179.9,max:179.9},
    {id:2,key:'s',name:'肩关节',servo:'12 + 13',min:-90,max:90},
    {id:3,key:'e',name:'肘关节',servo:'14',min:0,max:(3071-1024)*360/4096},
    {id:4,key:'t',name:'腕部俯仰',servo:'15',min:-90,max:90},
    {id:5,key:'r',name:'腕部旋转',servo:'16',min:180-4095*360/4096,max:180},
    {id:6,key:'g',name:'夹爪',servo:'17',min:700*360/4096,max:3396*360/4096}
  ];
  const modes={fine:{step:.25,spd:40,interval:250},normal:{step:1.25,spd:100,interval:250},fast:{step:2,spd:160,interval:200}};
  function makeStep(id, radians, deltaDegrees, speed=80){
    const joint=joints.find(j=>j.id===id);
    if(!joint || !Number.isFinite(radians) || !Number.isFinite(deltaDegrees) ||
       Math.abs(deltaDegrees)>2 || deltaDegrees===0 || !Number.isInteger(speed) || speed<20 || speed>160) throw Error('无效的关节或步长');
    const current=radians*180/Math.PI, epsilon=1e-7;
    const fail=(code,message)=>{const error=Error(message);error.code=code;throw error;};
    const label=x=>x.toFixed(2).replace(/\.?0+$/,'');
    // Only the outward direction is blocked at a boundary. A measured overshoot
    // <=2 degrees can return to the boundary; never command a large clamp jump.
    let target=current+deltaDegrees;
    if(current<joint.min-epsilon || current>joint.max+epsilon){
      const lower=current<joint.min, edge=lower?joint.min:joint.max;
      if(lower?deltaDegrees<0:deltaDegrees>0)fail(lower?'limit_lower':'limit_upper',`已到${lower?'下':'上'}限 ${label(edge)}°，请按${lower?'＋':'−'}返回`);
      if(Math.abs(edge-current)>2+epsilon)fail('feedback_range',`实测角 ${label(current)}° 超出范围，请核对舵机零点/反馈`);
      target=edge;
    }else{
      if(deltaDegrees<0&&current<=joint.min+epsilon)fail('limit_lower',`已到下限 ${label(joint.min)}°，请按＋返回`);
      if(deltaDegrees>0&&current>=joint.max-epsilon)fail('limit_upper',`已到上限 ${label(joint.max)}°，请按−返回`);
      target=Math.max(joint.min,Math.min(joint.max,target));
    }
    return {T:101,joint:id,rad:target*Math.PI/180,spd:speed,acc:5};
  }
  function makeJog(id, measured, previousTarget, sign, modeName){
    const joint=joints.find(j=>j.id===id), mode=modes[modeName];
    if(!joint||!mode||!Number.isFinite(measured)||(sign!==1&&sign!==-1))throw Error('无效调姿输入');
    const deg=measured*180/Math.PI, previous=Number.isFinite(previousTarget)?previousTarget:measured;
    // Never let a stale goal put a new command on the wrong side of feedback.
    const origin=deg<joint.min||deg>joint.max?measured:
      sign>0?Math.max(measured,previous):Math.min(measured,previous);
    const cmd=makeStep(id,origin,sign*mode.step,mode.spd);
    if(sign*(cmd.rad-measured)<=1e-10)throw Error('方向校验失败，未发送');
    const leadDegrees=(id===2||id===3)?8:4;
    if(Math.abs(cmd.rad-measured)>(leadDegrees+.01)*Math.PI/180){const e=Error(`目标已领先反馈${leadDegrees}°，实测未跟随；请松手查看舵机状态`);e.code='lead_limit';throw e;}
    return cmd;
  }
  const api={joints,modes,makeStep,makeJog};
  if(typeof module!=='undefined')module.exports=api;
  root.RoArmControl=api;
})(typeof globalThis!=='undefined'?globalThis:this);
