"""Numerical five-joint IK against official-model FK; no controller access."""
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from approach_pose import rigid


def solve_position(wrist,target_mm,seed,limits,tcp_link5_mm=None,tolerance_mm=1.):
    """Position-only hover. Tool orientation is explicitly unconstrained."""
    target=np.asarray(target_mm,float);q=np.asarray(seed,float).copy();bounds=np.asarray(limits,float)
    offset=np.zeros(3) if tcp_link5_mm is None else np.asarray(tcp_link5_mm,float)
    if (target.shape!=(3,) or offset.shape!=(3,) or q.shape!=(5,) or bounds.shape!=(5,2)
            or not all(np.isfinite(x).all() for x in (target,offset,q,bounds))
            or np.any(bounds[:,0]>=bounds[:,1]) or np.any(q<bounds[:,0]) or np.any(q>bounds[:,1])):
        raise ValueError('Invalid target, TCP offset, start angles or joint limits')
    def fk(angles):
        r,p=wrist.fk(angles);return r@offset+p
    for iteration in range(250):
        position=fk(q);error=target-position
        if np.linalg.norm(error)<=tolerance_mm:
            return dict(joints_rad=q.tolist(),position_residual_mm=float(np.linalg.norm(error)),
                        reference_base_mm=position.tolist(),orientation_constrained=False,
                        collision_checked=False,motion_enabled=False)
        jac=np.column_stack([(fk(q+np.eye(5)[i]*1e-5)-position)/1e-5 for i in range(5)])
        delta=jac.T@np.linalg.solve(jac@jac.T+np.eye(3)*.1,error)
        delta*=min(1,.1/max(float(np.max(np.abs(delta))),1e-9))
        for scale in (1,.5,.25,.1):
            trial=np.clip(q+scale*delta,bounds[:,0],bounds[:,1])
            if np.linalg.norm(target-fk(trial))<np.linalg.norm(error):q=trial;break
        else:break
    raise ValueError(f'No position IK: residual {np.linalg.norm(target-fk(q)):.2f} mm')


def raw_limits(wrist,vendor):
    path=Path(vendor)/'src/roarm_main/roarm_description/urdf/roarm_m3/roarm_m3.xacro'
    joints={j.find('child').attrib['link']:j for j in ET.parse(path).getroot().findall('joint')}
    bounds=[]
    for i in range(5):
        limit=joints[f'link{i+1}'].find('limit')
        model=np.array([float(limit.attrib['lower']),float(limit.attrib['upper'])])
        bounds.append(np.sort((model-wrist.offsets[i])/wrist.signs[i]))
    return np.asarray(bounds)


def downward_tilt_deg(wrist,q):
    # Official link5 -> hand_tcp translation is along link5 +Z.
    axis=wrist.fk(q)[0][:,2]
    return float(np.degrees(np.arccos(np.clip(-axis[2],-1,1))))


def solve_grasp_posture(wrist,target_mm,seed,limits,tcp_link5_mm,
                        closing_axis_base=None,closing_axis_local=None):
    """Score reachable postures by joint travel, limit margin and grasp alignment.

    No forced vertical orientation. A closing axis is an unoriented line:
    either finger order is accepted. This is a local candidate search, not a
    global or collision-aware optimum.
    """
    seed=np.asarray(seed,float);bounds=np.asarray(limits,float)
    target=np.asarray(target_mm,float);offset=np.asarray(tcp_link5_mm,float)
    aligned=closing_axis_base is not None and closing_axis_local is not None
    if aligned:
        desired=np.asarray(closing_axis_base,float);desired/=np.linalg.norm(desired)
        local=np.asarray(closing_axis_local,float);local/=np.linalg.norm(local)
    starts=[seed]
    for pitch in (-.35,.35):
        trial=seed.copy();trial[3]+=pitch;starts.append(np.clip(trial,bounds[:,0],bounds[:,1]))
    candidates=[]
    for start in starts:
        try:solution=solve_position(wrist,target,start,bounds,offset,tolerance_mm=.5)
        except ValueError:continue
        q=np.array(solution['joints_rad'])
        variants=[q]
        if aligned:
            # Try both equivalent finger orders; select by required joint travel.
            for sign in (-1,1):
                trial=q.copy()
                def error(angles):
                    r,p=wrist.fk(angles)
                    return np.r_[p+r@offset-target,60*(r@local-sign*desired),.3*(angles-seed)]
                for _ in range(160):
                    e=error(trial)
                    jac=np.column_stack([(error(trial+np.eye(5)[i]*1e-5)-e)/1e-5 for i in range(5)])
                    delta=-jac.T@np.linalg.solve(jac@jac.T+np.eye(len(e))*.1,e)
                    delta*=min(1.,.12/max(float(np.max(np.abs(delta))),1e-9))
                    for scale in (1.,.5,.25,.1):
                        step=np.clip(trial+delta*scale,bounds[:,0],bounds[:,1])
                        if np.linalg.norm(error(step))<np.linalg.norm(e)-1e-8:
                            trial=step;break
                    else:break
                try:variants.append(np.array(solve_position(wrist,target,trial,bounds,offset,tolerance_mm=.5)['joints_rad']))
                except ValueError:pass
        for goal in variants:
            r,p=wrist.fk(goal);position=p+r@offset
            angle=float(np.degrees(np.arccos(np.clip(abs((r@local)@desired),0,1)))) if aligned else None
            travel=float(np.linalg.norm(goal-seed))
            margin=float(np.min(np.minimum(goal-bounds[:,0],bounds[:,1]-goal)))
            # Penalize pointing upward; do not prefer exact downward vertical.
            upward=max(0.,float(r[2,2]))
            score=travel+.02/max(.02,margin)+10*upward+(0 if angle is None else 6*np.radians(angle))
            candidates.append((score,goal,float(np.linalg.norm(position-target)),angle,travel,margin))
    if not candidates:raise ValueError('No reachable TCP posture found')
    score,goal,residual,angle,travel,margin=min(candidates,key=lambda item:item[0])
    return dict(joints_rad=goal.tolist(),position_residual_mm=residual,
                downward_tilt_deg=downward_tilt_deg(wrist,goal),closing_axis_error_deg=angle,
                candidate_count=len(candidates),posture_score=score,joint_travel_deg=float(np.degrees(travel)),
                joint_limit_margin_deg=float(np.degrees(margin)),orientation_constrained=False,
                orientation_preference='joint travel + limit margin + grasp-axis alignment; no forced vertical',
                collision_checked=False)


def solve_downward(wrist,target_mm,seed,limits,tcp_link5_mm=None):
    """Position first, prefer link5 +Z down; tool roll remains unconstrained.

    Finite directional weights allow tilt when the five-joint arm cannot reach
    the requested position with an exactly vertical tool. No TCP is assumed.
    """
    target=np.asarray(target_mm,float);seed=np.asarray(seed,float);bounds=np.asarray(limits,float)
    offset=np.zeros(3) if tcp_link5_mm is None else np.asarray(tcp_link5_mm,float)
    positional=solve_position(wrist,target,seed,bounds,offset)
    candidates=[np.array(positional['joints_rad'])]
    for weight in (100.,40.,15.):
        q=candidates[-1].copy()
        def error(angles):
            rotation,position=wrist.fk(angles)
            return np.r_[position+rotation@offset-target,weight*(rotation[:,2]-[0,0,-1])]
        for _ in range(180):
            e=error(q)
            jac=np.column_stack([(error(q+np.eye(5)[i]*1e-5)-e)/1e-5 for i in range(5)])
            delta=-jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*.1,e)
            delta*=min(1.,.12/max(float(np.max(np.abs(delta))),1e-9))
            improved=False
            for scale in (1.,.5,.25,.1):
                trial=np.clip(q+scale*delta,bounds[:,0],bounds[:,1])
                if np.linalg.norm(error(trial))<np.linalg.norm(e)-1e-8:
                    q=trial;improved=True;break
            if not improved:break
        # Restore position accuracy after optimizing the directional preference.
        try:
            fixed=solve_position(wrist,target,q,bounds,offset,tolerance_mm=.5)
            candidates.append(np.array(fixed['joints_rad']))
        except ValueError:continue
    goal=min(candidates,key=lambda q:(downward_tilt_deg(wrist,q),float(np.linalg.norm(q-seed))))
    rotation,position=wrist.fk(goal);position=position+rotation@offset
    return dict(joints_rad=goal.tolist(),position_residual_mm=float(np.linalg.norm(position-target)),
                downward_tilt_deg=downward_tilt_deg(wrist,goal),orientation_constrained=False,
                orientation_preference='link5 +Z toward base -Z; soft preference',collision_checked=False)


def solve(wrist,target,seed,limits,position_tolerance_mm=1.,angle_tolerance_deg=2.,iterations=200):
    target=rigid(target);seed=np.asarray(seed,float);bounds=np.asarray(limits,float)
    if (seed.shape!=(5,) or bounds.shape!=(5,2) or not np.isfinite(seed).all()
            or not np.isfinite(bounds).all() or np.any(bounds[:,0]>=bounds[:,1])):
        raise ValueError('Expected five finite joint angles and increasing limits')
    if np.any(seed<bounds[:,0]) or np.any(seed>bounds[:,1]):raise ValueError('Measured start is outside model joint limits')
    def error(q):
        r,p=wrist.fk(q)
        angle=cv2.Rodrigues(target[:3,:3]@r.T)[0].ravel()
        return np.r_[target[:3,3]-p,80*angle]
    best=None
    rng=np.random.default_rng(42)
    starts=[seed]+[np.clip(seed+rng.normal(0,.3,5),bounds[:,0],bounds[:,1]) for _ in range(5)]
    for start in starts:
        q=start.copy()
        for iteration in range(iterations):
            e=error(q);position=float(np.linalg.norm(e[:3]));angle=float(np.degrees(np.linalg.norm(e[3:])/80))
            if best is None or np.linalg.norm(e)<best[0]:best=(np.linalg.norm(e),position,angle)
            if position<=position_tolerance_mm and angle<=angle_tolerance_deg:
                return dict(joints_rad=q.tolist(),position_residual_mm=position,angle_residual_deg=angle,
                            iterations=iteration,collision_checked=False,physical_joint_mapping_verified=False,
                            motion_enabled=False,scope='Model IK only; does not authorize execution')
            epsilon=1e-5
            jac=np.column_stack([(error(q+np.eye(5)[i]*epsilon)-e)/epsilon for i in range(5)])
            delta=-jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*.1,e)
            delta*=min(1,.15/max(np.max(np.abs(delta)),1e-9))
            # Backtracking avoids accepting an iteration which worsens the pose.
            for scale in (1,.5,.25,.1):
                trial=np.clip(q+delta*scale,bounds[:,0],bounds[:,1])
                if np.linalg.norm(error(trial))<np.linalg.norm(e):q=trial;break
            else:break
    raise ValueError(f'No five-joint IK within tolerances; best position={best[1]:.2f}mm angle={best[2]:.2f}deg')
