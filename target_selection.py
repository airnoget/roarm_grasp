"""Explicit selection retains the complete scene for obstacle reasoning."""
def selected_proposal(result):
    proposals=result.get('proposals',[])
    identity=result.get('selected_target_id')
    if identity is not None:
        matches=[p for p in proposals if p.get('target_id')==identity]
        if len(matches)!=1:raise ValueError('Selected target has no unique current depth fit')
        return matches[0]
    if len(result.get('native_detections',[]))==1 and len(proposals)==1:return proposals[0]
    raise ValueError('Select a confirmed target with a current depth fit')
