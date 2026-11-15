"""Reversible explicit 1280x800 near-range disparity configuration."""
def configure(device,sdk):
    changes=[]
    try:
        for name,value in [('OB_PROP_DISP_SEARCH_RANGE_MODE_INT',2),('OB_PROP_DISP_SEARCH_OFFSET_INT',127)]:
            prop=getattr(sdk.OBPropertyID,name)
            if not device.is_property_supported(prop,sdk.OBPermissionType.PERMISSION_READ_WRITE):
                raise ValueError(f'Near mode requires readable/writable {name}')
            old=device.get_int_property(prop)
            changes.append((prop,old))
            device.set_int_property(prop,value)
            if device.get_int_property(prop)!=value:raise ValueError(f'Near mode readback failed: {name}')
    except Exception:
        for prop,old in reversed(changes):device.set_int_property(prop,old)
        raise
    def restore():
        for prop,old in reversed(changes):device.set_int_property(prop,old)
    return restore
