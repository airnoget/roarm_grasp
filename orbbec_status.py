"""Read-only device settings; unsupported values remain explicitly unknown."""

PROPERTIES = [
    ('OB_PROP_DEPTH_ROTATE_INT','int'),('OB_PROP_DEPTH_MIRROR_BOOL','bool'),('OB_PROP_DEPTH_FLIP_BOOL','bool'),
    ('OB_PROP_DISP_SEARCH_RANGE_MODE_INT','int'),('OB_PROP_DISP_SEARCH_OFFSET_INT','int'),
    ('OB_PROP_CURRENT_DISP_SEARCH_RANGE_MODE_INT','int'),('OB_PROP_CURRENT_DISP_SEARCH_OFFSET_INT','int'),
    ('OB_PROP_LASER_BOOL','bool'),('OB_PROP_LASER_CONTROL_INT','int'),
    ('OB_PROP_LASER_POWER_ACTUAL_LEVEL_INT','int'),
    ('OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL','bool'),('OB_PROP_IR_AUTO_EXPOSURE_BOOL','bool'),
    ('OB_PROP_DEPTH_EXPOSURE_INT','int'),('OB_PROP_IR_EXPOSURE_INT','int'),
    ('OB_PROP_DEPTH_GAIN_INT','int'),('OB_PROP_IR_GAIN_INT','int'),
    ('OB_PROP_MIN_DEPTH_INT','int'),('OB_PROP_MAX_DEPTH_INT','int'),
    ('OB_PROP_DEPTH_PRECISION_LEVEL_INT','int'),
    ('OB_PROP_LASER_HIGH_TEMPERATURE_PROTECT_BOOL','bool'),
    ('OB_PROP_LASER_OVERCURRENT_PROTECTION_STATUS_BOOL','bool'),
]


def snapshot(device, sdk):
    result=dict(properties={},info={},settings_modified=False)
    def read(fn):
        try:return dict(status='read',value=fn())
        except Exception as exc:return dict(status='unavailable',reason=str(exc))
    info=device.get_device_info()
    for name in ('get_name','get_firmware_version','get_connection_type','get_hardware_version'):
        result['info'][name]=read(lambda name=name:str(getattr(info,name)()))
    result['depth_mode']=read(lambda:str(device.get_current_depth_mode_name()))
    for name,kind in PROPERTIES:
        prop=getattr(sdk.OBPropertyID,name,None)
        if prop is None:
            result['properties'][name]=dict(status='unsupported_by_sdk');continue
        def get(prop=prop,kind=kind):
            if not device.is_property_supported(prop,sdk.OBPermissionType.PERMISSION_READ):
                raise ValueError('Property not readable on this device')
            return getattr(device,f'get_{kind}_property')(prop)
        result['properties'][name]=read(get)
    return result
