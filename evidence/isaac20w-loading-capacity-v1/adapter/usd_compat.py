"""USD 4.5 compatibility for source-scene xform helpers."""


def patch_scene_transforms(scene_module):
    from pxr import Gf, UsdGeom

    def get_or_add(xform, op_type, add):
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == op_type:
                return op
        return add()

    def set_xform(prim, translate=None, rotate_xyz=None, scale=None):
        xform = UsdGeom.Xformable(prim)
        if translate is not None:
            op = get_or_add(xform, UsdGeom.XformOp.TypeTranslate, xform.AddTranslateOp)
            op.Set(Gf.Vec3d(*(float(value) for value in translate)))
        if rotate_xyz is not None:
            op = get_or_add(xform, UsdGeom.XformOp.TypeRotateXYZ, xform.AddRotateXYZOp)
            op.Set(Gf.Vec3f(*(float(value) for value in rotate_xyz)))
        if scale is not None:
            op = get_or_add(xform, UsdGeom.XformOp.TypeScale, xform.AddScaleOp)
            op.Set(Gf.Vec3f(*(float(value) for value in scale)))

    def set_root_pose(root_prim, position, state):
        set_xform(root_prim, translate=position,
                  rotate_xyz=(state.roll_deg, state.pitch_deg, state.yaw_deg))

    def set_root_scale(root_prim, scale):
        set_xform(root_prim, scale=(scale, scale, scale))

    def spin_rotors(rotor_prims, time_s):
        angle = (float(time_s) * 900.0) % 360.0
        for index, prim in enumerate(rotor_prims):
            name = prim.GetName()
            if "QuadRotor" in name:
                direction = 1.0 if index % 2 == 0 else -1.0
                rotation = (0.0, 0.0, angle * direction + index * 45.0)
            elif "TailRotor" in name or index == 2:
                rotation = (0.0, 90.0, angle)
            else:
                rotation = (0.0, 0.0, angle + index * 90.0)
            set_xform(prim, rotate_xyz=rotation)

    scene_module._set_xform = set_xform
    scene_module._set_root_pose = set_root_pose
    scene_module._set_root_scale = set_root_scale
    scene_module._spin_rotors = spin_rotors
