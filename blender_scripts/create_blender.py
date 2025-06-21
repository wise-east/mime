# blender script to create a video from a mocap and avatar 
# this is not meant to be run as a script directly (e.g. python create_blender.py), but rather as a script to be run in blender 

import bpy
from pathlib import Path
from typing import Any, Optional
import math 
import time 
# define constants
PACKAGE_DIR = "/project/jonmay_1426/hjcho/mime"

# if blender version is 4.3
if bpy.app.version[0] == 4 and bpy.app.version[1] == 3:
    bpy.ops.preferences.addon_enable(module="Cats-Blender-Plugin-Unofficial--blender-43")
    bpy.ops.preferences.addon_enable(module="rokoko-studio-live-blender-master")

# define functions
def reset_blend_file():
    # Change to object mode if needed
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # select everything delete them 
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear all data blocks
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
        
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
        
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
        
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)
        
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
        
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)
        
    for image in bpy.data.images:
        bpy.data.images.remove(image)

def import_and_rename_armature(filepath, new_name):
    # Import the FBX file
    bpy.ops.import_scene.fbx(filepath=filepath, automatic_bone_orientation=True)

    # Find the imported armature
    imported_armature = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            imported_armature = obj
            break
    
    # Rename the armature
    if imported_armature:
        imported_armature.name = new_name
        print(f"Armature renamed to: {new_name}")
    else:
        print("No armature found in the imported file.")

def adjust_mocap_armature(armature): 
    
    armature.select_set(True)

    # start pose mode with CATS add on 
    bpy.ops.cats_manual.start_pose_mode()

    # rotate by 180 degrees on z-axis 
    bpy.ops.transform.rotate(value=3.14159, orient_axis='Z', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)


    # From the poll function, armature_obj must already be in pose mode, but it's possible it might not be the
    # active object e.g., the user has multiple armatures opened in pose mode, but a different armature is currently
    # active. We can use an operator override to tell the operator to treat armature_obj as if it's the active
    # object even if it's not, skipping the need to actually set armature_obj as the active object.
    op_override(bpy.ops.pose.armature_apply, {'active_object': armature})
    bpy.ops.object.posemode_toggle()

def select_and_activate_bone(bone_name):

    # Ensure we're in Pose Mode
    if bpy.context.active_object.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')
        
    # Deselect all bones
    bpy.ops.pose.select_all(action='DESELECT')
        
    # Select the specific bone
    bpy.context.object.data.bones[bone_name].select = True  
    
    # set the specific bone to be active bone 
    bpy.context.object.data.bones.active = bpy.context.object.data.bones[bone_name] 

def rotate_camera_z(angle_degrees):
    camera = bpy.data.objects["Camera"]
    
    # Store initial values
    initial_location = camera.location.copy()
    initial_rotation = camera.rotation_euler.copy()
    
    # Calculate initial radius in XY plane
    radius = math.sqrt(initial_location.x**2 + initial_location.y**2)
    
    # Calculate initial angle in XY plane
    initial_angle = math.atan2(initial_location.y, initial_location.x)
    
    # Calculate new angle (initial + rotation)
    angle_rad = math.radians(angle_degrees) + initial_angle
    
    # Set new position while keeping original z-height
    camera.location.x = radius * math.cos(angle_rad)
    camera.location.y = radius * math.sin(angle_rad)
    
    # Rotate the camera by the same angle around Z
    camera.rotation_euler.z = initial_rotation.z + math.radians(angle_degrees)  

def op_override(operator, context_override: dict[str, Any], context: Optional[bpy.types.Context] = None,
                execution_context: Optional[str] = None,
                undo: Optional[bool] = None, **operator_args) -> set[str]:
    """Call an operator with a context override"""
    args = []
    if execution_context is not None:
        args.append(execution_context)
    if undo is not None:
        args.append(undo)

    if context is None:
        context = bpy.context
    with context.temp_override(**context_override):
        return operator(*args, **operator_args)

# Empty the blend file
reset_blend_file()
# settings for debugging 
# bpy.context.space_data.show_word_wrap = True

def create_blender_file(mocap_path, avatar, angle:int=0, start_frame:int=None, end_frame:int=None, force=False):
    
    # set rendering configurations 
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.max_bounces = 1
    bpy.context.scene.cycles.caustics_reflective = False
    bpy.context.scene.cycles.caustics_refractive = False
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720 
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.cycles.use_light_tree = False
    bpy.context.scene.cycles.adaptive_threshold = 0.5
    bpy.context.scene.cycles.preview_adaptive_threshold = 0.5
    bpy.context.scene.cycles.denoising_use_gpu = True
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.render.film_transparent = True

    # set fps to 24 
    bpy.context.scene.render.fps = 60

    # set to default world
    bpy.context.scene.world = bpy.data.worlds["World"]
    frame = 150

    avatar_path = f"{PACKAGE_DIR}/data/mixamo_avatars/{avatar}.fbx"
    if avatar == "man": 
        mixamorig_name = "mixamorig4"
    elif avatar == "woman": 
        mixamorig_name = "mixamorig2"
    else: 
        mixamorig_name = "mixamorig"

    action_name = mocap_path.split("/")[-1].replace(".fbx", "")
    actor_name = "j" if "justin" in mocap_path else "n"
    blend_file_path = f"{PACKAGE_DIR}/data/blend_files/{avatar}_{actor_name}_{action_name}_angle{angle}.blend"
    if Path(blend_file_path).exists() and not force: 
        print(f"Blend file already exists: {blend_file_path}")
        return True

    # import motion capture data 
    import_and_rename_armature(mocap_path, "mocap_armature")
    # select the mocap armature using name
    mocap_armature = bpy.data.objects["mocap_armature"]

    adjust_mocap_armature(mocap_armature)

    # load the avatar 
    import_and_rename_armature(avatar_path, "avatar_armature")
    avatar_armature = bpy.data.objects["avatar_armature"]

    bpy.ops.cats_manual.start_pose_mode()
    bpy.context.view_layer.objects.active = bpy.data.objects["avatar_armature"]
    #bpy.ops.pose.select_all(action='DESELECT')
    #bpy.context.scene.armature = 'avatar_armature'

    select_and_activate_bone(f"{mixamorig_name}:RightHandThumb1") 
    bpy.ops.transform.rotate(value=0.45903, orient_axis='X', orient_type='LOCAL')

    select_and_activate_bone(f"{mixamorig_name}:LeftHandThumb1") 
    bpy.ops.transform.rotate(value=0.45903, orient_axis='X', orient_type='LOCAL')

    # adjust avatar's arms if avatar is basic
    if avatar == "man" or avatar == "woman": 
        select_and_activate_bone(f"{mixamorig_name}:RightShoulder") 
        bpy.ops.transform.rotate(value=0.32733, orient_axis='X', orient_type='LOCAL')

        select_and_activate_bone(f"{mixamorig_name}:LeftShoulder") 
        bpy.ops.transform.rotate(value=0.32733, orient_axis='X', orient_type='LOCAL')

        select_and_activate_bone(f"{mixamorig_name}:RightArm") 
        bpy.ops.transform.rotate(value=-0.320752, orient_axis='X', orient_type='LOCAL')

        select_and_activate_bone(f"{mixamorig_name}:LeftArm") 
        bpy.ops.transform.rotate(value=-0.320752, orient_axis='X', orient_type='LOCAL')

    op_override(bpy.ops.pose.armature_apply, {'active_object': mocap_armature}) # why does this work with mocap_armature but not with avatar_armature? 
    bpy.ops.object.posemode_toggle()

    # start retargeting with rokoko studio live add on 
    bpy.context.scene.rsl_retargeting_armature_source = bpy.data.objects["mocap_armature"]
    bpy.context.scene.rsl_retargeting_armature_target = bpy.data.objects["avatar_armature"]

    bpy.ops.rsl.build_bone_list()
    
    # remove spine 3 if it goes in wrong 
    if len(bpy.context.scene.rsl_retargeting_bone_list) > 5: 
        bpy.context.scene.rsl_retargeting_bone_list[4].bone_name_target = ""
    # if avatar == "man" or avatar == "woman":
    
    bpy.context.scene.rsl_retargeting_bone_list[2].bone_name_target = f"{mixamorig_name}:Spine1"
    bpy.context.scene.rsl_retargeting_bone_list[3].bone_name_target = f"{mixamorig_name}:Spine2"
    bpy.context.scene.rsl_retargeting_bone_list[4].bone_name_target = ""
    bpy.context.scene.rsl_retargeting_bone_list[10].bone_name_target = f"{mixamorig_name}:RightArm"
    bpy.context.scene.rsl_retargeting_bone_list[37].bone_name_target = f"{mixamorig_name}:LeftArm"
    bpy.context.scene.rsl_retargeting_bone_list[53].bone_name_target = f"{mixamorig_name}:LeftHandPinky4"
    bpy.context.scene.rsl_retargeting_bone_list[43].bone_name_target = f"{mixamorig_name}:LeftHandMiddle4"
    bpy.context.scene.rsl_retargeting_bone_list[48].bone_name_target = f"{mixamorig_name}:LeftHandRing4"
    bpy.context.scene.rsl_retargeting_bone_list[58].bone_name_target = f"{mixamorig_name}:LeftHandIndex4"

    bpy.context.scene.rsl_retargeting_bone_list[35].bone_name_target = f"{mixamorig_name}:RightHandThumb4"
    bpy.context.scene.rsl_retargeting_bone_list[31].bone_name_target = f"{mixamorig_name}:RightHandIndex4"
    bpy.context.scene.rsl_retargeting_bone_list[26].bone_name_target = f"{mixamorig_name}:RightHandPinky4"
    bpy.context.scene.rsl_retargeting_bone_list[21].bone_name_target = f"{mixamorig_name}:RightHandRing4"
    bpy.context.scene.rsl_retargeting_bone_list[16].bone_name_target = f"{mixamorig_name}:RightHandMiddle4"
 
    bpy.ops.rsl.retarget_animation()

    # add plane 
    bpy.ops.mesh.primitive_plane_add(size=50, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.object.is_shadow_catcher = True

    # add light 
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 3), scale=(1, 1, 1))
    bpy.context.object.data.cycles.max_bounces = 1
    bpy.context.object.data.energy = 5

    # add camera 
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(10.809, -9.5732, 5.1592), rotation=(1.3, 0, 0.8), scale=(1, 1, 1))
    # set as active camera 
    bpy.context.scene.camera = bpy.data.objects["Camera"]

    # find frame end value for mocap data 
    mocap_data = bpy.data.objects["mocap_armature"]
    frame_start = int(mocap_data.animation_data.action.frame_range[0]) if start_frame is None else start_frame
    frame_end = int(mocap_data.animation_data.action.frame_range[1]) if end_frame is None else end_frame
    frame_step = 1 
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    bpy.context.scene.frame_step = frame_step
    print(f"Frame start and end value for mocap data: {frame_start} - {frame_end}")


    # create a render sample and save it for debugging purposes
    # for angle in [45 for _ in range(8)]:
    #     rotate_camera_z(angle)
    #     bpy.context.scene.frame_current = frame 
    #     bpy.context.scene.render.filepath = f"{PACKAGE_DIR}/data/render_samples/{avatar}_angle{angle}_frame{frame}.png"
    #     bpy.ops.render.render(write_still=True, animation=False)

    # save blend file 
    rotate_camera_z(angle)
    bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)
    
    return False
    
    
# load all mocap data (mocap_j & mocap_n) 
mocap_data_filenames = list(Path(f"{PACKAGE_DIR}/data/mocap_j").glob("*.fbx")) + list(Path(f"{PACKAGE_DIR}/data/mocap_n").glob("*.fbx"))
# sort by name 
mocap_data_filenames.sort()

filtered_mocap_info_fn = f"{PACKAGE_DIR}/data/filtered_mocap_info.csv"
# read csv file without using pandas 
with open(filtered_mocap_info_fn, "r") as f: 
    lines = f.readlines()
    mocap_info = [line.split(",") for line in lines[1:]]
# first column is the mocap path, third column is the start frame, fourth column is the end frame, and fifth column is whether to use this data
mocap_info_dict = {
    f"{PACKAGE_DIR}/data/{row[0]}": {
        "start_frame": int(row[2]),
        "end_frame": int(row[3]),
        "use": row[5] == "TRUE"
    } 
    for row in mocap_info
}

# check that all mocap paths exist
for mocap_path in mocap_info_dict:  
    if not Path(mocap_path).exists(): 
        print(f"Mocap path does not exist: {mocap_path}")
        continue 


avatars = ["man", "spacesuit", "woman"]

# save all relevant assets for blend file
bpy.ops.file.autopack_toggle()

test = False

reset_blend_file()

for angle in range(0, 360, 90): 
    for avatar in avatars: 
        # only create other angles for man avatar 
        if angle != 0 and avatar != "man": 
            continue 
        
        for mocap_path in mocap_data_filenames: 
            mocap_path = str(mocap_path)
            if mocap_info_dict[mocap_path]["use"] == False: 
                continue 
            
            try: 
                already_created = create_blender_file(str(mocap_path), avatar, angle, start_frame=mocap_info_dict[mocap_path]["start_frame"], end_frame=mocap_info_dict[mocap_path]["end_frame"], force=False)
                
                if already_created: 
                    continue 
            except Exception as e: 
                print(f"Error creating blender file for {mocap_path} and {avatar}: {e}")
                continue 
            
            time.sleep(5)
            reset_blend_file()
            
            if test: 
                breakpoint()
            


