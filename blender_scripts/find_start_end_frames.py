import bpy
scene = bpy.context.scene
print("%d,%d" % (scene.frame_start, scene.frame_end))