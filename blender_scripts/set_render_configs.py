import bpy 

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