import bpy 
import math
import sys 

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
    


# get the angle from the command line
angle_degrees = float(sys.argv[-1])
rotate_camera_z(angle_degrees)

# render animation 
bpy.ops.render.render(animation=True)
