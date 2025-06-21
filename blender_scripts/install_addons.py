import os
import zipfile
import requests
import bpy
import sys
import traceback

def install_local_addon(name, zip_path, module_name):
    """Install addon from local zip file"""
    try:
        print(f"Starting {name} addon installation...")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        
        addon_dir = os.path.join(bpy.utils.user_resource('SCRIPTS'), 'addons')
        print(f"Addon directory: {addon_dir}")
        
        # Create addons directory if it doesn't exist
        os.makedirs(addon_dir, exist_ok=True)
        
        # Extract the addon
        try:
            print(f"Extracting {name} addon from {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(addon_dir)
        except zipfile.BadZipFile as e:
            print(f"Error extracting zip: {e}")
            return False
        
        # Enable the addon
        print(f"Enabling {name} addon...")
        try:
            # Set addon support levels before enabling
            if hasattr(bpy.context.window_manager, "addon_support"):
                bpy.context.window_manager.addon_support = {'OFFICIAL', 'COMMUNITY'}
            
            bpy.ops.preferences.addon_enable(module=module_name)
        except Exception as e:
            print(f"Error enabling addon: {e}")
            return False
            
        print(f"{name} addon installation complete!")
        return True
        
    except Exception as e:
        print(f"Unexpected error installing {name}:")
        print(traceback.format_exc())
        return False

def install_addon(name, url, module_name):
    """Generic function to install Blender addons"""
    try:
        print(f"Starting {name} addon installation...")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        
        addon_dir = os.path.join(bpy.utils.user_resource('SCRIPTS'), 'addons')
        zip_path = os.path.join(addon_dir, f"{name.lower()}.zip")
        
        print(f"Addon directory: {addon_dir}")
        print(f"Zip path: {zip_path}")
        
        # Create addons directory if it doesn't exist
        os.makedirs(addon_dir, exist_ok=True)
        
        # Download the addon with error handling
        print(f"Downloading {name} addon...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {name}: {e}")
            return False
            
        # Save the file
        try:
            with open(zip_path, 'wb') as f:
                f.write(response.content)
        except IOError as e:
            print(f"Error saving zip file: {e}")
            return False
            
        # Extract the addon
        try:
            print(f"Extracting {name} addon...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(addon_dir)
        except zipfile.BadZipFile as e:
            print(f"Error extracting zip: {e}")
            return False
        finally:
            # Try to remove the zip file
            try:
                os.remove(zip_path)
            except OSError:
                pass
        
        # Enable the addon
        print(f"Enabling {name} addon...")
        try:
            bpy.ops.preferences.addon_enable(module=module_name)
        except Exception as e:
            print(f"Error enabling addon: {e}")
            return False
            
        print(f"{name} addon installation complete!")
        return True
        
    except Exception as e:
        print(f"Unexpected error installing {name}:")
        print(traceback.format_exc())
        return False

def install_rokoko_addon():
    return install_addon(
        "Rokoko",
        "https://github.com/Rokoko/rokoko-studio-live-blender/archive/refs/heads/master.zip",
        "rokoko-studio-live-blender-master"
    )

def download_cats_addon():
    # download the cats addon from https://mime-understanding.s3.com/CatsBlenderPlugin4.3Latest.zip
    url = "https://mime-understanding.s3.com/CatsBlenderPlugin4.3Latest.zip"
    response = requests.get(url)
    with open("blender_scripts/CatsBlenderPlugin4.3Latest.zip", "wb") as f:
        f.write(response.content)
    
def install_cats_addon():
    local_zip = "blender_scripts/CatsBlenderPlugin4.3Latest.zip"
    return install_local_addon(
        "CATS",
        local_zip,
        "Cats-Blender-Plugin-Unofficial--blender-43"  # Updated module name to match the actual folder name
    )

if __name__ == "__main__":
    try:
        print("Starting addon installation script...")
        print(f"Blender version: {bpy.app.version_string}")
        
        if not bpy.app.background:
            print("Warning: Not running in background mode")
            
        if not os.path.exists("blender_scripts/CatsBlenderPlugin4.3Latest.zip"):
            download_cats_addon()
            
        success = install_rokoko_addon() and install_cats_addon()
        
        if success:
            print("All addons installed successfully")
        else:
            print("Some addons failed to install")
            sys.exit(1)
            
    except Exception as e:
        print("Critical error in main script:")
        print(traceback.format_exc())
        sys.exit(1)
    finally:
        bpy.ops.wm.quit_blender()