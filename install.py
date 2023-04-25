import launch

#todo: check if ffmpeg-python is used anywhere, could have sworn i removed it but not 100% enough to change now.
if not launch.is_installed("ffmpeg-python"):
    launch.run_pip("install ffmpeg-python", "requirements for TemporalKit extension")

if not launch.is_installed("moviepy"):
    launch.run_pip("install moviepy", "requirements for TemporalKit extension")
    
if not launch.is_installed("imageio_ffmpeg"):
    launch.run_pip("install imageio_ffmpeg", "requirements for TemporalKit extension")

if not launch.is_installed("scenedetect"):
    launch.run_pip("install scenedetect", "requirements for TemporalKit extension")