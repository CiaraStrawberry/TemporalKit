import launch

if not launch.is_installed("ffmpeg"):
    launch.run_pip("install ffmpeg-python", "Install \"ffmpeg-python\" requirements for TemporalKit extension")

if not launch.is_installed("moviepy"):
    launch.run_pip("install moviepy", "Install \"moviepy\" requirements for TemporalKit extension")
    
if not launch.is_installed("imageio_ffmpeg"):
    launch.run_pip("install imageio_ffmpeg", "Install \"imageio_ffmpeg\" requirements for TemporalKit extension")

if not launch.is_installed("scenedetect"):
    launch.run_pip("install scenedetect", "Install \"scenedetect\" requirements for TemporalKit extension")
