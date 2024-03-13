TemporalKit
===
An all in one solution for adding Temporal Stability to a Stable Diffusion Render via an automatic1111 extension

---

***You must install FFMPEG to path before running this***

You can find a demonstration run through with a single batch here: <br>
https://twitter.com/CiaraRowles1/status/1645923461343363072

And a batch demonstration here: <br>
https://mobile.twitter.com/CiaraRowles1/status/1646458056803250178

Ebsynth tutorial: <br>
https://twitter.com/CiaraRowles1/status/1648462374125576192 <br>
**NOTE: EBSYNTH DOES NOT REGISTER THE KEYFRAMES IF YOU USE ABOVE 20.**

Ebsynth split frames tutorial: <br>
https://www.youtube.com/watch?v=z3YNHiuvxyg&ab_channel=CiaraRowles

Example results you can get:

https://user-images.githubusercontent.com/13116982/234425054-9a1bbf30-93a8-4f5b-9e80-4376ab3c510a.mp4

<br><br>

---

Supported version
---

| Stable Diffusion version | Support level                       |
|--------------------------|-------------------------------------|
| 1.7.0                    | Support v1.2                        |
| 1.6.x                    | Support v1.2                        |
| <1.6.0                   | Not Support, may encounter problems |

---

The values in the extension are as follows
---

| Variable            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `FPS`               | The fps the video is extracted and produced at.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `batch_Size`        | This is the number of frames between each keyframe, so for example if you had an fps of 30, and a batch size of 10, it would make 3 keyframes a second and estimate the rest.                                                                                                                                                                                                                                                                                    |
| `per side`          | This is the square root of the number of frames per plate, so for example a per side value of 2 would make 4 plates, 3, 9 plates, 4 16 plates.                                                                                                                                                                                                                                                                                                                   |
| `Resolution`        | The size of each plate, it is strongly reccomended you set this to a multiple of your per side variable                                                                                                                                                                                                                                                                                                                                                          |
| `batch settings`    | Only open this drop down if you want to generate a folder of plates.                                                                                                                                                                                                                                                                                                                                                                                             |
| `Max Frames`        | When generating a folder of plates, this gets how many frames at the above fps you want to get, and then divides them into plates in groups of (per side * per side * batch size)                                                                                                                                                                                                                                                                                |
| `Border Frames`     | Every batch generated plate will contain this many frames from the next plate and blend between them.                                                                                                                                                                                                                                                                                                                                                            |
| `Batch Folder`      | If you're generating a batch of plates, just specify a empty folder and on clicking run, it will populate it with the relevant folders and files, all you need to do is go to img2img batch processing in  original sd, enter the newly create input folder as the input, the newly created output folder as the output, generate, move back to the temporal-kit Batch-Warp Tab, put in the whole folder directory and click read and it will set everything up. |
| `Output Resolution` | The maximum resolution on any side of the output video.                                                                                                                                                                                                                                                                                                                                                                                                          |

<br><br>

---

FAQ:
---

> **Q:** My video has smearing.
> 
> **A:** Use a higher fps and/or lower batchnumber, the closer together the keyframes the less artifacts.

> **Q:** Stable diffusion cannot be turned on after installing this Extension. `ModuleNotFoundError: No module named 'tqdm.auto'`
> 
> **A:** Because the dependency currently used by this plug-in uses an old version of tqdm, an error occurs. The short-term solution is to manually install a new version(4.66.1) of tqdm. <br>
> ```bash
> pip install tqdm==4.66.1
> ```
> [More information](https://github.com/CiaraStrawberry/TemporalKit/issues/104#issuecomment-1722527970)

<br><br>

---

Step-by-Step Tutorial
---
Written with reference to this [web page](https://stable-diffusion-art.com/video-to-video/#Method_5_Temporal_Kit) teaching and my [own(cocomine)](https://github.com/cocomine/) experience



### Step 1: Install Extensions on WebUI

Open `Extensions` tab > `Install from URL` > Paste follow link in `URL for extension’s git repository` > Click `Install`

```
https://github.com/cocomine/TemporalKit
```

### Step 2: Install FFMPEG

#### Ubuntu

```bash
sudo apt install ffmpeg
```

#### Arch Linux
```bash
sudo pacman -S ffmpeg
```

#### Windows
Download and install from https://ffmpeg.org/download.html <br>
Make sure to add ffmpeg to your PATH. <br>
Learn more: https://www.wikihow.com/Install-FFmpeg-on-Windows

### Step 4: Prepare your video
Create a folder in your desired location. This folder will be used to store the files that need to be processed.
Prepare the video you need to use (referred to as the original video in subsequent teaching), and understand the format of the original video, such as resolution and frame.

### Step 5: Extract frames from the original video
1. Open `Temporal-Kit` Tab on Top.
2. Open `Pre-Process` Tab.
2. Drag & Drop the original video into the `Input Video`.
3. Set `Video FPS` to the frame rate of the original video. 
4. Set `Frames per keyframe` to the number of frames between each keyframe. For example, if the original video is 30fps and you set it to 10, then 3 keyframes will be generated per second, and the rest will be estimated.
5. Set `Side` to the square root of the number of frames per plate. For example, if you set it to 2, 4 plates(2x2) will be generated, 3, 9 plates(3x3), 4, 16 plates(4x4).
6. Set `Height Resolution` to the size of each plate. It is strongly recommended that you set this to a multiple of your side variable. <br> For example, if you want to generate 4 plates and set the side to 2, each plate high 512, then you need to set the height resolution to 1024(512x2).
7. Set `Target Folder` to the folder you created in [step 4](#step-4-prepare-your-video).
8. Tick the `Batch Run`.
9. Tick the `Split Video`

When you complete the above steps you should see a structure similar to this in the folder you specified (depending on the length of your video)

![folder structure](/readme_img/1.png)

You will see a structure like this in the video clips divided into folders named by numbers.

![folder structure](/readme_img/2.png)

> If you encounter out of memory issue in the next **img2img** step, reduce the `side` or `Height Resolution` parameters.

### Step 6: Perform Img2img on keyframes
Go to the **Img2img** page. Switch to the **Batch** tab. Set the following parameters:

**Input directory**: The name of your [target directory](#step-4-prepare-your-video) with `\input` appended. E.g. `YOUR_FOLDER_PATH_IN_SETP_4\input` <br>
**Output directory**: Similarly but with `\output` appended. E.g. `YOUR_FOLDER_PATH_IN_SETP_4\output`

Enter a **prompt** and a **negative prompt** like txt2img. <br>
**Sampling method:** DPM++2M Karras <br>
**Sampling steps:** 20 <br>
**CFG scale:** 7 <br>
**Denoising strength:** 0.5 (adjust accordingly) <br>
> The above parameters can be changed as needed.


### Step 6.1: Control Net (option, which would give better results if available)
In ControlNet (Unit 0) section, set:
+ Enable: Yes
+ Pixel Perfect: Yes
+ ControlType: Tile
+ Preprocessor: tile_resample
+ Model: control_xxxx_tile

Press **Generate**. After it is done, you will find the image in the batch output folder.

> Make sure to open the image in full size and inspect the details in full size. Make sure they look sharp and have a consistent style.

> If you want to obtain high-resolution images, please put the output images back into img2img, and adjust resizd by `1.5-2.0`, Denoising strength `0.3-0.4`, and then generate.

### Step 7: Prepare EbSynth data
Go to `Temporal-Kit` page and switch to the `Ebsynth-Process` tab.

**Input Folder:** Put in the same [target folder](#step-4-prepare-your-video) path you put in the Pre-Processing page.

Click `Read Setting`. If your input folder is correct, the video and the settings will be populated.

Click prepare ebsynth. After it is done, you should see the keys folder populated with your stylized keyframes, and the 
frames folder populated with your images. _(In a folder named with a number)_

![folder structure](/readme_img/3.png)
![folder structure](/readme_img/4.png)

> Please note that this program does not generate `.ebs` files. When your images are imported into the program, they will be automatically populated.

### Step 8: Process with EbSynth
Now open the **EbSynth** program.

Open the File Explorer and navigate to the folder your creation in [step4](#step-4-prepare-your-video). You should folder 
like the ones showed below. Then open the `keys.ebs` file. _(In a folder named with a number)_

![folder structure](/readme_img/5.png)
![ebs](/readme_img/6.png)

Click **Run All** and wait for them to complete.
When it is done, you should see a series of `out_#####` directories generated in the [target project folder](#step-4-prepare-your-video).

Then repeat the above steps in other folders named with numbers.

> Please download the program from the official website. <br>
> https://ebsynth.com/

### Step 9: Generate the final video
Now go back to **AUTOMATIC1111**(webUI). You should still be on the **Temporal Kit** page and **Ebsynth-Process** tab.

Click **recombine ebsynth** and you are done!

![](https://stable-diffusion-art.com/wp-content/uploads/2023/06/temporalkit_ebsynth_ds0.5.gif)

Look how smooth the video is. With some tweaking, you can probably make it better!

---

Updates Log
---
### 1.2
- Add progress display on recombine ebsynth

### 1.1
- fix UI is not displayed
- fix problem with image sorting
- Fixed tqdm reinstall version 4.66.1
- fix recombine video

### 1.0
- Improved interface layout [cocomine contributed]
- Add progress display (Some pages) [cocomine contributed]
- Generate ebsynth files [cocomine contributed]

---

TODO
---

- set up diffusion based upscaling for the plates output 
- get the img2img button working with batch processing.
- add a check to see if the output folder was added.
- fix that weird shutdown error it gives after running
- hook up to the api.
- flowmaps from game engine export\import support

_Thanks to RAFT for the optical flow system._
