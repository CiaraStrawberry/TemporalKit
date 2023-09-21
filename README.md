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

Example results you can get: <br>
https://user-images.githubusercontent.com/13116982/234425054-9a1bbf30-93a8-4f5b-9e80-4376ab3c510a.mp4

---

The values in the extension are as follows
---

| Variable            | Description |
|---------------------| --- |
| `FPS`               | The fps the video is extracted and produced at. |
| `batch_Size`        | This is the number of frames between each keyframe, so for example if you had an fps of 30, and a batch size of 10, it would make 3 keyframes a second and estimate the rest. |
| `per side`          | This is the square root of the number of frames per plate, so for example a per side value of 2 would make 4 plates, 3, 9 plates, 4 16 plates. |
| `Resolution`        | The size of each plate, it is strongly reccomended you set this to a multiple of your per side variable |
| `batch settings`    | Only open this drop down if you want to generate a folder of plates. |
| `Max Frames`        | When generating a folder of plates, this gets how many frames at the above fps you want to get, and then divides them into plates in groups of (per side * per side * batch size) |
| `Border Frames`     | Every batch generated plate will contain this many frames from the next plate and blend between them. |
| `Batch Folder`      | If you're generating a batch of plates, just specify a empty folder and on clicking run, it will populate it with the relevant folders and files, all you need to do is go to img2img batch processing in  original sd, enter the newly create input folder as the input, the newly created output folder as the output, generate, move back to the temporal-kit Batch-Warp Tab, put in the whole folder directory and click read and it will set everything up. |
| `Output Resolution` | The maximum resolution on any side of the output video. |

---

FAQ:
---

> **Q:** My video has smearing.
> 
> **A:** Use a higher fps and/or lower batchnumber, the closer together the keyframes the less artifacts.

---

Step-by-Step Tutorial
---

### Step 1: Install FFMPEG


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
