# Copyright © rogangriffin
# Thx & refer to https://gist.github.com/rogangriffin/450756d17c253a417b48e8a08bf3e703

class EBSynthKeyFrame:
    useFirstFrame = False
    useLastFrame = False
    firstFrame = 0
    keyFrame = 0
    lastFrame = 0
    outputPath = ""

    def __init__(self, inuseFirstFrame=False, inuseLastFrame=False, infirstFrame=0, inkeyFrame=0, inlastFrame=0, inoutputPath=""):
        self.useFirstFrame = inuseFirstFrame
        self.useLastFrame = inuseLastFrame
        self.firstFrame = infirstFrame
        self.keyFrame = inkeyFrame
        self.lastFrame = inlastFrame
        self.outputPath = inoutputPath

    def GetByteArray(self):
        byteArray = bytearray()
        #Write middle keyframe number
        byteArray.extend(self.keyFrame.to_bytes(4, byteorder = 'little'))
        #Write if we're using the first keyframe
        if self.useFirstFrame:
            byteArray.append(1)
        else:
            byteArray.append(0)
        #Write if we're using the last keyframe
        if self.useLastFrame:
            byteArray.append(1)
        else:
            byteArray.append(0)
        #Write first keyframe number
        byteArray.extend(self.firstFrame.to_bytes(4, byteorder = 'little'))
        #Write last keyframe number
        byteArray.extend(self.lastFrame.to_bytes(4, byteorder = 'little'))
        #Write out path utf-8 character count
        outputPathLength = len(self.outputPath).to_bytes(4, byteorder = 'little')
        byteArray.extend(outputPathLength)
        #Write video path
        outputPathBA = bytearray(self.outputPath, "utf-8")
        byteArray.extend(outputPathBA)
        return byteArray

class EBSynthProject:
    keyFrames = []
    videoPath = ""
    keyframesPath = ""
    maskPath = ""
    maskOn = False

    def __init__(self, inVideoPath="", inKeyframesPath="", inMaskPath="", inMaskOn=False):
        self.videoPath = inVideoPath
        self.keyframesPath = inKeyframesPath
        self.maskPath = inMaskPath
        self.maskOn = inMaskOn

    def WriteToFile(self, filePath):
        #Write header
        byteArray = bytearray(b'\x45\x42\x53\x30\x35\x00')

        #Write video path utf-8 character count
        videoPathLength = len(self.videoPath).to_bytes(4, byteorder = 'little')
        byteArray.extend(videoPathLength)
        #Write video path
        videoPathBA = bytearray(self.videoPath, "utf-8")
        byteArray.extend(videoPathBA)

        #Write mask path utf-8 character count
        maskPathLength = len(self.maskPath).to_bytes(4, byteorder = 'little')
        byteArray.extend(maskPathLength)
        #Write mask path
        maskPathBA = bytearray(self.maskPath, "utf-8")
        byteArray.extend(maskPathBA)

        #Write keys path utf-8 character count
        keysPathLength = len(self.keyframesPath).to_bytes(4, byteorder = 'little')
        byteArray.extend(keysPathLength)

        #Write keyframes path
        keysPathBA = bytearray(self.keyframesPath, "utf-8")
        byteArray.extend(keysPathBA)

        #Mask on/off
        if self.maskOn:
            byteArray.append(1)
        else:
            byteArray.append(0)

        unknownBA = bytearray(b'\x00\x00\x80\x3F\x00\x00\x80\x40\x00\x00\x80\x3F\x00\x00\x20\x41\x00\x00\x80\x3F\x00\xC0\x5A\x45')
        byteArray.extend(unknownBA)
        #Write keyframe count
        keyframeCount = len(self.keyFrames).to_bytes(4, byteorder = 'little')
        byteArray.extend(keyframeCount)

        #Write keyframes
        for keyFrame in self.keyFrames:
            keyframeBA = keyFrame.GetByteArray()
            byteArray.extend(keyframeBA)

        #End byte array
        endBA = bytearray(b'\x56\x30\x32\x00\x02\x00\x00\x00\x01\x00\x00\xF0\x41\xC0\x02\x00\x00')
        byteArray.extend(endBA)

        newFile = open(filePath, "wb")
        newFile.write(byteArray)
        newFile.close()

    def AddKeyFrame(self, useFirstFrame=False, useLastFrame=False, firstFrame=0, keyFrame=0, lastFrame=0, outputPath=""):
        keyFrame = EBSynthKeyFrame(useFirstFrame, useLastFrame, firstFrame, keyFrame, lastFrame, outputPath)
        self.keyFrames.append(keyFrame)

# Usage
# project = EBSynthProject(videoPath, keysPath, maskPath, useMask)
# project.AddKeyFrame(useFirstFrame, useLastFrame, firstFrame, middleFrame, lastFrame, outputPath)
# project.WriteToFile(outpathEBSFilePath)

# project = EBSynthProject("../video/[####].png", "../keys/[####].png", "../mask/[#####].png", False)
# project.AddKeyFrame(False, True, 1, 1, 5, "../out0001/[####].png")
# project.AddKeyFrame(True, True, 1, 5, 10, "../out0005/[####].png")
# project.AddKeyFrame(True, True, 5, 10, 15, "../out0010/[####].png")
# project.AddKeyFrame(True, False, 10, 15, 20, "../out0015/[####].png")
# project.WriteToFile("generated.ebs")