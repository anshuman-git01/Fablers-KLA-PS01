Background
In semiconductor manufacturing, microscopic inspection images are used to measure and verify chip quality at every stage of production. These images must be extremely sharp and clean because a single pixel of noise or a small loss of detail can hide a defect that causes a chip to fail.
In practice, inspection images are often degraded by two types of signal loss:
Speckle Noise — random pixel-level noise that makes the image look ‘grainy.’ This noise can actually push pixel values beyond the true image range, meaning some pixels appear brighter or darker than they actually are in reality.
Spatial Resolution Reduction — the image has been ‘shrunk’ (downsampled), losing fine detail. A 512×512 pixel ground truth image becomes a blurry 256×256 pixel image, or a 256×256 becomes 128×128. Details that were visible at full resolution are gone.
Currently, engineers work with these degraded images and make do with the noise and lost detail. AI-powered restoration can recover information that appears lost removing noise while sharpening detail back to the original resolution.
You will receive a training dataset of paired images: for each sample, you get a degraded image (noisy \+ low resolution) and the corresponding ground truth image (clean \+ full resolution). Your job is to train an AI model that learns to reverse the degradation taking a bad image as input and producing a restored image that matches the ground truth as closely as possible.

**The degradation types your model must handle:**

| Degradation Type | What It Looks Like | What Your Model Must Fix |
| :---- | :---- | :---- |
| Speckle Noise | Random pixel-level noise, image looks grainy. Some pixel values are pushed beyond the true range of the image. | Remove the grain while preserving the real image details underneath. Do not blur the image to remove noise, that destroys useful information. |
| Gaussian Noise | Image appears soft and hazy edges and fine structures lose sharpness. | Restore edge sharpness and contrast without introducing artificial patterns or ringing. |
| Spatial Resolution Reduction (Super-Resolution) | Image has been downsampled: 512×512 → 256×256, or 256×256 → 128×128. Fine details are lost. | Upscale the image back to the original resolution (512×512 or 256×256) while reconstructing the fine details that were lost during downsampling. |

 

* Your model must handle ALL degradation types simultaneously a single image may have speckle noise AND reduced resolution at the same time.  
* The test set will include images from different sources than the training data (out-of-distribution). Your model must generalize not just memorize the training examples.  
* Speed matters. Your model will be benchmarked on inference time. A model that produces great results but takes 10 minutes per image is less useful than one that produces good results in 10 seconds.

**Training Data — What You Get**

**KLA will provide a paired training dataset. For each sample, you receive:**

| What You Get | Resolution | Description |
| :---- | :---- | :---- |
| Ground Truth Image (clean, full resolution) | 512×512 pixels or 256×256 pixels | The ‘correct answer’ — the image as it should look. High resolution, high signal-to-noise ratio. This is what your model’s output should match. |
| Degraded Image (noisy, low resolution) | 256×256 pixels or 128×128 pixels | The input to your model — noisy and downsampled. Your model takes this as input and must produce an output that matches the ground truth. |

**Important data notes:**

* The degraded image intensity range may EXCEED the ground truth range this is expected behaviour caused by speckle noise pushing pixel values beyond the original signal. Your model must handle this.  
* The images come from diverse data origins different types of semiconductor structures. Your model should generalize across these variations, not overfit to one type.  
* Images are grayscale (single channel). Colour images are NOT part of this challenge.

**Test Data — What Comes Later**  
After the training phase, KLA will release a test dataset. The test set contains:

* In-distribution samples: images similar to what you trained on. Tests accuracy.  
* Out-of-distribution samples: images from different sources than the training data. Tests generalization and robustness  whether your model can handle image types it has never seen.

