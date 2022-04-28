# lasfunc

A set of functions for vapoursynth.

## Usage

### Encode a clip using ffmpeg

```python
output.ffv1(clip:vs.VideoNode, file_str:str,
    mux:str='nut', executable:str='ffmpeg') -> None
# YUV411P8, YUV410P8, YUV420P8, YUV422P8, YUV440P8, YUV444P8, YUV420P9,
# YUV422P9, YUV444P9, YUV420P10, YUV422P10, YUV444P10, YUV420P12, YUV422P12,
# YUV444P12, YUV420P14, YUV422P14, YUV444P14, YUV420P16, YUV422P16, YUV444P16,
# Gray8, Gray9, Gray10, Gray12, Gray16, RGB27, RGB30, RGB36, RGB42, RGB48.

output.libx264(clip:vs.VideoNode, file_str:str, preset:str='veryslow', crf:int=7, 
        crf_max:Optional[int]=None, gop:Optional[int]=None, threads:int=3, 
        rc_lookahead:Optional[int]=None, mux:str='nut', executable:str='ffmpeg') -> None
# Lossless
# YUV420P8, YUV422P8, YUV444P8, YUV420P10, YUV422P10, YUV444P10.

output.llx264(clip:vs.VideoNode, file_str:str, preset:str='veryslow',
    mux:str='nut', executable:str='ffmpeg') -> None
# YUV420P8, YUV422P8, YUV444P8, YUV420P10, YUV422P10, YUV444P10.

output.libsvtav1(clip:vs.VideoNode, file_str:str, speed:int=6, quantizer:int=32,
        gop:Optional[int]=None, lad:Optional[int]=None,
        tiles_row:Optional[int]=None, tiles_col:Optional[int]=None,
        sc_detection:bool=False, threads:int=4, mux:str='nut', executable:str='ffmpeg') -> None
# YUV420P8, YUV420P10

output.libaom(clip:vs.VideoNode, file_str:str, speed:int=4, quantizer:int=32,
    gop:Optional[int]=None, lif:Optional[int]=None,
    tiles_row:Optional[int]=None, tiles_col:Optional[int]=None,
    enable_cdef:bool=True, enable_restoration:Optional[bool]=None, 
    enable_chroma_deltaq:bool=True, arnr_strength:int=2,
    arnr_max_frames:int=5, threads:int=4, mux:str='nut', executable='ffmpeg') -> None
# YUV420P8, YUV422P8, YUV444P8, YUV420P10, YUV422P10, YUV444P10, YUV420P12,
# YUV422P12, YUV444P12, Gray8, Gray10, Gray12, RGB24, RGB30, RGB36.
```

Container is not dependent on filename extension.
Default container is nut.
For mkv use "matroska"

### Encode a clip using a specific encoder

```python
output.aomenc(clip:vs.VideoNode, file_str:str, mux:str="webm", speed:int=4, 
    usage:str="q", quantizer:int=32, bitrate_min:int=1500,
    bitrate_mid:int=2000, bitrate_max:int=2500, gop:Optional[int]=None, 
    lif:Optional[int]=None, tiles_row:Optional[int]=None,
    tiles_col:Optional[int]=None, enable_cdef:bool=True,
    enable_restoration:Optional[bool]=None, enable_chroma_deltaq:bool=True,
    arnr_strength:int=2, arnr_max_frames:int=5, executable:str="aomenc") -> None
# YUV420P8, YUV422P8, YUV444P8, YUV420P10, YUV422P10, YUV444P10, YUV420P12,
# YUV422P12, YUV444P12
# The args that default to None are actually "Auto"
# Default container is webm

output.rav1e(clip:vs.VideoNode, file_str:str, speed:int=6, scd_speed:int=1,
    quantizer:int=100, gop:Optional[int]=None, 
    tiles_row:Optional[int]=None, tiles_col:Optional[int]=None,
    color_range:Optional[str]=None, primaties:Optional[str]=None, 
    transfer:Optional[str]=None, matrix:Optional[str]=None, 
    mastering_display:Optional[str]=None, content_light:Optional[str]=None, 
    executable:str="rav1e") -> None
# YUV420P8, YUV422P8, YUV444P8, YUV420P10, YUV422P10, YUV444P10,
# YUV420P12, YUV422P12, YUV444P12
# Container is ivf
```

### Read images

```python
imwri_src(dir:str, fpsnum:int, fpsden:int, firstnum:int=0,
    alpha:bool=False, ext:str=".png") -> vs.VideoNode
```

### Pad video to fit in box

```python
boundary_pad(clip:vs.VideoNode, boundary_width:int, boundary_height:int,
    x:int=50, y:int=50) -> vs.VideoNode
```

### Resize functions

```python
resize(clip:vs.VideoNode, width:Optional[int]=None,
    height:Optional[int]=None, scale:Optional[Union[int, float]]=None,
    kernel:str="didée", taps:Optional[int]=None) -> vs.VideoNode
```

```python
ssim_downsample(clip:vs.VideoNode, width:int, height:int, smooth:Union[int,float]=1,
        kernel:str="didée", gamma:bool=False, curve=vs.TransferCharacteristics.TRANSFER_BT709,
        sigmoid:bool=False, epsilon:float=0.000001) -> vs.VideoNode
```

```python
boundary_resize(clip:vs.VideoNode, width:Optional[int]=None,
    height:Optional[int]=None, scale:Optional[Union[int, float]]=None,
    multiple:int=2, crop:bool=False, kernel:str="didée",
    taps:Optional[int]=None, ssim:bool=False, ssim_kwargs:dict={}) -> vs.VideoNode
```

```python
down_to_444(clip:vs.VideoNode, kernel:str="didée") -> vs.VideoNode
# 4k 420 -> 1080p 444
up_to_444(clip:vs.VideoNode, kernel:str="robidoux") -> vs.VideoNode
# 4k 420 -> 4k 444
```

```python
nnedi3_rpow2(clip:vs.VideoNode, rfactor:int=2, correct_shift:bool=True,
    width:Optional[int]=None, height:Optional[int]=None,
    kernel:str="didée", nsize:int=0, nns:int=2, qual:int=2, etype:int=0, 
    pscrn:Optional[int]=None, opt:bool=True, 
    int16_prescreener:bool=True, int16_predictor:bool=True, exp:int=0) -> vs.VideoNode
```

### Adaptive Noise

```python
adaptive_noise(clip:vs.VideoNode, strength:float=0.25, static:bool=True,
    luma_scaling:float=12.0, show_mask:bool=False, noise_type:int=2) -> vs.VideoNode:
# Based on kagefunc's https://kageru.moe/blog/article/adaptivegrain
# uses https://github.com/wwww-wwww/vs-noise
```

### bt2020 to 709 according to bt2390

```python
bt2390_ictcp(clip:vs.VideoNode, source_peak:Optional[int]=None,
    target_nits:float=1) -> vs.VideoNode
```

### mvtools based scene detection

```python
mv_scene_detection(clip:vs.VideoNode, preset:str='fast', super_pel:int=2,
    thscd1:int=140, thscd2:int=15,
    overlap:int=0, overlapv:Optional[int]=None, search:Optional[int]=None,
    dct:int=0, truemotion:bool=True, blksize:int=8, blksizev:int=8,
    searchparam:int=2, badSAD:int=10000, badrange:int=24, divide:int=0) -> vs.VideoNode
```

### mvtools based motion interpolation

```python
mv_motion_interpolation(clip:vs.VideoNode, fpsnum:int=60, fpsden:int=1, preset:str='fast', 
    super_pel:int=2, block:bool=True, flow_mask:Optional[int]=None,
    block_mode:Optional[int]=None, Mblur:float=15.0,
    thscd1:int=140, thscd2:int=15, blend:bool=True,
    overlap:int=0, overlapv:Optional[int]=None, search:Optional[int]=None,
    dct:int=0, truemotion:bool=True, blksize:int=8, blksizev:int=8, searchparam:int=2,
    badSAD:int=10000, badrange:int=24, divide:int=0) -> vs.VideoNode
```

### mvtools based smoothing

```python
mv_flux_smooth(clip:vs.VideoNode, temporal_threshold:int=12, 
    super_params:dict={}, analyse_params:dict={}, compensate_params:dict={},
    planes=[0,1,2]) -> vs.VideoNode
    # port from https://forum.doom9.org/showthread.php?s=d58237a359f5b1f2ea45591cceea5133&p=1572664#post1572664
```

### Spatio-Temporal Pressdown using Motion Compensation

```python
STPressoMC(clip:vs.VideoNode, limit:int=3, bias:int=24, RGVS_mode:int=4,
    temporal_threshold:int=12, temporal_limit:int=3, temporal_bias:int=49, back:int=1,
    super_params:dict={}, analyse_params:dict={}, compensate_params:dict={}) -> vs.VideoNode
# orginal script by Didée
```

### Deringing

```python
HQDeringmod(clip:vs.VideoNode, p:Optional[vs.VideoNode]=None,
    ringmask:Optional[vs.VideoNode]=None, mrad:int=1, msmooth:int=1,
    incedge:bool=False, mthr:int=60, minp:int=1, nrmode:Optional[int]=None,
    sharp:int=1, drrep:int=24, thr:float=12.0, elast:float=2.0,
    darkthr:Optional[float]=None, planes:List[int]=[0], show:bool=False) -> vs.VideoNode
# original script by mawen1250
```

### Smart Sharpening

```python
fine_sharp(clip:vs.VideoNode, mode:int=1, 
    sstr:float=2.5, cstr:Optional[float]=None, xstr:float=0, lstr:float=1.5,
    pstr:float=1.28, ldmp:Optional[float]=None, hdmp:float=0.01, rep:int=12) -> vs.VideoNode
    original script by Didée
```

### Edge Detection

```python
edge_detection(clip:vs.VideoNode, planes:Optional[Union[int,List[int]]]=None,
    method:str="kirsch", thr:Optional[int]=None, scale:bool=False) -> vs.VideoNode
```

```python
retinex_edge(clip:vs.VideoNode, method="prewitt", sigma:int=1,
    thr:Optional[int]=None, plane:int=0) -> vs.VideoNode
```

### Debanding

```python
retinex_deband(clip:vs.VideoNode, preset:str="high/nograin",
    f3kdb_kwargs:dict={}, method:str="p", thr:Optional[int]=None) -> vs.VideoNode
```
