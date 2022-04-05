# lasfunc
A set of functions for vapoursynth.

## Usage

### Encode a clip to a lossless format using ffmpeg.
    output.ffv1(clip:vs.VideoNode, file_str:str, mux:str='nut', exec:str='ffmpeg') -> None
    output.llx264(clip:vs.VideoNode, file_str:str, preset:str='veryslow', mux:str='nut', exec:str='ffmpeg') -> None
    # Contained in nut, you can use "matroska"

### Encode a clip to av1 using aomenc.
    output.aom(clip:vs.VideoNode, file_str:str, mux:str="webm", speed:int=4,
        usage:str="q", quantizer:int=32, bitrate_min:int=1500,bitrate_mid:int=2000,
        bitrate_max:int=2500, gop:int=None, lif:int=None,tiles_row:int=None,
        tiles_col:int=None, enable_cdef:bool=True, enable_restoration:bool=None,
        enable_chroma_deltaq:bool=True,arnr_strength:int=2, arnr_max_frames:int=5,
        exec:str="aomenc") -> None
    # The args that default to None are actually "Auto"
    # Default container is webm

### Read images.
    imwri_src(dir:str, fpsnum:int, fpsden:int, firstnum:int=0, alpha:bool=False) -> vs.VideoNode

### Should be part of python.
    round_to_closest(n:Union[int,float]) -> int

### Pad video to fit in box.
    boundary_pad(clip:vs.VideoNode, boundary_width:int, boundary_height:int) -> vs.VideoNode

### Resize video to fit in box.
    boundary_resize(clip:vs.VideoNode, boundary_width:int, boundary_height:int, 
        multiple:int=1, crop:bool=False, 
        resize_kernel:str="Spline36") -> vs.VideoNode

### adaptive_noise
    adaptive_noise(clip:vs.VideoNode, strength:float=0.25, static:bool=True,
        luma_scaling:float=12.0, show_mask:bool=False, noise_type:int=2) -> vs.VideoNode:
    # Based on kagefunc's https://kageru.moe/blog/article/adaptivegrain
    # uses https://github.com/wwww-wwww/vs-noise

### bt2020 to 709 according to bt2390.
    bt2390_ictcp(clip:vs.VideoNode, source_peak:Optional[int]=None,
        target_nits:float=1) -> vs.VideoNode

### mvtools based scene detection.
    mv_scene_detection(clip:vs.VideoNode, preset:str='fast', super_pel:int=2,
        thscd1:int=140, thscd2:int=15,
        overlap:int=0, overlapv:Optional[int]=None, search:Optional[int]=None,
        dct:int=0, truemotion:bool=True, blksize:int=8, blksizev:int=8,
        searchparam:int=2, badSAD:int=10000, badrange:int=24, divide:int=0) -> vs.VideoNode

### mvtools based motion interpolation.
    mv_motion_interpolation(clip:vs.VideoNode, fpsnum:int=60000, fpsden:int=1001, preset:str='fast', 
        super_pel:int=2, block:bool=True, flow_mask:Optional[int]=None,
        block_mode:Optional[int]=None, Mblur:float=15.0,
        thscd1:int=140, thscd2:int=15, blend:bool=True,
        overlap:int=0, overlapv:Optional[int]=None, search:Optional[int]=None,
        dct:int=0, truemotion:bool=True, blksize:int=8, blksizev:int=8, searchparam:int=2,
        badSAD:int=10000, badrange:int=24, divide:int=0) -> vs.VideoNode

### Spatio-Temporal Pressdown using Motion Compensation
    STPressoMC(clip:vs.VideoNode, limit:int=3, bias:int=24, RGVS_mode:int=4,
    temporal_threshold:int=12, temporal_limit:int=3, temporal_bias:int=49, back:int=1,
    super_params:dict={}, analyse_params:dict={}, compensate_params:dict={}) -> vs.VideoNode:
    # orginal script by Didée, taken from xvs
