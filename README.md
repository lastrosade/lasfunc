# lasfunc
A set of functions for vapoursynth.

## Usage

### Write a clip to a y4m file.
    write_to_file(clip:vs.VideoNode, filestr:str) -> None

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

### bt2020 to 709 according to bt2390.
    bt2390_ictcp(clip:vs.VideoNode, source_peak:Optional[int]=None,
        target_nits:float=1) -> vs.VideoNode

### I use this for vsrife.
    yuvtorgb(clip:vs.VideoNode, bits:int=32) -> vs.VideoNode
    rgbtoyuv(clip:vs.VideoNode, bits:int=16, css:str="444") -> vs.VideoNode

### mvtools based scene detection.
    mvscd(clip:vs.VideoNode, preset:str='fast', super_pel:int=2,
        thscd1:int=140, thscd2:int=15,
        overlap:int=0, overlapv:Optional[int]=None, search:Optional[int]=None,
        dct:int=0, truemotion:bool=True, blksize:int=8, blksizev:int=8,
        searchparam:int=2, badSAD:int=10000, badrange:int=24, divide:int=0) -> vs.VideoNode

### mvtools based motion interpolation.
    mvmi(clip:vs.VideoNode, fpsnum:int=60000, fpsden:int=1001, preset:str='fast', 
        super_pel:int=2, block:bool=True, flow_mask:Optional[int]=None,
        block_mode:Optional[int]=None, Mblur:float=15.0,
        thscd1:int=140, thscd2:int=15, blend:bool=True,
        overlap:int=0, overlapv:Optional[int]=None, search:Optional[int]=None,
        dct:int=0, truemotion:bool=True, blksize:int=8, blksizev:int=8, searchparam:int=2,
        badSAD:int=10000, badrange:int=24, divide:int=0) -> vs.VideoNode
