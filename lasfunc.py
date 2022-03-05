import vapoursynth as vs
import math, os, subprocess
from typing import Optional, Union

# import havsfunc
# import kagefunc
# import mvsfunc

# Requriements: imwri, mvtools, fmtc, 
# https://github.com/wwww-wwww/vs-noise
# https://git.kageru.moe/kageru/adaptivegrain

# Docs
# https://github.com/vapoursynth/vs-imwri/blob/master/docs/imwri.rst
# https://github.com/dubhater/vapoursynth-mvtools
# http://avisynth.org.ru/mvtools/mvtools2.html

# TODO: Bring in more functions, native way to encode ffv1/av1? sounds hard af.
    # from havsfunc: HQDeringmod
    # New adaptive_grain based on https://github.com/wwww-wwww/vs-noise

class ffmpeg:
    # TODO: Handle rgb.

    def __get_progress__(a, b):
        s = "Progress: {}% {}/{}".format(str(math.floor((a/b)*100)).rjust(3,' '), str(a).rjust(str(b).__len__(),' '), b)
        print(s, end="\r")

    def aom(clip:vs.VideoNode, filestr:str, speed:int=4, quantizer:int=32, gop:int=None, lif:int=None, tiles_row:int=None, tiles_col:int=None, enable_cdef:bool=True, enable_restoration:bool=None, enable_chroma_deltaq:bool=True, arnr_strength:int=2, arnr_max_frames:int=5):
        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10', 'YUV420P12', 'YUV422P12', 'YUV444P12']:
            raise ValueError('Pixel format must be one of YUV420P8 YUV422P8 YUV444P8 YUV420P10 YUV422P10 YUV444P10 YUV420P12 YUV422P12 YUV444P12')

        if gop is None: gop = min(300, round(clip.fps.numerator/clip.fps.denominator)*10)
        if lif is None: lif = min(35, gop)
        if tiles_row is None: tiles_row = math.floor(clip.height/1080)
        if tiles_col is None: tiles_col = math.floor(clip.width/1920)

        if enable_restoration is None: 
            if (clip.height*clip.width >= 3840*2160): # if smaller than 2160p
                enable_restoration = True
            else: enable_restoration = False

        aom_params = f"enable-qm=1:qm-min=5"

        if (clip.height*clip.width < 1920*1080): # if smaller than 1080p
            aom_params += ":max-partition-size=64:sb-size=64"

        ffmpeg_str = f"ffmpeg -y -hide_banner -v 8 -i - -c libaom-av1 \
-cpu-used {speed} -crf {quantizer} -g {gop} -lag-in-frames {lif} -tile-columns {tiles_col} \
-tile-rows {tiles_row} -enable-cdef {enable_cdef} -enable-restoration {enable_restoration} \
-arnr-strength {arnr_strength} -arnr-max-frames {arnr_max_frames} -aom-params \"{aom_params}\" -f ivf -"

        file = open(filestr, 'wb')
        with subprocess.Popen(ffmpeg_str, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True, progress_update=ffmpeg.__get_progress__)
            process.stdin.close()
        file.close()

    def svt(clip:vs.VideoNode, filestr:str, speed:int=6, quantizer:int=32, gop:int=None, lad:int=None, tiles_row:int=None, tiles_col:int=None, sc_detection:bool=False):
        if clip.format.name not in ['YUV420P8', 'YUV420P10']:
            raise ValueError('Pixel format must be one of YUV420P8 YUV420P10')

        if gop is None: gop = min(300, round(clip.fps.numerator/clip.fps.denominator)*10)
        if lad is None: lad = min(120, gop)
        if tiles_row is None: tiles_row = math.floor(clip.height/1080)
        if tiles_col is None: tiles_col = math.floor(clip.width/1920)

        ffmpeg_str = f"ffmpeg -y -hide_banner -v 8 -i - -c libsvtav1 \
-preset {speed} -rc cqp -qp {quantizer} -g {gop} -la_depth {lad} \
-tile_rows {tiles_row} -tile_columns {tiles_col} -sc_detection {sc_detection} -f ivf -"

        file = open(filestr, 'wb')
        with subprocess.Popen(ffmpeg_str, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True, progress_update=ffmpeg.__get_progress__)
            process.stdin.close()
        file.close()

    def ffv1(clip:vs.VideoNode, filestr:str):
        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10', 'YUV420P12', 'YUV422P12', 'YUV444P12', 'YUV420P16', 'YUV422P16', 'YUV444P16']:
            raise ValueError('Pixel format must be one of YUV420P8 YUV422P8 YUV444P8 YUV420P10 YUV422P10 YUV444P10 YUV420P12 YUV422P12 YUV444P12 YUV420P16 YUV422P16 YUV444P16')

        ffmpeg_str = f"ffmpeg -y -hide_banner -v 8 -i - -c ffv1 -f nut - "

        file = open(filestr, 'wb')
        with subprocess.Popen(ffmpeg_str, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True, progress_update=ffmpeg.__get_progress__)
            process.stdin.close()
        file.close()

def round_to_closest(n:Union[int,float]) -> int:
    # I'm amazed that's not a thing.
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)

def boundary_pad(clip:vs.VideoNode, boundary_width:int, boundary_height:int) -> vs.VideoNode:
    if (boundary_width > clip.width) or (boundary_height > clip.height):
        clip = vs.core.std.AddBorders(clip, left=(boundary_width-clip.width)/2, right=(boundary_width-clip.width)/2, top=(boundary_height-clip.height)/2, bottom=(boundary_height-clip.height)/2)
    return clip

def boundary_resize(clip:vs.VideoNode, boundary_width:int, boundary_height:int, 
        multiple:int=1, crop:bool=False, 
        resize_kernel:str="Spline36") -> vs.VideoNode:

    # Determine rescaling values
    new_height = original_height = clip.height
    new_width  = original_width  = clip.width

    # first check if we need to scale the height
    if new_height > boundary_height:
        # scale height to fit instead
        new_height = boundary_height
        # scale width to maintain aspect ratio
        new_width = round_to_closest((new_height * original_width) / original_height)

    # then check if we need to scale even with the new height
    if new_width > boundary_width:
        # scale width to fit
        new_width = boundary_width
        # scale height to maintain aspect ratio
        new_height = round_to_closest((new_width * original_height) / original_width)

    resize_func = getattr(vs.core.resize, resize_kernel)
    clip = resize_func(clip=clip, height=new_height, width=new_width)

    # Make divisible by 2 for vp9 :)
    if multiple > 1:
        new_height_div = math.floor(new_height/multiple)*multiple
        new_width_div  = math.floor(new_width/multiple)*multiple
        if crop:
            if new_height_div != new_height or new_width_div != new_width:
                clip = vs.core.std.CropAbs(clip=clip, height=new_height_div,
                                                        width=new_width_div)
    return clip

def bt2390_ictcp(clip:vs.VideoNode, source_peak:Optional[int]=None,
        target_nits:float=1) -> vs.VideoNode:
    # Stolen from somewhere, idk where.
    # clip = bt2390-vs.bt2390_ictcp(clip,target_nits=100,source_peak=1000)

    # TODO: Rewrite this mess, Fix var names names.

    if source_peak is None:
        source_peak = clip.get_frame(0).props.MasteringDisplayMax_Luminance

    primaries = "2020"    
    source_peak = source_peak
    matrix_in_s = "2020ncl"
    transfer_in_s = "st2084"    
    exposure_bias1 = source_peak/target_nits
    source_peak = source_peak 
    width = clip.width
    height = clip.height    
    width_n = width
    height_n = (height*width_n)/width

    clip = vs.core.resize.Spline36(clip=clip, format=vs.YUV444PS,
                        filter_param_a=0,width=width_n,height=height_n,
                        filter_param_b=0.75, chromaloc_in_s="center",
                        chromaloc_s="center", range_in_s="limited",
                        range_s="full", dither_type="none")

    luma_w = source_peak/10000   
    # eotf^-1  y=((x^0.1593017578125) * 18.8515625 + 0.8359375 / (x^0.1593017578125) * 18.6875 + 1)^78.84375
    luma_w = ((((luma_w**0.1593017578125)*18.8515625)+0.8359375)/(((luma_w**0.1593017578125)*18.6875)+1))**78.84375


    luma_max = target_nits/10000
    #eotf^-1  y=((x^0.1593017578125) * 18.8515625 + 0.8359375 / (x^0.1593017578125) * 18.6875 + 1)^78.84375
    luma_max = ((((luma_max**0.1593017578125)*18.8515625)+0.8359375)/(((luma_max**0.1593017578125)*18.6875)+1))**78.84375    

    #max_luma=(luma_max-0)/(luma_w-0) ==> max_luma=luma_max/luma_w
    max_luma = luma_max/luma_w  

    #ks1=(1.5*luma_max)- 0.5      
    ks1 = (1.5*luma_max)- 0.5

    #ks2=(1.5*max_luma)- 0.5
    ks2 = (1.5*max_luma)- 0.5 

    #ks=(ks1+ks2)/2    
    ks = (ks1+ks2)/2

    c_ictcp = vs.core.resize.Spline36(clip=clip, format=vs.YUV444PS, filter_param_a=0, filter_param_b=0.75, chromaloc_in_s="center", transfer_in_s=transfer_in_s, chromaloc_s="center", range_in_s="full", range_s="full", dither_type="none", nominal_luminance=source_peak, matrix_in_s=matrix_in_s, matrix_s="ictcp")

    luma_intensity = vs.core.std.ShufflePlanes(c_ictcp, planes=[0], colorfamily=vs.GRAY)
    chroma_tritanopia = vs.core.std.ShufflePlanes(c_ictcp, planes=[1], colorfamily=vs.GRAY)
    chroma_protanopia = vs.core.std.ShufflePlanes(c_ictcp, planes=[2], colorfamily=vs.GRAY)

    luma_intensity = vs.core.std.Limiter(luma_intensity, 0, luma_w, planes=[0])      

    clip = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, filter_param_a=0, filter_param_b=0.75, chromaloc_in_s="center", chromaloc_s="center", range_in_s="full", range_s="full", dither_type="none", matrix_in_s="2020ncl")
    clip = vs.core.std.Limiter(clip, 0, luma_w) 

    cs = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, transfer_in_s=transfer_in_s, transfer_s="linear", dither_type="none", nominal_luminance=source_peak)

    cs = vs.core.std.Expr(clips=[cs], expr="x {exposure_bias1} *".format(exposure_bias1=exposure_bias1)) 

    r = vs.core.std.ShufflePlanes(cs, planes=[0], colorfamily=vs.GRAY)
    g = vs.core.std.ShufflePlanes(cs, planes=[1], colorfamily=vs.GRAY)
    b = vs.core.std.ShufflePlanes(cs, planes=[2], colorfamily=vs.GRAY)
    max = vs.core.std.Expr(clips=[r,g,b], expr="x y max z max") 
    min = vs.core.std.Expr(clips=[r,g,b], expr="x y min z min")
    sat = vs.core.std.Expr(clips=[r,g,b], expr="x x * y y * + z z * + x y + z + /")       
    l = vs.core.std.Expr(clips=[r,g,b], expr="0.2627 x * 0.6780 y * + 0.0593 z * +") 
    l = vs.core.std.ShufflePlanes(clips=[l,l,l], planes=[0,0,0], colorfamily=vs.RGB)      

    saturation_mult1 = vs.core.std.Expr(clips=[sat], expr="x 1 - {exposure_bias1} 1 - /".format(exposure_bias1=exposure_bias1))
    saturation_mult1 = vs.core.std.Limiter(saturation_mult1, 0, 1)    

    c1 = vs.core.std.MaskedMerge(cs, l, saturation_mult1)
    clip = vs.core.std.Merge(cs, c1, 0.5)

    clip = vs.core.std.Expr(clips=[clip], expr="x {exposure_bias1} /".format(exposure_bias1=exposure_bias1)) 

    clip = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, transfer_in_s="linear", transfer_s=transfer_in_s, dither_type="none", nominal_luminance=source_peak,cpu_type=None)

    e1 = vs.core.std.Expr(clips=[clip], expr="x  {luma_w} /".format(luma_w=luma_w))
    t = vs.core.std.Expr(clips=[e1], expr="x {ks} - 1 {ks} - /".format(ks=ks))    
    p = vs.core.std.Expr(clips=[t], expr="2 x 3 pow * 3 x 2 pow * - 1 + {ks} * 1 {ks} - x 3 pow 2 x 2 pow * - x + * + -2 x 3 pow * 3 x 2 pow * + {max_luma} * +".format(ks=ks,max_luma=max_luma))    
    e2 = vs.core.std.Expr(clips=[e1,p], expr="x {ks} < x y ?".format(ks=ks))
    crgb = vs.core.std.Expr(clips=[e2], expr="x {luma_w} *".format(luma_w=luma_w))   
    crgb = vs.core.std.Limiter(crgb, 0, 1)  

    rgb = crgb

    crgb = vs.core.resize.Spline36(clip=crgb, format=vs.YUV444PS, filter_param_a=0, filter_param_b=0.75, chromaloc_in_s="center", transfer_in_s=transfer_in_s, transfer_s=transfer_in_s, chromaloc_s="center", range_in_s="full", range_s="full", dither_type="none", nominal_luminance=target_nits, matrix_s="ictcp",cpu_type=None)

    Irgb = vs.core.std.ShufflePlanes(crgb, planes=[0], colorfamily=vs.GRAY)

    saturation_mult1 = vs.core.std.Expr(clips=[saturation_mult1], expr="1 x -".format(luma_w=luma_w,luma_max=luma_max))

    saturation_mult = vs.core.std.Expr(clips=[luma_intensity,Irgb], expr="y x  /")
    saturation_mult = vs.core.std.Limiter(saturation_mult, 0, 1)

    chroma_tritanopia = vs.core.std.Expr(clips=[chroma_tritanopia,saturation_mult1], expr="x y *")
    chroma_protanopia = vs.core.std.Expr(clips=[chroma_protanopia,saturation_mult1], expr="x y *") 

    chroma_tritanopia = vs.core.std.Expr(clips=[chroma_tritanopia,saturation_mult], expr="x y *")
    chroma_protanopia = vs.core.std.Expr(clips=[chroma_protanopia,saturation_mult], expr="x y *")    

    c_ictcp = vs.core.std.ShufflePlanes(clips=[Irgb,chroma_tritanopia,chroma_protanopia], planes=[0,0,0], colorfamily=vs.YUV)

    c_ictcp = vs.core.resize.Spline36(clip=c_ictcp, format=vs.RGBS, filter_param_a=0, filter_param_b=0.75, chromaloc_in_s="center", transfer_in_s=transfer_in_s,transfer_s=transfer_in_s, chromaloc_s="center", range_in_s="full", range_s="full", dither_type="none", nominal_luminance=target_nits, matrix_in_s="ictcp",cpu_type=None)

    c_ictcp = vs.core.std.Limiter(c_ictcp, 0, 1)

    clip = vs.core.std.Merge(c_ictcp, rgb, 0.75)
    clip = vs.core.std.Limiter(clip, 0, 1)

    clip = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, transfer_in_s=transfer_in_s, transfer_s="linear", dither_type="none", nominal_luminance=target_nits)

    clip = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, primaries_in_s=primaries, primaries_s="709", dither_type="none")
    clip = vs.core.std.Limiter(clip, 0, 1)

    clip = vs.core.std.Expr(clips=[clip], expr="x 1 2.2 / pow")

#    clip = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, transfer_in_s="linear", transfer_s="709", dither_type="none")
    clip = vs.core.std.Limiter(clip, 0, 1)

    clip = vs.core.resize.Spline36(clip=clip, format=vs.YUV422P16, matrix_s="709", filter_param_a=0, filter_param_b=0.75, range_in_s="full",range_s="limited", chromaloc_in_s="center", chromaloc_s="center", dither_type="none")  
    return clip

def imwri_src(dir:str, fpsnum:int, fpsden:int, firstnum:int=0, alpha:bool=False) -> vs.VideoNode:

    srcs = [dir + src for src in os.listdir(dir)]
    clip = vs.core.imwri.Read(srcs, firstnum=firstnum, alpha=alpha)
    clip = vs.core.std.AssumeFPS(clip=clip, fpsnum=fpsnum, fpsden=fpsden)
    return clip

def yuvtorgb(clip:vs.VideoNode, bits:int=32) -> vs.VideoNode:

    clip = vs.core.fmtc.resample(clip=clip, css="444")
    clip = vs.core.fmtc.matrix(clip=clip, mat="709", col_fam=vs.RGB)
    clip = vs.core.fmtc.bitdepth(clip=clip, bits=bits)
    return clip

def rgbtoyuv(clip:vs.VideoNode, bits:int=16, css:str="444") -> vs.VideoNode:

    clip = vs.core.fmtc.matrix(clip, mat="709", fulls=True, fulld=False, col_fam=vs.YUV)
    clip = vs.core.fmtc.resample(clip, css=css)
    clip = vs.core.fmtc.bitdepth(clip, bits=bits)
    return clip

def mvscd(clip:vs.VideoNode, preset:str='fast', super_pel:int=2,

        thscd1:int=140, thscd2:int=15,

        overlap:int=0, overlapv:Optional[int]=None, search:Optional[int]=None,
        dct:int=0, truemotion:bool=True, blksize:int=8, blksizev:int=8,
        searchparam:int=2, badSAD:int=10000, badrange:int=24, divide:int=0) -> vs.VideoNode:
    # mvtools scene detection

    # thSCD1 (int): threshold which decides whether a block has changed between the previous frame and the current one. When a block has changed, it means that motion estimation for it isn't relevant. It occurs for example at scene changes. So it is one of the thresholds used to tweak the scene changes detection engine. Raising it will lower the number of blocks detected as changed. It may be useful for noisy or flickered video. The threshold is compared to the SAD (Sum of Absolute Differences, a value which says how bad the motion estimation was ) value. For exactly identical blocks we have SAD=0. But real blocks are always different because of objects complex movement (zoom, rotation, deformation), discrete pixels sampling, and noise. Suppose we have two compared 8x8 blocks with every pixel different by 5. It this case SAD will be 8x8x5 = 320 (block will not detected as changed for thSCD1=400). If you use 4x4 blocks, SAD will be 320/4. If you use 16x16 blocks, SAD will be 320*4. Really this parameter is scaled internally in MVTools, and you must always use reduced to block size 8x8 value. Default is 400 (since v.1.4.1).
    # thSCD2 (int): threshold which sets how many blocks have to change for the frame to be considered as a scene change. It is ranged from 0 to 255, 0 meaning 0 %, 255 meaning 100 %. Default is 130 ( which means 51 % ). 

    if preset == 'fast':
        preset_number = 0
    elif preset == 'medium':
        preset_number = 1
    elif preset == 'slow':
        preset_number = 2

    if overlapv is None: overlapv = overlap
    if search is None: search = [0,5,3][preset_number]

    analyse_params = {
        'overlap' : overlap,
        'overlapv':overlapv,
        'search' : search,
        'dct':dct,
        'truemotion' : truemotion,
        'blksize' : blksize,
        'blksizev':blksizev,
        'searchparam':searchparam,
        'badsad':badSAD,
        'badrange':badrange,
        'divide':divide
    }

    mvsuper = vs.core.mv.Super(clip, pel=super_pel, sharp=2, rfilter=4)
    vectors = vs.core.mv.Analyse(mvsuper, isb=True, **analyse_params)

    clip = vs.core.mv.SCDetection(clip, vectors=vectors, thscd1=thscd1, thscd2=int(thscd2*255/100))
    return clip

def mvmi(clip:vs.VideoNode, fpsnum:int=60, fpsden:int=1, preset:str='fast', 
        super_pel:int=2, block:bool=True, flow_mask:Optional[int]=None,
        block_mode:Optional[int]=None, Mblur:float=15.0,

        thscd1:int=140, thscd2:int=15, blend:bool=True,

        overlap:int=0, overlapv:Optional[int]=None, search:Optional[int]=None,
        dct:int=0, truemotion:bool=True, blksize:int=8, blksizev:int=8, searchparam:int=2,
        badSAD:int=10000, badrange:int=24, divide:int=0) -> vs.VideoNode:
    # mvtools motion interpolation

    # Source: xvs.mvfrc, modified

    # TODO: clean variable names, make easier to use. for real I'm never using all of these params.

    if preset == 'fast':
        preset_number = 0
    elif preset == 'medium':
        preset_number = 1
    elif preset == 'slow':
        preset_number = 2

    if overlapv is None: overlapv = overlap
    if search is None: search = [0,5,3][preset_number]
    if block_mode is None: block_mode = [0,0,3][preset_number]
    if flow_mask is None: flow_mask = [0,0,2][preset_number]

    analyse_params = {
        'overlap' : overlap,
        'overlapv':overlapv,
        'search' : search,
        'dct':dct,
        'truemotion' : truemotion,
        'blksize' : blksize,
        'blksizev':blksizev,
        'searchparam':searchparam,
        'badsad':badSAD,
        'badrange':badrange,
        'divide':divide
    }

    #block or flow Params 
    block_or_flow_params = {
        'thscd1':thscd1,
        'thscd2':int(thscd2*255/100),
        'blend':blend,
        'num':fpsnum,
        'den':fpsden
    }

    mvsuper = vs.core.mv.Super(clip, pel=super_pel, sharp=2, rfilter=4)
    block_vectors = vs.core.mv.Analyse(mvsuper, isb=True, **analyse_params)
    flow_vectors = vs.core.mv.Analyse(mvsuper, isb=False, **analyse_params)

    if clip.fps.numerator/clip.fps.denominator > fpsnum/fpsden:
        clip = vs.core.mv.FlowBlur(clip, mvsuper, block_vectors, flow_vectors, blur=Mblur)

    if block == True:
        out =  vs.core.mv.BlockFPS(clip, mvsuper, block_vectors, flow_vectors, **block_or_flow_params, mode=block_mode)
    else:
        out = vs.core.mv.FlowFPS(clip, mvsuper, block_vectors, flow_vectors, **block_or_flow_params, mask=flow_mask)
    return out

def adaptive_noise(clip: vs.VideoNode, strength:float=0.25, static:bool=True,
        luma_scaling:float=12.0, show_mask:bool=False, noise_type:int=2) -> vs.VideoNode:
    # Based on kagefunc's
    # https://kageru.moe/blog/article/adaptivegrain
    # uses https://github.com/wwww-wwww/vs-noise

    mask = vs.core.adg.Mask(clip.std.PlaneStats(), luma_scaling)
    grained = vs.core.noise.Add(clip, var=strength, constant=static, type=noise_type)
    if show_mask:
        return mask

    return vs.core.std.MaskedMerge(clip, grained, mask)
