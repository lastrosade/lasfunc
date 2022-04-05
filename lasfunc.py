import vapoursynth as vs
import math, os, subprocess
from typing import Optional, Union, List

# import havsfunc
# import kagefunc
# import mvsfunc

# Requriements: imwri, mvtools, fmtc, rgvs, flux
# https://github.com/wwww-wwww/vs-noise
# https://git.kageru.moe/kageru/adaptivegrain

# Docs
# https://github.com/vapoursynth/vs-imwri/blob/master/docs/imwri.rst
# https://github.com/dubhater/vapoursynth-mvtools
# http://avisynth.org.ru/mvtools/mvtools2.html

# TODO: Bring in more functions, native way to encode ffv1/av1? sounds hard af.
    # from havsfunc: HQDeringmod
    # New adaptive_grain based on https://github.com/wwww-wwww/vs-noise

class util:
    # stolen from vsutil btw.

    def plane(clip:vs.VideoNode, planeno:int) -> vs.VideoNode:
        if clip.format.num_planes == 1 and planeno == 0:
            return clip
        return vs.core.std.ShufflePlanes(clip, planeno, vs.GRAY)

    def join(planes:List[vs.VideoNode], family:vs.ColorFamily=vs.YUV) -> vs.VideoNode:
        if family not in [vs.RGB, vs.YUV]:
            raise vs.Error('Color family must have 3 planes.')
        return vs.core.std.ShufflePlanes(clips=planes, planes=[0, 0, 0], colorfamily=family)

    def split(clip:vs.VideoNode) -> List[vs.VideoNode]:
        return [util.plane(clip, x) for x in range(clip.format.num_planes)]

class output:
    def rav1e(clip:vs.VideoNode, file_str:str, speed:int=6, scd_speed:int=1,
            quantizer:int=100, gop:int=None, tiles_row:int=None, tiles_col:int=None,
            color_range:str=None, primaties:str=None, transfer:str=None,
            matrix:str=None, mastering_display:str=None, content_light:str=None, exec:str="rav1e"):
            # two_pass:bool=False, two_pass_file:str=None

        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10', 'YUV420P12', 'YUV422P12', 'YUV444P12']:
            raise vs.Error('Pixel format must be one of `YUV420P8 YUV422P8 YUV444P8 YUV420P10 YUV422P10 YUV444P10 YUV420P12 YUV422P12 YUV444P12`')

        if gop is None: gop = min(300, round(clip.fps.numerator/clip.fps.denominator)*10)

        args = [
            exec,
            "-",
            "-o", "-",
            "--quantizer", f"{quantizer}",
            "--speed", f"{speed}",
            "--scd_speed", f"{scd_speed}",
            "--keyint", f"{gop}"
        ]

        if color_range       is not None: args += [f"--range {color_range}"]
        if primaties         is not None: args += [f"--primaries {primaties}"]
        if transfer          is not None: args += [f"--transfer {transfer}"]
        if matrix            is not None: args += [f"--matrix {matrix}"]
        if mastering_display is not None: args += [f"--mastering-display {mastering_display}"]
        if content_light     is not None: args += [f"--content-light {content_light}"]
        if tiles_row         is not None: args += [f"--tile-rows {tiles_row}"]
        if tiles_col         is not None: args += [f"--tile-cols {tiles_col}"]

        args += ["-"]

        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True)
            process.stdin.close()
        file.close()

    def aomenc(clip:vs.VideoNode, file_str:str, mux:str="webm", speed:int=4, 
            usage:str="q", quantizer:int=32, bitrate_min:int=1500,
            bitrate_mid:int=2000, bitrate_max:int=2500, gop:int=None, lif:int=None,
            tiles_row:int=None, tiles_col:int=None, enable_cdef:bool=True, 
            enable_restoration:bool=None, enable_chroma_deltaq:bool=True,
            arnr_strength:int=2, arnr_max_frames:int=5, exec:str="aomenc"):

        # Only Q or VBR

        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10', 'YUV420P12', 'YUV422P12', 'YUV444P12']:
            raise vs.Error('Pixel format must be one of `YUV420P8 YUV422P8 YUV444P8 YUV420P10 YUV422P10 YUV444P10 YUV420P12 YUV422P12 YUV444P12`')

        if (clip.format.name in ["YUV420P8", "YUV420P10"]):
            profile=0
        elif (clip.format.name in ["YUV444P8", "YUV444P10"]):
            profile=1
        elif (clip.format.name in ["YUV422P8", "YUV422P10", "YUV422P12", "YUV420P12", "YUV444P12"]):
            profile=2

        if (mux not in ["ivf", "webm", "obu"]):
            raise vs.Error('Muxing container format must be one of `ivf webm obu`')

        if gop is None: gop = min(300, round(clip.fps.numerator/clip.fps.denominator)*10)
        if lif is None: lif = min(35, gop)
        if tiles_row is None: tiles_row = math.floor(clip.height/1080)
        if tiles_col is None: tiles_col = math.floor(clip.width/1920)

        if usage == "q":
            quantizer_args = ["--end-usage=q", f"--cq-level={quantizer}"]
        elif usage == "vbr":
            bitrate_undershoot_pct = round_to_closest(((bitrate_mid - bitrate_min) / bitrate_min)*100)
            bitrate_overshoot_pct = round_to_closest(((bitrate_max - bitrate_mid) / bitrate_mid)*100)
            quantizer_args = [
                "--end-usage=vbr",
                "--bias-pct=75",
                f"--target-bitrate={bitrate}",
                f"--undershoot-pct={bitrate_undershoot_pct}",
                f"--overshoot-pct={bitrate_overshoot_pct}"
            ]

        if enable_cdef: enable_cdef = 1
        else: enable_cdef = 0
        if enable_chroma_deltaq: enable_chroma_deltaq = 1
        else: enable_chroma_deltaq = 0

        if enable_restoration is None: 
            if (clip.height*clip.width >= 3200*2000) or not enable_restoration: # if smaller than 2160p
                enable_restoration = 0
            else: enable_restoration = 1

        if (clip.height*clip.width < 1280*720):
            parts_args = ["--max-partition-size=64", "--sb-size=64"] # sb-size used to be 32 but now it can't work? `dynamic, 64, 128`
        elif (clip.height*clip.width < 1920*1080):
            parts_args = ["--max-partition-size=64", "--sb-size=64"]
        else:
            parts_args = []

        args = [
            exec,
            "-",
            f"--{mux}" ,
            "--passes=1",
            f"--profile={profile}",
            f"--bit-depth={clip.format.bits_per_sample}",
            f"--cpu-used={speed}"
        ] + quantizer_args + [
            f"--fps={clip.fps.numerator}/{clip.fps.denominator}",
            f"--input-chroma-subsampling-x={clip.format.subsampling_w}",
            f"--input-chroma-subsampling-y={clip.format.subsampling_h}",
            f"--input-bit-depth={clip.format.bits_per_sample}",
            f"--kf-max-dist={gop}",
            f"--lag-in-frames={lif}",
            "--row-mt=1",
            f"--tile-columns={tiles_row}",
            f"--tile-rows={tiles_row}",
            f"--enable-chroma-deltaq={enable_chroma_deltaq}",
            f"--enable-cdef={enable_cdef}",
            f"--enable-restoration={enable_restoration}",
            f"--arnr-strength={arnr_strength}",
            f"--arnr-maxframes={arnr_max_frames}",
        ] + parts_args + [
            "--min-q=1",
            f"--enable-fwd-kf=1",
            "--quant-b-adapt=1",
            "--enable-dual-filter=0",
            f"--enable-qm=1",
            "--qm-min=5",
            "--output=-"
        ]

        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True)
            process.stdin.close()
        file.close()

    def __get_progress__(a, b):
        s = f"Progress: {str(math.floor((a/b)*100)).rjust(3,' ')}% {str(a).rjust(str(b).__len__())}/{b}"
        print(s, end="\r")

    # def aom(clip:vs.VideoNode, file_str:str, speed:int=4, quantizer:int=32,
    #         gop:int=None, lif:int=None, tiles_row:int=None, tiles_col:int=None,
    #         enable_cdef:bool=True, enable_restoration:bool=None, 
    #         enable_chroma_deltaq:bool=True, arnr_strength:int=2,
    #         arnr_max_frames:int=5, mux:str='ivf'):
    #     #   TODO: Handle rgb lmao.

    #     if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10', 'YUV420P12', 'YUV422P12', 'YUV444P12']:
    #         raise vs.Error('Pixel format must be one of `YUV420P8 YUV422P8 YUV444P8 YUV420P10 YUV422P10 YUV444P10 YUV420P12 YUV422P12 YUV444P12`')

    #     if gop is None: gop = min(300, round(clip.fps.numerator/clip.fps.denominator)*10)
    #     if lif is None: lif = min(35, gop)
    #     if tiles_row is None: tiles_row = math.floor(clip.height/1080)
    #     if tiles_col is None: tiles_col = math.floor(clip.width/1920)

    #     if enable_restoration is None: 
    #         if (clip.height*clip.width >= 3200*2000): # if smaller than 2160p
    #             enable_restoration = True
    #         else: enable_restoration = False

    #     aom_params = f"enable-qm=1:qm-min=5"

    #     if (clip.height*clip.width < 1280*720):
    #         aom_params += ":max-partition-size=64:sb-size=32"
    #     elif (clip.height*clip.width < 1920*1080): # if smaller than 1080p
    #         aom_params += ":max-partition-size=64:sb-size=64"

    #     ffmpeg_str = f"ffmpeg -y -hide_banner -v 8 -i - -c libaom-av1 -cpu-used {speed} -crf {quantizer} -g {gop} -lag-in-frames {lif} -tile-columns {tiles_col} -tile-rows {tiles_row} -enable-cdef {enable_cdef} -enable-restoration {enable_restoration} -arnr-strength {arnr_strength} -arnr-max-frames {arnr_max_frames} -aom-params \"{aom_params}\" -f {mux} -"

    #     file = open(file_str, 'wb')
    #     with subprocess.Popen(ffmpeg_str, stdin=subprocess.PIPE, stdout=file) as process:
    #         clip.output(process.stdin, y4m=True, progress_update=output.__get_progress__)
    #         process.stdin.close()
    #     file.close()

    def svt(clip:vs.VideoNode, file_str:str, speed:int=6, quantizer:int=32,
            gop:int=None, lad:int=None, tiles_row:int=None, tiles_col:int=None,
            sc_detection:bool=False, mux:str='nut', exec:str='ffmpeg'):

        if clip.format.name not in ['YUV420P8', 'YUV420P10']:
            raise vs.Error('Pixel format must be one of `YUV420P8 YUV420P10`')

        if gop is None: gop = min(300, round(clip.fps.numerator/clip.fps.denominator)*10)
        if lad is None: lad = min(120, gop)
        if tiles_row is None: tiles_row = math.floor(clip.height/1080)
        if tiles_col is None: tiles_col = math.floor(clip.width/1920)

        args = [
            exec,
            "-y",
            "-hide_banner",
            "-v", "8",
            "-i", "-",
            "-c", "libsvtav1",
            "-preset", f"{speed}",
            "-rc", "cqp",
            "-qp", f"{quantizer}",
            "-g", f"{gop}",
            "-la_depth", f"{lad}",
            "-tile_rows", f"{tiles_row}",
            "-tile_columns", f"{tiles_col}",
            "-sc_detection", f"{sc_detection}",
            "-f", f"{mux}",
            "-"
        ]

        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True, progress_update=output.__get_progress__)
            process.stdin.close()
        file.close()

    def libx264(clip:vs.VideoNode, file_str:str, preset:str='veryslow', crf:int=7, 
            crf_max:int=None, gop:int=None, threads:int=3, 
            rc_lookahead:int=None, mux:str='nut', exec:str='ffmpeg'):

        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10']:
            raise vs.Error('Pixel format must be one of `YUV420P8 YUV420P10`')

        args = [
            exec,
            "-y",
            "-hide_banner",
            "-v", "8",
            "-i", "-",
            "-c", "libx264",
            "-preset", f"{preset}",
            "-crf", f"{crf}"
        ]

        if gop is not None: args += ["-g", f"{gop}"]
        if crf_max is not None: ffmpeg_str += ["-crf_max", f"{crf_max}"]

        args += [
            "-threads", f"{threads}",
            "-f", f"{mux}",
            "-"
        ]

        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True, progress_update=output.__get_progress__)
            process.stdin.close()
        file.close()

    def llx264(clip:vs.VideoNode, file_str:str, preset:str='veryslow', mux:str='nut', exec:str='ffmpeg'):

        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10']:
            raise vs.Error('Pixel format must be one of `YUV420P8 YUV420P10`')

        args = [
            exec,
            "-y",
            "-hide_banner",
            "-v", "8",
            "-i", "-",
            "-c", "libx264",
            "-preset", f"{preset}",
            "-qp", "0",
            "-f", f"{mux}",
            "-"
        ]

        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True, progress_update=output.__get_progress__)
            process.stdin.close()
        file.close()

    def ffv1(clip:vs.VideoNode, file_str:str, mux:str='nut', exec:str='ffmpeg'):
        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10', 'YUV420P12', 'YUV422P12', 'YUV444P12', 'YUV420P16', 'YUV422P16', 'YUV444P16']:
            raise vs.Error('Pixel format must be one of `YUV420P8 YUV422P8 YUV444P8 YUV420P10 YUV422P10 YUV444P10 YUV420P12 YUV422P12 YUV444P12 YUV420P16 YUV422P16 YUV444P16`')

        args = [
            exec,
            "-y",
            "-hide_banner",
            "-v", "8",
            "-i", "-",
            "-c", "ffv1",
            "-f", f"{mux}",
            "-"
        ]

        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True, progress_update=output.__get_progress__)
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
        resize_kernel:str="didée") -> vs.VideoNode:

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

    if resize_kernel == "didée":
        # https://forum.doom9.org/showthread.php?p=1748922#post1748922
        clip = vs.core.resize.Bicubic(clip=clip, height=new_height, width=new_width, filter_param_a=-1/2, filter_param_b=1/4)
    else:
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

def down_to_444(clip:vs.VideoNode, width:Optional[int]=None, height:Optional[int]=None,
                resize_kernel_y="Spline36", resize_kernel_uv="Spline16") -> vs.VideoNode:
    # 4k 420 -> 1080p 444

    if source.format.color_family != vs.YUV:
        raise vs.Error('down_to_444: only YUV format is supported')

    if width is None: width = clip.width/2
    if height is None: height = clip.height/2

    resize_func_y  = getattr(vs.core.resize, resize_kernel_y)
    resize_func_uv = getattr(vs.core.resize, resize_kernel_uv)

    y = vs.core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
    y = resize_func_y(clip=y, height=height, width=width)

    u = vs.core.std.ShufflePlanes(clip, planes=1, colorfamily=vs.GRAY)
    u = resize_func_uv(clip=u, height=height, width=width)
    
    v = vs.core.std.ShufflePlanes(clip, planes=2, colorfamily=vs.GRAY)
    v = resize_func_uv(clip=v, height=height, width=width)

    clip = vs.core.std.ShufflePlanes(clips=[y,u,v], planes=[0,0,0], colorfamily=vs.YUV)
    return clip

# def down_to_444(clip:vs.VideoNode) -> int:
#     (y, u, v) = util.split(clip)
#     y = vs.core.resize.Bilinear(y, width=clip.width/2, height=clip.height/2)
#     clip = util.join((y, u, v), vs.YUV)
#     return clip

def bt2390_ictcp(clip:vs.VideoNode, source_peak:Optional[int]=None,
        target_nits:float=1) -> vs.VideoNode:
    # Stolen from somewhere, idk where.
    # clip = bt2390-vs.bt2390_ictcp(clip,target_nits=100,source_peak=1000)

    # TODO: Rewrite this mess, Fix var names names.

    if source_peak is None:
        source_peak = clip.get_frame(0).props.MasteringDisplayMaxLuminance

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

    clip = vs.core.resize.Spline36(clip=clip, format=vs.YUV444PS, filter_param_a=0, width=width_n, height=height_n, filter_param_b=0.75, chromaloc_in_s="center", chromaloc_s="center", range_in_s="limited", range_s="full", dither_type="none")

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

    c_ictcp = vs.core.resize.Spline36(clip=clip, format=vs.YUV444PS, filter_param_a=0, filter_param_b=0.75, chromaloc_in_s="center", transfer_in_s=transfer_in_s, chromaloc_s="center", range_in_s="full",range_s="full", dither_type="none", nominal_luminance=source_peak, matrix_in_s=matrix_in_s, matrix_s="ictcp")

    luma_intensity = vs.core.std.ShufflePlanes(c_ictcp, planes=[0], colorfamily=vs.GRAY)
    chroma_tritanopia = vs.core.std.ShufflePlanes(c_ictcp, planes=[1], colorfamily=vs.GRAY)
    chroma_protanopia = vs.core.std.ShufflePlanes(c_ictcp, planes=[2], colorfamily=vs.GRAY)

    luma_intensity = vs.core.std.Limiter(luma_intensity, 0, luma_w, planes=[0])      

    clip = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, filter_param_a=0, filter_param_b=0.75, chromaloc_in_s="center", chromaloc_s="center", range_in_s="full", range_s="full", dither_type="none", matrix_in_s="2020ncl")
    clip = vs.core.std.Limiter(clip, 0, luma_w) 

    cs = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, transfer_in_s=transfer_in_s, transfer_s="linear", dither_type="none", nominal_luminance=source_peak)

    cs = vs.core.std.Expr(clips=[cs], expr=f"x {exposure_bias1} *") 

    r = vs.core.std.ShufflePlanes(cs, planes=[0], colorfamily=vs.GRAY)
    g = vs.core.std.ShufflePlanes(cs, planes=[1], colorfamily=vs.GRAY)
    b = vs.core.std.ShufflePlanes(cs, planes=[2], colorfamily=vs.GRAY)
    max = vs.core.std.Expr(clips=[r,g,b], expr="x y max z max") 
    min = vs.core.std.Expr(clips=[r,g,b], expr="x y min z min")
    sat = vs.core.std.Expr(clips=[r,g,b], expr="x x * y y * + z z * + x y + z + /")       
    l = vs.core.std.Expr(clips=[r,g,b], expr="0.2627 x * 0.6780 y * + 0.0593 z * +") 
    l = vs.core.std.ShufflePlanes(clips=[l,l,l], planes=[0,0,0], colorfamily=vs.RGB)      

    saturation_mult1 = vs.core.std.Expr(clips=[sat], expr=f"x 1 - {exposure_bias1} 1 - /")
    saturation_mult1 = vs.core.std.Limiter(saturation_mult1, 0, 1)    

    c1 = vs.core.std.MaskedMerge(cs, l, saturation_mult1)
    clip = vs.core.std.Merge(cs, c1, 0.5)

    clip = vs.core.std.Expr(clips=[clip], expr=f"x {exposure_bias1} /") 

    clip = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, transfer_in_s="linear", transfer_s=transfer_in_s, dither_type="none", nominal_luminance=source_peak,cpu_type=None)

    e1 = vs.core.std.Expr(clips=[clip], expr=f"x  {luma_w} /")
    t = vs.core.std.Expr(clips=[e1], expr=f"x {ks} - 1 {ks} - /")    
    p = vs.core.std.Expr(clips=[t], expr=f"2 x 3 pow * 3 x 2 pow * - 1 + {ks} * 1 {ks} - x 3 pow 2 x 2 pow * - x + * + -2 x 3 pow * 3 x 2 pow * + {max_luma} * +")    
    e2 = vs.core.std.Expr(clips=[e1,p], expr=f"x {ks} < x y ?")
    crgb = vs.core.std.Expr(clips=[e2], expr=f"x {luma_w} *")   
    crgb = vs.core.std.Limiter(crgb, 0, 1)  

    rgb = crgb

    crgb = vs.core.resize.Spline36(clip=crgb, format=vs.YUV444PS, filter_param_a=0, filter_param_b=0.75, chromaloc_in_s="center", transfer_in_s=transfer_in_s, transfer_s=transfer_in_s, chromaloc_s="center", range_in_s="full", range_s="full", dither_type="none", nominal_luminance=target_nits, matrix_s="ictcp",cpu_type=None)

    Irgb = vs.core.std.ShufflePlanes(crgb, planes=[0], colorfamily=vs.GRAY)

    saturation_mult1 = vs.core.std.Expr(clips=[saturation_mult1], expr=f"1 x -")

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

    # clip = vs.core.resize.Spline36(clip=clip, format=vs.RGBS, transfer_in_s="linear", transfer_s="709", dither_type="none")
    clip = vs.core.std.Limiter(clip, 0, 1)

    clip = vs.core.resize.Spline36(clip=clip, format=vs.YUV444P16, matrix_s="709", filter_param_a=0, filter_param_b=0.75, range_in_s="full", range_s="limited", chromaloc_in_s="center", chromaloc_s="center", dither_type="none")
    return clip

def imwri_src(dir:str, fpsnum:int, fpsden:int, firstnum:int=0, alpha:bool=False) -> vs.VideoNode:

    srcs = [dir + src for src in os.listdir(dir)]
    clip = vs.core.imwri.Read(srcs, firstnum=firstnum, alpha=alpha)
    clip = vs.core.std.AssumeFPS(clip=clip, fpsnum=fpsnum, fpsden=fpsden)
    return clip

def mv_scene_detection(clip:vs.VideoNode, preset:str='fast', super_pel:int=2,

        thscd1:int=140, thscd2:int=15,

        overlap:int=0, overlapv:Optional[int]=None, search:Optional[int]=None,
        dct:int=0, truemotion:bool=True, blksize:int=8, blksizev:int=8,
        searchparam:int=2, badSAD:int=10000, badrange:int=24, divide:int=0) -> vs.VideoNode:
    # mvtools scene detection

    # thSCD1 (int): threshold which decides whether a block has changed between
    # the previous frame and the current one. When a block has changed,
    # it means that motion estimation for it isn't relevant.
    # It occurs for example at scene changes. So it is one of the thresholds
    # used to tweak the scene changes detection engine. Raising it will lower
    # the number of blocks detected as changed. It may be useful for noisy or
    # flickered video. The threshold is compared to the SAD
    # (Sum of Absolute Differences, a value which says how bad the motion
    # estimation was ) value. For exactly identical blocks we have SAD=0.
    # But real blocks are always different because of objects complex movement
    # (zoom, rotation, deformation), discrete pixels sampling, and noise.
    # Suppose we have two compared 8x8 blocks with every pixel different by 5.
    # It this case SAD will be 8x8x5 = 320 (block will not detected
    # as changed for thSCD1=400). If you use 4x4 blocks,
    # SAD will be 320/4. If you use 16x16 blocks, SAD will be 320*4.
    # Really this parameter is scaled internally in MVTools, and you must
    # always use reduced to block size 8x8 value. Default is 400 (since v.1.4.1).

    # thSCD2 (int): threshold which sets how many blocks have to change
    # for the frame to be considered as a scene change.
    # It is ranged from 0 to 255, 0 meaning 0 %, 255 meaning 100 %.
    # Default is 130 ( which means 51 % ). 

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

    _fun_super       = vs.core.mvsf.Super       if clip.format.sample_type == vs.FLOAT else vs.core.mv.Super
    _fun_analyse     = vs.core.mvsf.Analyse     if clip.format.sample_type == vs.FLOAT else vs.core.mv.Analyse
    _fun_scdetection = vs.core.mvsf.SCDetection if clip.format.sample_type == vs.FLOAT else vs.core.mv.SCDetection

    mvsuper = _fun_super(clip, pel=super_pel, sharp=2, rfilter=4)
    vectors = _fun_analyse(mvsuper, isb=True, **analyse_params)

    clip = _fun_scdetection(clip, vectors=vectors, thscd1=thscd1, thscd2=int(thscd2*255/100))
    return clip

def mv_motion_interpolation(clip:vs.VideoNode, fpsnum:int=60, fpsden:int=1, preset:str='fast', 
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

    _fun_super       = vs.core.mvsf.Super       if clip.format.sample_type == vs.FLOAT else vs.core.mv.Super
    _fun_analyse     = vs.core.mvsf.Analyse     if clip.format.sample_type == vs.FLOAT else vs.core.mv.Analyse
    _fun_flowblur    = vs.core.mvsf.FlowBlur    if clip.format.sample_type == vs.FLOAT else vs.core.mv.FlowBlur
    _fun_blockfps    = vs.core.mvsf.BlockFPS    if clip.format.sample_type == vs.FLOAT else vs.core.mv.BlockFPS
    _fun_flowfps     = vs.core.mvsf.FlowFPS     if clip.format.sample_type == vs.FLOAT else vs.core.mv.FlowFPS

    mvsuper = _fun_super(clip, pel=super_pel, sharp=2, rfilter=4)
    block_vectors = _fun_analyse(mvsuper, isb=True, **analyse_params)
    flow_vectors = _fun_analyse(mvsuper, isb=False, **analyse_params)

    if clip.fps.numerator/clip.fps.denominator > fpsnum/fpsden:
        clip = _fun_flowblur(clip, mvsuper, block_vectors, flow_vectors, blur=Mblur)

    if block == True:
        out = _fun_blockfps(clip, mvsuper, block_vectors, flow_vectors, **block_or_flow_params, mode=block_mode)
    else:
        out = _fun_flowfps(clip, mvsuper, block_vectors, flow_vectors, **block_or_flow_params, mask=flow_mask)
    return out

def adaptive_noise(clip:vs.VideoNode, strength:float=0.25, static:bool=True,
        luma_scaling:float=12.0, show_mask:bool=False, noise_type:int=2) -> vs.VideoNode:
    # Based on kagefunc's
    # https://kageru.moe/blog/article/adaptivegrain
    # uses https://github.com/wwww-wwww/vs-noise

    mask = vs.core.adg.Mask(clip.std.PlaneStats(), luma_scaling)
    grained = vs.core.noise.Add(clip, var=strength, constant=static, type=noise_type)
    if show_mask:
        return mask

    return vs.core.std.MaskedMerge(clip, grained, mask)

def scale(i:int, depth_out:int=16, depth_in:int=8) -> int:
    # converting the values in one depth to another
    return i*2**(depth_out-depth_in)

def mv_flux_smooth(clip:vs.VideoNode, temporal_threshold:int=12, 
    super_params:dict={}, analyse_params:dict={}, compensate_params:dict={},
    planes=[0,1,2]) -> vs.VideoNode:
    # port from https://forum.doom9.org/showthread.php?s=d58237a359f5b1f2ea45591cceea5133&p=1572664#post1572664
    # allow setting parameters for mvtools

    _fun_super       = vs.core.mvsf.Super       if clip.format.sample_type == vs.FLOAT else vs.core.mv.Super
    _fun_analyse     = vs.core.mvsf.Analyse     if clip.format.sample_type == vs.FLOAT else vs.core.mv.Analyse
    _fun_flowblur    = vs.core.mvsf.FlowBlur    if clip.format.sample_type == vs.FLOAT else vs.core.mv.FlowBlur
    _fun_blockfps    = vs.core.mvsf.BlockFPS    if clip.format.sample_type == vs.FLOAT else vs.core.mv.BlockFPS
    _fun_flowfps     = vs.core.mvsf.FlowFPS     if clip.format.sample_type == vs.FLOAT else vs.core.mv.FlowFPS
    _fun_compensate  = vs.core.mvsf.Compensate     if clip.format.sample_type == vs.FLOAT else vs.core.mv.Compensate
    # _fun_recalculate = vs.core.mvsf.Recalculate if clip.format.sample_type == vs.FLOAT else vs.core.mv.Recalculate
    # _fun_degrain1    = vs.core.mvsf.Degrain1    if clip.format.sample_type == vs.FLOAT else vs.core.mv.Degrain1
    # _fun_degrain2    = vs.core.mvsf.Degrain2    if clip.format.sample_type == vs.FLOAT else vs.core.mv.Degrain2
    # _fun_degrain3    = vs.core.mvsf.Degrain3    if clip.format.sample_type == vs.FLOAT else vs.core.mv.Degrain3

    super_params = {"pel":2, "sharp":1, **super_params}
    analyse_params = {"truemotion":False, "delta":1, "blksize":16, "overlap":8, **analyse_params}
    mvsuper = _fun_super(clip, **super_params)
    mvanalyset = _fun_analyse(mvsuper, isb=True, **analyse_params)
    mvanalysef = _fun_analyse(mvsuper, isb=False, **analyse_params)
    mvcompensavet = _fun_compensate(clip, mvsuper, mvanalyset, **compensate_params)
    mvcompensavef = _fun_compensate(clip, mvsuper, mvanalysef, **compensate_params)
    mvinterleave = vs.core.std.Interleave([mvcompensavef, clip, mvcompensavet])
    fluxsmooth = vs.core.flux.SmoothT(mvinterleave, temporal_threshold=temporal_threshold, planes=planes)
    return vs.core.std.SelectEvery(fluxsmooth, 3, 1)

def STPressoMC(clip:vs.VideoNode, limit:int=3, bias:int=24, RGVS_mode:int=4,
    temporal_threshold:int=12, temporal_limit:int=3, temporal_bias:int=49, back:int=1,
    super_params:dict={}, analyse_params:dict={}, compensate_params:dict={}) -> vs.VideoNode:
    # orginal script by Didée, taken from xvs

    # The goal of STPressoMC (Spatio-Temporal Pressdown using Motion Compensation) is
    # to "dampen the grain just a little, to keep the original look,
    # and make it fast". In other words it makes a video more
    # compressible without losing detail and original grain structure.

    # limit = 3     Spatial limit: the spatial part won't change a pixel more than this. 
    # bias = 24     The percentage of the spatial filter that will apply. 
    # RGVS_mode = 4 The spatial filter is RemoveGrain, this is its mode. 
    # temporal_threshold = 12  Temporal threshold for FluxSmooth; Can be set "a good bit bigger" than usually. 
    # temporal_limit = 3  The temporal filter won't change a pixel more than this. 
    # temporal_bias = 49  The percentage of the temporal filter that will apply. 
    # back = 1  After all changes have been calculated, reduce all pixel changes by this value. (Shift "back" towards original value) 

    # STPresso is recommended for content up to 720p because
    # "the spatial part might be a bit too narrow for 1080p encoding
    # (since it's only a 3x3 kernel)". 

    # Differences:
    # high depth support
    # automatically adjust parameters to fit into different depth
    # you have less choice in RGVS_mode

    depth = clip.format.bits_per_sample
    LIM1 = round(limit*100.0/bias-1.0) if limit > 0 else round(100.0/bias)
    LIM1 = scale(LIM1,depth)
    #(limit>0) ? round(limit*100.0/bias-1.0) :  round(100.0/bias)
    LIM2 = 1 if limit < 0 else limit
    LIM2 = scale(LIM2,depth)
    #(limit<0) ? 1 : limit
    BIA = bias
    BK = scale(back,depth)
    TBIA = bias
    TLIM1 = round(temporal_limit*100.0/temporal_bias-1.0) if temporal_limit > 0 else round(100.0/temporal_bias)
    TLIM1 = scale(TLIM1,depth)
    #(temporal_limit>0) ? string( round(temporal_limit*100.0/temporal_bias-1.0) ) : string( round(100.0/temporal_bias) )
    TLIM2  = 1 if temporal_limit < 0 else temporal_limit
    TLIM2 = scale(TLIM2,depth)
    #(temporal_limit<0) ? "1" : string(temporal_limit)
    clip_rgvs = vs.core.rgvs.RemoveGrain(clip,RGVS_mode)
    ####
    if limit < 0:
        expr  = f"x y - abs {LIM1} < x x {scale(1, depth)} x y - x y - abs / * - ?"
        texpr = f"x y - abs {TLIM1} < x x {scale(1, depth)} x y - x y - abs / * - ?"
    else:
        expr  =  f"x y - abs {scale(1, depth)} < x x {LIM1} + y < x {LIM2} + x {LIM1} - y > x {LIM2} - x {scale(100, depth)} {BIA} - * y {BIA} * + {scale(100, depth)} / ? ? ?"
        texpr =  f"x y - abs {scale(1, depth)} < x x {TLIM1} + y < x {TLIM2} + x {TLIM1} - y > x {TLIM2} - x {scale(100, depth)} {TBIA} - * y {TBIA} * + {scale(100, depth)} / ? ? ?"
    L=[]
    for plane in range(0,3):
        C = vs.core.std.ShufflePlanes(clip, plane, colorfamily=vs.GRAY)
        B = vs.core.std.ShufflePlanes(clip_rgvs, plane, colorfamily=vs.GRAY)
        O = vs.core.std.Expr([C,B],expr)
        L.append(O)
    if temporal_threshold != 0:
        st = mv_flux_smooth(clip_rgvs, temporal_threshold, super_params, analyse_params, compensate_params, [0,1,2])
        diff = vs.core.std.MakeDiff(clip_rgvs, st, [0,1,2])
        last = vs.core.std.ShufflePlanes(L, [0,0,0], colorfamily=vs.YUV)
        diff2 = vs.core.std.MakeDiff(last,diff, [0,1,2])
        for i in range(0,3):
            c = L[i]
            b = vs.core.std.ShufflePlanes(diff2, i, colorfamily=vs.GRAY)
            L[i] = vs.core.std.Expr([c,b],texpr)
    if back != 0:
        bexpr = f"x {BK} + y < x {BK} + x {BK} - y > x {BK} - y ? ?"
        Y = vs.core.std.ShufflePlanes(clip, 0, colorfamily=vs.GRAY)
        L[0] = vs.core.std.Expr([L[0],Y], bexpr)
    output = vs.core.std.ShufflePlanes(L, [0,0,0], colorfamily=vs.YUV)
    return output
