import vapoursynth as vs
import math
import os
import subprocess
from functools import partial
from typing import Optional, Union, List, NamedTuple, Literal
# https://peps.python.org/pep-0483/#fundamental-building-blocks

# Requriements: imwri, mvtools, fmtc, rgvs, rgsf, flux,
#               ctmf, vs-noise, adaptivegrain, znedi3,
#               tcanny, neo_f3kdb

# Docs
# https://github.com/vapoursynth/vs-imwri/blob/master/docs/imwri.rst
# http://avisynth.org.ru/mvtools/mvtools2.html
# https://github.com/dubhater/vapoursynth-mvtools
# https://amusementclub.github.io/fmtconv/doc/fmtconv.html
# https://github.com/dubhater/vapoursynth-fluxsmooth
# https://github.com/IFeelBloated/RGSF
# https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF
# https://github.com/wwww-wwww/vs-noise
# https://git.kageru.moe/kageru/adaptivegrain
# https://github.com/sekrit-twc/znedi3
# https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny
# https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb
# https://f3kdb.readthedocs.io/en/stable/usage.html

class helper:

    def round_to_closest(x:Union[int, float]) -> int:
        # I'm amazed that's not a thing.

        if x - math.floor(x) < 0.5:
            return math.floor(x)
        return math.ceil(x)

    def scale_depth(i:int, depth_out:int=16, depth_in:int=8) -> int:
        # converting the values in one depth to another

        return i*2**(depth_out-depth_in)

    def scale255(value:int, peak:int) -> int:

        return helper.round_to_closest(peak * value/255)

    def mod_m(x:Union[int, float], m:Union[int, float]=4.0) -> Union[int, float]:

        return 16 if x < 16 else math.floor(x / m) * m

    def clamp(minimum:Union[int, float], x:Union[int, float], maximum:Union[int, float]) -> int:

        return helper.round_to_closest(max(minimum, min(x, maximum)))

class util:

    def fmtc_kernel_kwargs(kernel:str, taps:Optional[int]=None):

        kernel = kernel.lower()
        if kernel in ["point", "nearest", "neighbour"]:
            kernel_kwargs = {"kernel": "point"}
        # most of those are from lvsfunc
        elif kernel in ["rect", "box", "linear", "bilinear", "blackman", 
                        "blackmanminlobe", "spline", "sinc", "spline16",
                        "spline36", "spline64"]:
            kernel_kwargs = {"kernel": kernel}
        elif kernel == "lanczos":
            taps = 3 if taps is None else taps
            kernel_kwargs = {"kernel": "lanczos", "taps": taps}
        elif kernel == "spline144":
            kernel_kwargs = {"kernel": "spline", "taps": 6}
        # Bicubics
        elif kernel in ["didée", "didee"]:
            kernel_kwargs = {"kernel": "bicubic", "a1": -1/2, "a2": 1/4}
        elif kernel in ["mitchell", "mitchell-netravali"]:
            kernel_kwargs = {"kernel": "bicubic", "a1": 1/3, "a2": 1/3}
        elif kernel in ["catrom", "catmullrom", "catmull-rom"]:
            kernel_kwargs = {"kernel": "bicubic", "a1": 0, "a2": 1/2}
        elif kernel == "bicubicsharp":
            kernel_kwargs = {"kernel": "bicubic", "a1": 0, "a2": 1}
        elif kernel == "robidouxsoft":
            x = (9 - 3 * math.sqrt(2)) / 7
            kernel_kwargs = {"kernel": "bicubic",
                            "a1": x, 
                            "a2": (1 - x) / 2}
        elif kernel == "robidoux":
            kernel_kwargs = {"kernel": "bicubic",   
                            "a1": 12 / (19 + 9 * math.sqrt(2)), 
                            "a2": 113 / (58 + 216 * math.sqrt(2))}
        elif kernel == "robidouxsharp":
            kernel_kwargs = {"kernel": "bicubic",   
                            "a1": 6 / (13 + 7 * math.sqrt(2)),
                            "a2": 7 / (2 + 12 * math.sqrt(2))}
        elif kernel == "bspline":
            kernel_kwargs = {"kernel": "bicubic", "a1": 1, "a2": 0}
        else:
            raise vs.Error("fmtc_kernel_kwargs: unkown kernel")
        return kernel_kwargs

    # stolen from vsutil btw.

    def plane(clip:vs.VideoNode, plane:int) -> vs.VideoNode:

        if clip.format.num_planes == 1 and plane == 0:
            return clip
        return vs.core.std.ShufflePlanes(clip, plane, vs.GRAY)

    def join(planes:List[vs.VideoNode], family:vs.ColorFamily=vs.YUV) -> vs.VideoNode:

        if family not in [vs.RGB, vs.YUV]:
            raise vs.Error('Color family must have 3 planes.')
        return vs.core.std.ShufflePlanes(clips=planes, planes=[0, 0, 0], colorfamily=family)

    def split(clip:vs.VideoNode) -> List[vs.VideoNode]:

        return [util.plane(clip, x) for x in range(clip.format.num_planes)]

    def edge_detection(clip:vs.VideoNode, planes:Optional[Union[int,List[int]]]=None,
        method:str="kirsch", thr:Optional[int]=None, scale:bool=False) -> vs.VideoNode:

        if planes is None:
            planes = list(range(clip.format.num_planes))
        elif isinstance(planes, int):
            planes = [planes]

        max_value = 1 if clip.format.sample_type == vs.FLOAT else (1 << clip.format.bits_per_sample) - 1

        if method in ["t", "tcanny"]:
            # Fastest
            if thr is None: thr = 5
            edge = vs.core.tcanny.TCanny(clip, mode=1, sigma=1)
            # edge = vs.core.tcanny.TCanny(clip, sigma=1.25, t_h=4.0, t_l=1.0, op=1, mode=1) # Blue

        elif method in ["p", "prewitt"]:
            # Best??
            if thr is None: thr = 15
            edge = vs.core.std.Expr([
                clip.std.Convolution(matrix=[1, 1, 0, 1, 0, -1, 0, -1, -1], planes=planes, saturate=False),
                clip.std.Convolution(matrix=[1, 1, 1, 0, 0, 0, -1, -1, -1], planes=planes, saturate=False),
                clip.std.Convolution(matrix=[1, 0, -1, 1, 0, -1, 1, 0, -1], planes=planes, saturate=False),
                clip.std.Convolution(matrix=[0, -1, -1, 1, 0, -1, 1, 1, 0], planes=planes, saturate=False)],
                expr=['x y max z max a max' if i in planes else '' for i in range(clip.format.num_planes)])

        elif method in ["k", "kirsch"]:
            # Slowest and maybe most chaotic
            if thr is None: thr = 40
            edge = vs.core.std.Expr([
                clip.std.Convolution(matrix=[ 5,  5,  5, -3,  0, -3, -3, -3, -3], planes=planes, saturate=False),
                clip.std.Convolution(matrix=[-3,  5,  5, -3,  0,  5, -3, -3, -3], planes=planes, saturate=False),
                clip.std.Convolution(matrix=[-3, -3,  5, -3,  0,  5, -3, -3,  5], planes=planes, saturate=False),
                clip.std.Convolution(matrix=[-3, -3, -3, -3,  0,  5, -3,  5,  5], planes=planes, saturate=False)],
                expr=['x y max z max a max' if i in planes else '' for i in range(clip.format.num_planes)])
        else:
            raise ValueError("edge_detection: invalid method.")

        if scale:
            edge = vs.core.std.Expr([edge], expr=f'x {helper.scale255(thr, max_value)} < 0 x ?')

        return edge

    def retinex_edge(clip:vs.VideoNode, method="prewitt", sigma:int=1,
        thr:Optional[int]=None, plane:int=0) -> vs.VideoNode:
        # from kageunc

        plane_clip = util.plane(clip, plane=plane)
        max_value = 1 if clip.format.sample_type == vs.FLOAT else (1 << clip.format.bits_per_sample) - 1
        ret = vs.core.retinex.MSRCP(plane_clip, sigma=[50, 200, 350], upper_thr=0.005)
        tcanny = ret.tcanny.TCanny(mode=1, sigma=sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
        return vs.core.std.Expr([util.edge_detection(plane_clip, method=method, thr=thr), tcanny], f'x y + {max_value} min')

    def mt_expand_multi(clip:vs.VideoNode, mode:str='rectangle', 
        planes:Optional[Union[int, List[int]]]=None, sw:int=1, sh:int=1) -> vs.VideoNode:

        if isinstance(planes, int):
            planes = [planes]

        if sw > 0 and sh > 0:
            mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
        elif sw > 0:
            mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
        elif sh > 0:
            mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
        else:
            mode_m = None

        if mode_m is not None:
            clip = util.mt_expand_multi(clip.std.Maximum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
        return clip

    def mt_inpand_multi(clip:vs.VideoNode, mode:str='rectangle', 
        planes:Optional[Union[int, List[int]]]=None, sw:int=1, sh:int=1) -> vs.VideoNode:

        if isinstance(planes, int):
            planes = [planes]

        if sw > 0 and sh > 0:
            mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
        elif sw > 0:
            mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
        elif sh > 0:
            mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
        else:
            mode_m = None

        if mode_m is not None:
            clip = util.mt_inpand_multi(clip.std.Minimum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
        return clip

    def mt_inflate_multi(clip:vs.VideoNode, planes:Optional[Union[int, List[int]]]=None, radius:int=1) -> vs.VideoNode:

        if isinstance(planes, int):
            planes = [planes]

        for i in range(radius):
            clip = vs.core.std.Inflate(clip, planes=planes)
        return clip

    def mt_deflate_multi(clip:vs.VideoNode, planes:Optional[Union[int, List[int]]]=None, radius:int=1) -> vs.VideoNode:

        if isinstance(planes, int):
            planes = [planes]

        for i in range(radius):
            clip = vs.core.std.Deflate(clip, planes=planes)
        return clip

    def check_color_family(color_family:str, valid_list:List[str]=None, invalid_list:Optional[List[str]]=None) -> vs.VideoNode:

        if valid_list is None:
            valid_list = ('RGB', 'YUV', 'GRAY')
        if invalid_list is None:
            invalid_list = ('COMPAT', 'UNDEFINED')
        # check invalid list
        for cf in invalid_list:
            if color_family == getattr(vs, cf, None):
                raise value_error(f'color family *{cf}* is not supported!')
        # check valid list
        if valid_list:
            if color_family not in [getattr(vs, cf, None) for cf in valid_list]:
                raise value_error(f'color family not supported, only {valid_list} are accepted')

    def min_blur(clip:vs.VideoNode, r:int=1, planes:Optional[Union[int, List[int]]]=None) -> vs.VideoNode:
        # by Didée (http://avisynth.nl/index.php/MinBlur)
        # Nifty Gauss/Median combination

        if planes is None:
            planes = list(range(clip.format.num_planes))
        elif isinstance(planes, int):
            planes = [planes]

        matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
        matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        if r <= 0:
            RG11 = sbr(clip, planes=planes)
            RG4 = clip.std.Median(planes=planes)
        elif r == 1:
            RG11 = clip.std.Convolution(matrix=matrix1, planes=planes)
            RG4 = clip.std.Median(planes=planes)
        elif r == 2:
            RG11 = clip.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
            RG4 = clip.ctmf.CTMF(radius=2, planes=planes)
        else:
            RG11 = clip.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
            if clip.format.bits_per_sample == 16:
                s16 = clip
                RG4 = clip.fmtc.bitdepth(bits=12, planes=planes, dmode=1).ctmf.CTMF(radius=3, planes=planes).fmtc.bitdepth(bits=16, planes=planes)
                RG4 = util.limit_filter(s16, RG4, thr=0.0625, elast=2, planes=planes)
            else:
                RG4 = clip.ctmf.CTMF(radius=3, planes=planes)

        expr = 'x y - x z - * 0 < x x y - abs x z - abs < y z ? ?'
        return vs.core.std.Expr([clip, RG11, RG4], expr=[expr if i in planes else '' for i in range(clip.format.num_planes)])

    def limit_filter(flt:vs.VideoNode, src:vs.VideoNode, ref:Optional[vs.VideoNode]=None,
        thr:Optional[float]=None, elast:Optional[float]=None, planes:Optional[Union[int, List[int]]]=None, 
        brighten_thr:Optional[float]=None, thrc:Optional[float]=None, force_expr:bool=True) -> vs.VideoNode:
        # from mvsfunc

        # It acts as a post-processor, and is very useful to limit the difference of filtering while avoiding artifacts.
        # Commonly used cases:
        #     de-banding
        #     de-ringing
        #     de-noising
        #     sharpening
        #     combining high precision source with low precision filtering: util.limit_filter(src, flt, thr=1.0, elast=2.0)

        # There are 2 implementations, default one with std.Expr, the other with std.Lut.
        # The Expr version supports all mode, while the Lut version doesn't support float input and ref clip.
        # Also the Lut version will truncate the filtering diff if it exceeds half the value range(128 for 8-bit, 32768 for 16-bit).
        # The Lut version might be faster than Expr version in some cases, for example 8-bit input and brighten_thr != thr.

        # Basic parameters
        #     flt {clip}: filtered clip, to compute the filtering diff
        #         can be of YUV/RGB/Gray color family, can be of 8-16 bit integer or 16/32 bit float
        #     src {clip}: source clip, to apply the filtering diff
        #         must be of the same format and dimension as "flt"
        #     ref {clip} (optional): reference clip, to compute the weight to be applied on filtering diff
        #         must be of the same format and dimension as "flt"
        #         default: None (use "src")
        #     thr {float}: threshold (8-bit scale) to limit filtering diff
        #         default: 1.0
        #     elast {float}: elasticity of the soft threshold
        #         default: 2.0
        #     planes {int, int[]}: specify which planes to process
        #         unprocessed planes will be copied from "flt"
        #         default: all planes will be processed, [0,1,2] for YUV/RGB input, [0] for Gray input

        # Advanced parameters
        #     brighten_thr {float}: threshold (8-bit scale) for filtering diff that brightening the image (Y/R/G/B plane)
        #         set a value different from "thr" is useful to limit the overshoot/undershoot/blurring introduced in sharpening/de-ringing
        #         default is the same as "thr"
        #     thrc {float}: threshold (8-bit scale) for chroma (U/V/Co/Cg plane)
        #         default is the same as "thr"
        #     force_expr {bool}
        #         - True: force to use the std.Expr implementation
        #         - False: use the std.Lut implementation if available
        #         default: True

        def _limit_filter_expr(defref, thr, elast, largen_thr, value_range) -> str:

            flt = " x "
            src = " y "
            ref = " z " if defref else src
            
            dif = f" {flt} {src} - "
            dif_ref = f" {flt} {ref} - "
            dif_abs = dif_ref + " abs "
            
            thr = thr * value_range / 255
            largen_thr = largen_thr * value_range / 255
            
            if thr <= 0 and largen_thr <= 0:
                limitExpr = f" {src} "
            elif thr >= value_range and largen_thr >= value_range:
                limitExpr = ""
            else:
                if thr <= 0:
                    limitExpr = f" {src} "
                elif thr >= value_range:
                    limitExpr = f" {flt} "
                elif elast <= 1:
                    limitExpr = f" {dif_abs} {thr} <= {flt} {src} ? "
                else:
                    thr_1 = thr
                    thr_2 = thr * elast
                    thr_slope = 1 / (thr_2 - thr_1)
                    # final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
                    limitExpr = f" {src} {dif} {thr_2} {dif_abs} - * {thr_slope} * + "
                    limitExpr = f" {dif_abs} {thr_1} <= {flt} {dif_abs} {thr_2} >= {src} " + limitExpr + " ? ? "
                
                if largen_thr != thr:
                    if largen_thr <= 0:
                        limitExprLargen = f" {src} "
                    elif largen_thr >= value_range:
                        limitExprLargen = f" {flt} "
                    elif elast <= 1:
                        limitExprLargen = f" {dif_abs} {largen_thr} <= {flt} {src} ? "
                    else:
                        thr_1 = largen_thr
                        thr_2 = largen_thr * elast
                        thr_slope = 1 / (thr_2 - thr_1)
                        # final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
                        limitExprLargen = f" {src} {dif} {thr_2} {dif_abs} - * {thr_slope} * + "
                        limitExprLargen = f" {dif_abs} {thr_1} <= {flt} {dif_abs} {thr_2} >= {src} " + limitExprLargen + " ? ? "
                    limitExpr = f" {flt} {ref} > " + limitExprLargen + " " + limitExpr + " ? "
            
            return limitExpr

        def _limit_diff_lut(diff:vs.VideoNode, thr:float, elast:float,
            largen_thr:Union[int, float], planes:List[vs.VideoNode]):

            # Get properties of input clip
            sFormat = diff.format

            sSType = sFormat.sample_type
            sbitPS = sFormat.bits_per_sample

            if sSType == vs.INTEGER:
                neutral = 1 << (sbitPS - 1)
                value_range = (1 << sbitPS) - 1
            else:
                neutral = 0
                value_range = 1

            # Process
            thr = thr * value_range / 255
            largen_thr = largen_thr * value_range / 255

            if thr <= 0 and largen_thr <= 0:
                return diff
            elif thr >= value_range / 2 and largen_thr >= value_range / 2:
                def limitLut(x):
                    return neutral
                return vs.core.std.Lut(diff, planes=planes, function=limitLut)
            elif elast <= 1:
                def limitLut(x):
                    dif = x - neutral
                    dif_abs = abs(dif)
                    thr_1 = largen_thr if dif > 0 else thr
                    return neutral if dif_abs <= thr_1 else x
                return vs.core.std.Lut(diff, planes=planes, function=limitLut)
            else:
                def limitLut(x):
                    dif = x - neutral
                    dif_abs = abs(dif)
                    thr_1 = largen_thr if dif > 0 else thr
                    thr_2 = thr_1 * elast

                    if dif_abs <= thr_1:
                        return neutral
                    elif dif_abs >= thr_2:
                        return x
                    else:
                        # final = flt - dif * (dif_abs - thr_1) / (thr_2 - thr_1)
                        thr_slope = 1 / (thr_2 - thr_1)
                        return round(dif * (dif_abs - thr_1) * thr_slope + neutral)
                return vs.core.std.Lut(diff, planes=planes, function=limitLut)

        # Get properties of input clip
        sFormat = flt.format
        if sFormat.id != src.format.id:
            raise value_error('"flt" and "src" must be of the same format!')
        if flt.width != src.width or flt.height != src.height:
            raise value_error('"flt" and "src" must be of the same width and height!')

        if ref is not None:
            if sFormat.id != ref.format.id:
                raise value_error('"flt" and "ref" must be of the same format!')
            if flt.width != ref.width or flt.height != ref.height:
                raise value_error('"flt" and "ref" must be of the same width and height!')

        sColorFamily = sFormat.color_family
        util.check_color_family(sColorFamily)
        sIsYUV = sColorFamily == vs.YUV

        sSType = sFormat.sample_type
        sbitPS = sFormat.bits_per_sample
        sNumPlanes = sFormat.num_planes

        # Parameters
        if thr is None:
            thr = 1.0
        elif isinstance(thr, int) or isinstance(thr, float):
            if thr < 0:
                raise value_error('valid range of "thr" is [0, +inf)')
        else:
            raise type_error('"thr" must be an int or a float!')

        if elast is None:
            elast = 2.0
        elif isinstance(elast, int) or isinstance(elast, float):
            if elast < 1:
                raise value_error('valid range of "elast" is [1, +inf)')
        else:
            raise type_error('"elast" must be an int or a float!')

        if brighten_thr is None:
            brighten_thr = thr
        elif isinstance(brighten_thr, int) or isinstance(brighten_thr, float):
            if brighten_thr < 0:
                raise value_error('valid range of "brighten_thr" is [0, +inf)')
        else:
            raise type_error('"brighten_thr" must be an int or a float!')

        if thrc is None:
            thrc = thr
        elif isinstance(thrc, int) or isinstance(thrc, float):
            if thrc < 0:
                raise value_error('valid range of "thrc" is [0, +inf)')
        else:
            raise type_error('"thrc" must be an int or a float!')

        if ref is not None or sSType != vs.INTEGER:
            force_expr = True

        # planes
        process = [0 for i in range(3)]

        if planes is None:
            process = [1 for i in range(3)]
        elif isinstance(planes, int):
            if planes < 0 or planes >= 3:
                raise vs.Error(f'valid range of planes is 0 to 3')
            process[planes] = 1
        elif isinstance(planes, list):
            for p in planes:
                if p < 0 or p >= 3:
                    raise vs.Error(f'valid range of planes is [0, 3]!')
                process[p] = 1

        # Process
        if thr <= 0 and brighten_thr <= 0:
            if sIsYUV:
                if thrc <= 0:
                    return src
            else:
                return src
        if thr >= 255 and brighten_thr >= 255:
            if sIsYUV:
                if thrc >= 255:
                    return flt
            else:
                return flt
        if thr >= 128 or brighten_thr >= 128:
            force_expr = True

        if force_expr: # implementation with std.Expr
            valueRange = (1 << sbitPS) - 1 if sSType == vs.INTEGER else 1
            limitExprY = _limit_filter_expr(ref is not None, thr, elast, brighten_thr, valueRange)
            limitExprC = _limit_filter_expr(ref is not None, thrc, elast, thrc, valueRange)
            expr = []
            for i in range(sNumPlanes):
                if process[i]:
                    if i > 0 and (sIsYUV):
                        expr.append(limitExprC)
                    else:
                        expr.append(limitExprY)
                else:
                    expr.append("")
            
            if ref is None:
                clip = vs.core.std.Expr([flt, src], expr)
            else:
                clip = vs.core.std.Expr([flt, src, ref], expr)
        else: # implementation with std.MakeDiff, std.Lut and std.MergeDiff
            diff = vs.core.std.MakeDiff(flt, src, planes=planes)
            if sIsYUV:
                if process[0]:
                    diff = _limit_diff_lut(diff, thr, elast, brighten_thr, [0])
                if process[1] or process[2]:
                    _planes = []
                    if process[1]:
                        _planes.append(1)
                    if process[2]:
                        _planes.append(2)
                    diff = _limit_diff_lut(diff, thrc, elast, thrc, _planes)
            else:
                diff = _limit_diff_lut(diff, thr, elast, brighten_thr, planes)
            clip = vs.core.std.MakeDiff(flt, diff, planes=planes)

        # Output
        return clip

class output:
    def rav1e(clip:vs.VideoNode, file_str:str, speed:int=6, scd_speed:int=1,
        quantizer:int=100, gop:Optional[int]=None, 
        tiles_row:Optional[int]=None, tiles_col:Optional[int]=None,
        color_range:Optional[str]=None, primaties:Optional[str]=None, 
        transfer:Optional[str]=None, matrix:Optional[str]=None, 
        mastering_display:Optional[str]=None, content_light:Optional[str]=None, 
        executable:str="rav1e"):

        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10', 'YUV420P12', 'YUV422P12', 'YUV444P12']:
            raise vs.Error(f"Pixel format must be one of `YUV420P8, YUV422P8, YUV444P8, YUV420P10, YUV422P10, YUV444P10, YUV420P12, YUV422P12, YUV444P12`  currently {clip.format.name}")

        if gop is None: gop = min(300, helper.round_to_closest(clip.fps)*10)

        args = [
            executable,
            "-",
            "-o", "-",
            "--quantizer", f"{quantizer}",
            "--speed", f"{speed}",
            "--scd_speed", f"{scd_speed}",
            "--keyint", f"{gop}"
        ]

        if color_range       is not None: args += ["--range", f"{color_range}"]
        if primaties         is not None: args += ["--primaries", f"{primaties}"]
        if transfer          is not None: args += ["--transfer", f"{transfer}"]
        if matrix            is not None: args += ["--matrix", f"{matrix}"]
        if mastering_display is not None: args += ["--mastering-display", f"{mastering_display}"]
        if content_light     is not None: args += ["--content-light", f"{content_light}"]
        if tiles_row         is not None: args += ["--tile-rows", f"{tiles_row}"]
        if tiles_col         is not None: args += ["--tile-cols", f"{tiles_col}"]

        args += ["-"]

        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, y4m=True)
            process.stdin.close()
        file.close()

    # def svtav1(clip:vs.VideoNode, file_str:str,


    #     stat_file:Optional[str]=None):

    #     if clip.format.name not in ['YUV420P8', 'YUV420P10']:
    #         raise vs.Error(f"Pixel format must be one of `YUV420P8, YUV420P10` currently {clip.format.name}")

    #     args = [
    #         executable,
    #         "-i", "-",
    #         "-b", "stdout",
    #         "--fps-num", f"{clip.fps.numerator}",
    #         "--fps-denom", f"{clip.fps.denominator}",
    #         "--width", f"{clip.width}",
    #         "--height", f"{clip.height}",
    #         "--input-depth", f"{clip.format.bits_per_sample}"
    #     ]

    #     if stat_file is not None: args += [
    #         "--stat-file", f"{stat_file}",
    #         "--enable-stat-report", "1"
    #     ]

    #     print(" ".join(args))
    #     file = open(file_str, 'wb')
    #     with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
    #         clip.output(process.stdin)
    #         process.stdin.close()
    #     file.close()

    def aomenc_fp(clip:vs.VideoNode, fpf=str, speed:int=6, executable:str="aomenc"):

        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10', 'YUV420P12', 'YUV422P12', 'YUV444P12']:
            raise vs.Error(f"Pixel format must be one of `YUV420P8, YUV422P8, YUV444P8, YUV420P10, YUV422P10, YUV444P10, YUV420P12, YUV422P12, YUV444P12`  currently {clip.format.name}")

        if (clip.format.name in ["YUV420P8", "YUV420P10"]):
            profile=0
        elif (clip.format.name in ["YUV444P8", "YUV444P10"]) and monochrome:
            profile=1
        elif (clip.format.name in ["YUV422P8", "YUV422P10", "YUV422P12", "YUV420P12", "YUV444P12"]):
            profile=2

        args = [
            executable,
            "-",
            "--passes=2",
            "--pass=1",
            f"--cpu-used={speed}",
            f"--fps={clip.fps.numerator}/{clip.fps.denominator}",
            f"--profile={profile}",
            f"--width={clip.width}",
            f"--height={clip.height}",
            f"--input-chroma-subsampling-x={clip.format.subsampling_w}",
            f"--input-chroma-subsampling-y={clip.format.subsampling_h}",
            f"--input-bit-depth={clip.format.bits_per_sample}",
            f"--bit-depth={clip.format.bits_per_sample}",
            f"--fpf={fpf}",
            "--output=-"
        ]

        print(" ".join(args))
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
            clip.output(process.stdin)
            process.stdin.close()

    def aomenc(clip:vs.VideoNode, file_str:str, mux:str="webm", speed:int=4, 
        usage:str="q", quantizer:int=32, bitrate_min:int=1500,
        bitrate_mid:int=2000, bitrate_max:int=2500, gop:Optional[int]=None, 
        lif:Optional[int]=None, tiles_row:Optional[int]=None,
        tiles_col:Optional[int]=None, enable_cdef:bool=True,
        enable_restoration:Optional[bool]=None, enable_chroma_deltaq:bool=True,
        arnr_strength:int=2, arnr_max_frames:int=5, resize_mode:Optional[int]=None,
        superres_mode:Optional[int]=None, tune:Optional[str]=None, tune_content:Optional[str]=None,
        fpf:Optional[str]=None, monochrome:bool=False, executable:str="aomenc"):
        # Only Q or VBR

        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10', 'YUV420P12', 'YUV422P12', 'YUV444P12']:
            raise vs.Error(f"Pixel format must be one of `YUV420P8, YUV422P8, YUV444P8, YUV420P10, YUV422P10, YUV444P10, YUV420P12, YUV422P12, YUV444P12`  currently {clip.format.name}")

        if (clip.format.name in ["YUV420P8", "YUV420P10"]):
            profile=0 # Main
        elif (clip.format.name in ["YUV444P8", "YUV444P10"]) and not monochrome:
            profile=1 # High
            # No monochrome support here or some reason.
        else:
        # elif (clip.format.name in ["YUV422P8", "YUV422P10", "YUV422P12", "YUV420P12", "YUV444P12"]):
            profile=2 # Professional

        if (mux not in ["ivf", "webm", "obu"]):
            raise vs.Error('Muxing container format must be one of `ivf webm obu`')

        if gop is None: gop = min(300, helper.round_to_closest(clip.fps)*10)
        if lif is None: lif = min(35, gop)
        if tiles_row is None: tiles_row = math.floor(clip.height/1080) if clip.height<clip.width else math.floor(clip.height/1920)
        if tiles_col is None: tiles_col = math.floor(clip.width/1920) if clip.height<clip.width else math.floor(clip.width/1080)

        enable_cdef = "1" if enable_cdef else "0"
        enable_chroma_deltaq = "1" if enable_chroma_deltaq else "0"

        if enable_restoration is None: 
            if (clip.height*clip.width >= 3200*2000) or not enable_restoration: # if smaller than 2160p
                enable_restoration = 0
            else: enable_restoration = 1

        args = [
            executable,
            "-",
            # "--rate-hist=1",
            f"--{mux}" ,
            f"--cpu-used={speed}"
        ]

        if fpf is not None: 
            args += [
                "--passes=2",
                "--pass=2",
                f"--fpf={fpf}"
            ]
        else:
            args += ["--passes=1"]

        if usage == "q":
            args += ["--end-usage=q", f"--cq-level={quantizer}"]
        elif usage == "vbr":
            args += [
                "--end-usage=vbr", "--bias-pct=75",
                f"--target-bitrate={bitrate_mid}",
                f"--undershoot-pct={helper.round_to_closest(((bitrate_mid - bitrate_min) / bitrate_min)*100)}",
                f"--overshoot-pct={helper.round_to_closest(((bitrate_max - bitrate_mid) / bitrate_mid)*100)}"
            ]
        else:
            raise vs.Error('Only q and vbr end usages are supported.')

        # if clip.format.color_family == vs.GRAY:
        #     args+="--monochrome"
        if monochrome:
            args += ["--monochrome"]

        args += [
            f"--fps={clip.fps.numerator}/{clip.fps.denominator}",
            f"--profile={profile}",
            f"--width={clip.width}",
            f"--height={clip.height}",
            f"--input-chroma-subsampling-x={clip.format.subsampling_w}",
            f"--input-chroma-subsampling-y={clip.format.subsampling_h}",
            f"--input-bit-depth={clip.format.bits_per_sample}",
            f"--bit-depth={clip.format.bits_per_sample}",
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
        ]

        # if (clip.height*clip.width < 1280*720):
        #     args += ["--max-partition-size=64", "--sb-size=64"] # sb-size used to be 32 but now it can't work? `dynamic, 64, 128`
        if (clip.height*clip.width < 1920*1080):
            args += ["--max-partition-size=64", "--sb-size=64"]

        if resize_mode is not None: args += [f"--resize-mode={resize_mode}"]
        if superres_mode is not None: args += [f"--superres-mode={superres_mode}"]
        if tune is not None: args += [f"--tune={tune}"]
        if tune_content is not None: args += [f"--tune-content={tune_content}"]

        args += [
            "--min-q=1",
            f"--enable-fwd-kf=1",
            "--quant-b-adapt=1",
            "--enable-dual-filter=0",
            f"--enable-qm=1",
            "--qm-min=5",
            "--output=-"
        ]

        print(" ".join(args))
        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin)
            process.stdin.close()
        file.close()

    def __get_progress__(a:int, b:int):

        print(f"Progress: {str(math.floor((a/b)*100)).rjust(3,' ')}% {str(a).rjust(str(b).__len__())}/{b}", end="\r")

    def __ff_fmt_conv__(vs_format_name:str) -> str:

        if vs_format_name not in ["YUV411P8","YUV410P8","YUV420P8","YUV422P8","YUV440P8","YUV444P8","YUV420P9","YUV422P9","YUV444P9","YUV420P10","YUV422P10","YUV444P10","YUV420P12","YUV422P12","YUV444P12","YUV420P14","YUV422P14","YUV444P14","YUV420P16","YUV422P16","YUV444P16","Gray8","Gray9","Gray10","Gray12","Gray16","RGB24","RGB27","RGB30","RGB36","RGB42","RGB48"]:
            raise vs.Error(f"Pixel format must be one of `YUV411P8, YUV410P8, YUV420P8, YUV422P8, YUV440P8, YUV444P8, YUV420P9, YUV422P9, YUV444P9, YUV420P10, YUV422P10, YUV444P10, YUV420P12, YUV422P12, YUV444P12, YUV420P14, YUV422P14, YUV444P14, YUV420P16, YUV422P16, YUV444P16, Gray8, Gray9, Gray10, Gray12, Gray16, RGB24, RGB27, RGB30, RGB36, RGB42, RGB48` currently {clip.format.name}")

        return {
            "YUV411P8": "yuv411p",
            "YUV410P8": "yuv410p",
            "YUV420P8": "yuv420p",
            "YUV422P8": "yuv422p",
            "YUV440P8": "yuv440p",
            "YUV444P8": "yuv444p",
            "YUV420P9": "yuv420p9le",
            "YUV422P9": "yuv422p9le",
            "YUV444P9": "yuv444p9le",
            "YUV420P10": "yuv420p10le",
            "YUV422P10": "yuv422p10le",
            "YUV444P10": "yuv444p10le",
            "YUV420P12": "yuv420p12le",
            "YUV422P12": "yuv422p12le",
            "YUV444P12": "yuv444p12le",
            "YUV420P14": "yuv420p14le",
            "YUV422P14": "yuv422p14le",
            "YUV444P14": "yuv444p14le",
            "YUV420P16": "yuv420p16le",
            "YUV422P16": "yuv422p16le",
            "YUV444P16": "yuv444p16le",
            "Gray8": "gray",
            "Gray9": "gray9le",
            "Gray10": "gray10le",
            "Gray12": "gray12le",
            "Gray16": "gray16le",
            "RGB24": "gbrp",
            "RGB27": "gbrp9le",
            "RGB30": "gbrp10le",
            "RGB36": "gbrp12le",
            "RGB42": "gbrp14le",
            "RGB48": "gbrp16le" 
        }.get(vs_format_name)

    def libx264(clip:vs.VideoNode, file_str:str, preset:str='veryslow', crf:int=7, 
            crf_max:Optional[int]=None, gop:Optional[int]=None, threads:int=3, 
            rc_lookahead:Optional[int]=None, mux:str='nut', executable:str='ffmpeg'):

        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10']:
            raise vs.Error(f"Pixel format must be one of `YUV420P8, YUV420P10` currently {clip.format.name}")

        args = [
            executable,
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

    def llx264(clip:vs.VideoNode, file_str:str, preset:str='veryslow', mux:str='nut', executable:str='ffmpeg'):

        if clip.format.name not in ['YUV420P8', 'YUV422P8', 'YUV444P8', 'YUV420P10', 'YUV422P10', 'YUV444P10']:
            raise vs.Error(f"Pixel format must be one of `YUV420P8, YUV420P10` currently {clip.format.name}")

        args = [
            executable,
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

    def ffv1(clip:vs.VideoNode, file_str:str, mux:str='nut', executable:str='ffmpeg'):

        if clip.format.name not in ["YUV411P8", "YUV410P8", "YUV420P8", "YUV422P8", "YUV440P8", "YUV444P8", "YUV420P9", "YUV422P9", "YUV444P9", "YUV420P10", "YUV422P10", "YUV444P10", "YUV420P12", "YUV422P12", "YUV444P12", "YUV420P14", "YUV422P14", "YUV444P14", "YUV420P16", "YUV422P16", "YUV444P16", "Gray8", "Gray9", "Gray10", "Gray12", "Gray16", "RGB27", "RGB30", "RGB36", "RGB42", "RGB48"]:
            raise vs.Error(f"Pixel format must be one of `YUV411P8, YUV410P8, YUV420P8, YUV422P8, YUV440P8, YUV444P8, YUV420P9, YUV422P9, YUV444P9, YUV420P10, YUV422P10, YUV444P10, YUV420P12, YUV422P12, YUV444P12, YUV420P14, YUV422P14, YUV444P14, YUV420P16, YUV422P16, YUV444P16, Gray8, Gray9, Gray10, Gray12, Gray16, RGB27, RGB30, RGB36, RGB42, RGB48` currently {clip.format.name}")

        if clip.format.color_family is vs.RGB:
            # rgb to gbr
            clip = util.join([
                    util.plane(clip, 1), # g
                    util.plane(clip, 2), # b 
                    util.plane(clip, 0)  # r
                ], vs.RGB)

        args = [
            executable,
            "-y",
            "-hide_banner",
            "-v", "8",
            "-f", "rawvideo",
            "-pixel_format", f"{output.__ff_fmt_conv__(clip.format.name)}",
            "-video_size", f"{clip.width}x{clip.height}",
            "-framerate", f"{clip.fps}",
            "-i", "-",
            "-c", "ffv1",
            "-f", f"{mux}",
            "-"
        ]

        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, progress_update=output.__get_progress__)
            process.stdin.close()
        file.close()

    def libaom(clip:vs.VideoNode, file_str:str, speed:int=4, quantizer:int=32,
        gop:Optional[int]=None, lif:Optional[int]=None,
        tiles_row:Optional[int]=None, tiles_col:Optional[int]=None,
        enable_cdef:bool=True, enable_restoration:Optional[bool]=None, 
        enable_chroma_deltaq:bool=True, arnr_strength:int=2,
        arnr_max_frames:int=5, threads:int=4, mux:str='nut', executable='ffmpeg'):

        if clip.format.name not in ["YUV420P8", "YUV422P8", "YUV444P8", "YUV420P10", "YUV422P10", "YUV444P10", "YUV420P12", "YUV422P12", "YUV444P12", "Gray8", "Gray10", "Gray12", "RGB24", "RGB30", "RGB36"]:
            raise vs.Error(f"Pixel format must be one of `YUV420P8, YUV422P8, YUV444P8, YUV420P10, YUV422P10, YUV444P10, YUV420P12, YUV422P12, YUV444P12, Gray8, Gray10, Gray12, RGB24, RGB30, RGB36` currently {clip.format.name}")

        if clip.format.color_family is vs.RGB:
            # rgb to gbr
            clip = util.join([
                    util.plane(clip, 1), # g
                    util.plane(clip, 2), # b 
                    util.plane(clip, 0)  # r
                ], vs.RGB)

        if gop is None: gop = min(300, helper.round_to_closest(clip.fps)*10)
        if lif is None: lif = min(35, gop)
        tiles_row = math.floor(clip.height/1080) if clip.height<clip.width else math.floor(clip.height/1920)
        tiles_col = math.floor(clip.width/1920) if clip.height<clip.width else math.floor(clip.width/1080)

        if enable_restoration is None: 
            if (clip.height*clip.width >= 3200*2000): # if smaller than 2160p
                enable_restoration = True
            else: enable_restoration = False

        if (clip.height*clip.width < 1280*720):
            aom_params = ":max-partition-size=64:sb-size=64"
        elif (clip.height*clip.width < 1920*1080): # if smaller than 1080p
            aom_params = ":max-partition-size=64:sb-size=64"

        args = [
            executable,
            "-y",
            "-hide_banner",
            "-v", "8",
            "-f", "rawvideo",
            "-pixel_format", f"{output.__ff_fmt_conv__(clip.format.name)}",
            "-video_size", f"{clip.width}x{clip.height}",
            "-framerate", f"{clip.fps}",
            "-i", "-",
            "-c", "libaom-av1",
            "-threads", f"{threads}",
            "-cpu-used", f"{speed}",
            "-crf", f"{quantizer}",
            "-g", f"{gop}",
            "-lag-in-frames", f"{lif}",
            "-tile-columns", f"{tiles_col}",
            "-tile-rows", f"{tiles_row}",
            "-enable-cdef", f"{enable_cdef}",
            "-enable-restoration", f"{enable_restoration}",
            "-arnr-strength", f"{arnr_strength}",
            "-arnr-max-frames", f"{arnr_max_frames}",
            "-aom-params", f"enable-qm=1:qm-min=5{aom_params}",
            "-f", f"{mux}",
            "-"
        ]

        file = open(file_str, 'wb')
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=file) as process:
            clip.output(process.stdin, progress_update=output.__get_progress__)
            process.stdin.close()
        file.close()

    def libsvtav1(clip:vs.VideoNode, file_str:str, speed:int=6, quantizer:int=32,
            gop:Optional[int]=None, lad:Optional[int]=None,
            tiles_row:Optional[int]=None, tiles_col:Optional[int]=None,
            sc_detection:bool=False, threads:int=4, mux:str='nut', executable:str='ffmpeg'):

        if clip.format.name not in ['YUV420P8', 'YUV420P10']:
            raise vs.Error(f"Pixel format must be one of `YUV420P8, YUV420P10` currently {clip.format.name}")

        if gop is None: gop = min(300, helper.round_to_closest(clip.fps)*10)
        if lad is None: lad = min(120, gop)
        if tiles_row is None: tiles_row = math.floor(clip.height/1080)
        if tiles_col is None: tiles_col = math.floor(clip.width/1920)

        args = [
            executable,
            "-y",
            "-hide_banner",
            "-v", "8",
            "-f", "rawvideo",
            "-pixel_format", f"{output.__ff_fmt_conv__(clip.format.name)}",
            "-video_size", f"{clip.width}x{clip.height}",
            "-framerate", f"{clip.fps}",
            "-i", "-",
            "-c", "libsvtav1",
            "-threads", f"{threads}",
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
            clip.output(process.stdin, progress_update=output.__get_progress__)
            process.stdin.close()
        file.close()

def boundary_pad(clip:vs.VideoNode, boundary_width:int, boundary_height:int,
    x:int=50, y:int=50) -> vs.VideoNode:

    if (boundary_width > clip.width) or (boundary_height > clip.height):
        clip = vs.core.std.AddBorders(clip,
            left   = (boundary_width-clip.width)  *(x/100),
            right  = (boundary_width-clip.width)  *(1-(x/100)),
            top    = (boundary_height-clip.height)*(y/100),
            bottom = (boundary_height-clip.height)*(1-(y/100)))
    return clip

def resize(clip:vs.VideoNode, width:Optional[int]=None,
    height:Optional[int]=None, scale:Optional[Union[int, float]]=None,
    kernel:str="didée", taps:Optional[int]=None) -> vs.VideoNode:

    if hasattr(vs.core, 'fmtc') is not True:
        raise RuntimeError("resize: fmtconv plugin is required")

    if scale is not None:
        scale_kwargs = {"scale": scale}
    elif width is not None and height is not None:
        scale_kwargs = {"w": width, "h": height}
    else:
        raise vs.Error("resize: Must specify scale or height and scale")

    kernel_kwargs = util.fmtc_kernel_kwargs(kernel, taps)

    clip = vs.core.fmtc.resample(clip, **scale_kwargs, **kernel_kwargs)
    return clip

def ssim_downsample(clip:vs.VideoNode, width:int, height:int, smooth:Union[int,float]=1,
        kernel:str="didée", gamma:bool=False, curve=vs.TransferCharacteristics.TRANSFER_BT709,
        sigmoid:bool=False, epsilon:float=0.000001) -> vs.VideoNode:

    # TODO: Clean
    # from https://gist.github.com/Ichunjo/16ab1f893588aafcb096c1f35a0cfb15

    CURVES = Literal [
        vs.TransferCharacteristics.TRANSFER_IEC_61966_2_1,
        vs.TransferCharacteristics.TRANSFER_BT709,
        vs.TransferCharacteristics.TRANSFER_BT601,
        vs.TransferCharacteristics.TRANSFER_ST240_M,
        vs.TransferCharacteristics.TRANSFER_BT2020_10,
        vs.TransferCharacteristics.TRANSFER_BT2020_12,
    ]

    class Coefs(NamedTuple):
        k0: float
        phi: float
        alpha: float
        gamma: float

    def get_coefs(curve: vs.TransferCharacteristics) -> Coefs:

        srgb = Coefs(0.04045, 12.92, 0.055, 2.4)
        bt709 = Coefs(0.08145, 4.5, 0.0993, 2.22222)
        smpte240m = Coefs(0.0912, 4.0, 0.1115, 2.22222)
        bt2020 = Coefs(0.08145, 4.5, 0.0993, 2.22222)

        gamma_linear_map = {
            vs.TransferCharacteristics.TRANSFER_IEC_61966_2_1: srgb,
            vs.TransferCharacteristics.TRANSFER_BT709: bt709,
            vs.TransferCharacteristics.TRANSFER_BT601: bt709,
            vs.TransferCharacteristics.TRANSFER_ST240_M: smpte240m,
            vs.TransferCharacteristics.TRANSFER_BT2020_10: bt2020,
            vs.TransferCharacteristics.TRANSFER_BT2020_12: bt2020
        }

        return gamma_linear_map[curve]

    def gamma2linear(clip: vs.VideoNode, curve:CURVES, gcor:float=1.0,
        sigmoid:bool=False, thr:float=0.5, cont:float=6.5,
        epsilon:float=1e-6) -> vs.VideoNode:

        assert clip.format
        if get_depth(clip) != 32 and clip.format.sample_type != vs.FLOAT:
            raise ValueError('Only 32 bits float is allowed')

        c = get_coefs(curve)

        expr = f'x {c.k0} <= x {c.phi} / x {c.alpha} + 1 {c.alpha} + / {c.gamma} pow ? {gcor} pow'
        if sigmoid:
            x0 = f'1 1 {cont} {thr} * exp + /'
            x1 = f'1 1 {cont} {thr} 1 - * exp + /'
            expr = f'{thr} 1 {expr} {x1} {x0} - * {x0} + {epsilon} max / 1 - {epsilon} max log {cont} / -'

        expr = f'{expr} 0.0 max 1.0 min'

        return vs.core.std.Expr(clip, expr).std.SetFrameProps(_Transfer=8)

    def linear2gamma(clip:vs.VideoNode, curve:CURVES, gcor:float=1.0,
        sigmoid:bool=False, thr:float=0.5, cont:float=6.5) -> vs.VideoNode:

        assert clip.format
        if get_depth(clip) != 32 and clip.format.sample_type != vs.FLOAT:
            raise ValueError('Only 32 bits float is allowed')

        c = get_coefs(curve)

        expr = 'x'
        if sigmoid:
            x0 = f'1 1 {cont} {thr} * exp + /'
            x1 = f'1 1 {cont} {thr} 1 - * exp + /'
            expr = f'1 1 {cont} {thr} {expr} - * exp + / {x0} - {x1} {x0} - /'

        expr += f' {gcor} pow'
        expr = f'{expr} {c.k0} {c.phi} / <= {expr} {c.phi} * {expr} 1 {c.gamma} / pow {c.alpha} 1 + * {c.alpha} - ?'
        expr = f'{expr} 0.0 max 1.0 min'

        return vs.core.std.Expr(clip, expr).std.SetFrameProps(_Transfer=curve)

    if hasattr(vs.core, 'fmtc') is not True:
        raise RuntimeError("resize: fmtconv plugin is required")

    if clip.format.bits_per_sample != 32 and clip.format.sample_type != vs.FLOAT:
        raise ValueError('ssim_downsample: only 32 bits float is allowed')

    if isinstance(smooth, int):
        filter_func = partial(vs.core.std.BoxBlur, hradius=smooth, vradius=smooth)
    elif isinstance(smooth, float):
        filter_func = partial(vs.core.tcanny.TCanny, sigma=smooth, mode=-1)
    else:
        vs.Error("ssim_downsample: smooth must be an int or float")

    if gamma:
        clip = gamma2linear(clip, curve, sigmoid=sigmoid, epsilon=epsilon)

    l = resize(clip=clip, width=width, height=height, kernel=kernel)
    l2 = resize(clip=clip.std.Expr('x dup *'), width=width, height=height, kernel=kernel)

    m = filter_func(l)

    sl_plus_m_square = filter_func(l.std.Expr('x dup *'))
    sh_plus_m_square = filter_func(l2)
    m_square = m.std.Expr('x dup *')
    r = vs.core.std.Expr([sl_plus_m_square, sh_plus_m_square, m_square], f'x z - {epsilon} < 0 y z - x z - / sqrt ?')
    t = filter_func(vs.core.std.Expr([r, m], 'x y *'))
    m = filter_func(m)
    r = filter_func(r)
    d = vs.core.std.Expr([m, r, l, t], 'x y z * + a -')

    if gamma:
        d = linear2gamma(d, curve, sigmoid=sigmoid)

    return d

def nnedi3_rpow2(clip:vs.VideoNode, rfactor:int=2, correct_shift:bool=True,
    width:Optional[int]=None, height:Optional[int]=None,
    kernel:str="didée", nsize:int=0, nns:int=2, qual:int=2, etype:int=0, 
    pscrn:Optional[int]=None, opt:bool=True, 
    int16_prescreener:bool=True, int16_predictor:bool=True, exp:int=0) -> vs.VideoNode:

    # nnedi3_rpow2 is for enlarging images by powers of 2.

    # Parameters:
    #   rfactor (int): Image enlargement factor.
    #       Must be a power of 2 in the range [2 to 1024].
    #   correct_shift (bool): If False, the shift is not corrected.
    #       The correction is accomplished by using the subpixel
    #       cropping capability of fmtc's resizers.
    #   width (int): If correcting the image center shift by using the
    #       "correct_shift" parameter, width/height allow you to set a
    #       new output resolution.
    #   kernel (string): Sets the resizer used for correcting the image
    #       center shift that nnedi3_rpow2 introduces.
    #   nsize, nns, qual, etype, pscrn, opt, int16_prescreener,
    #   int16_predictor, exp:
    #       See https://github.com/sekrit-twc/znedi3

    if width is None:
        width = clip.width*rfactor
    if height is None:
        height = clip.height*rfactor
    hshift = 0.0
    vshift = -0.5

    nnedi_kwargs = dict(dh=True, nsize=nsize, nns=nns, qual=qual, etype=etype,
                    pscrn=pscrn, opt=opt, int16_prescreener=int16_prescreener,
                    int16_predictor=int16_predictor, exp=exp)
    chroma_kwargs = dict(sy=-0.5, planes=[2, 3, 3])
    kernel_kwargs = util.fmtc_kernel_kwargs(kernel)

    tmp = 1
    times = 0
    while tmp < rfactor:
        tmp *= 2
        times += 1

    # Checks

    if rfactor < 2 or rfactor > 1024:
        raise ValueError("nnedi3_rpow2: rfactor must be between 2 and 1024")

    if tmp != rfactor:
        raise ValueError("nnedi3_rpow2: rfactor must be a power of 2")

    if hasattr(vs.core, 'nnedi3') is not True:
        raise RuntimeError("nnedi3_rpow2: nnedi3 plugin is required")

    if correct_shift or clip.format.subsampling_h:
        if hasattr(vs.core, 'fmtc') is not True:
            raise RuntimeError("nnedi3_rpow2: fmtconv plugin is required")

    last = clip

    for i in range(times):
        field = 1 if i == 0 else 0
        last = vs.core.nnedi3.nnedi3(last, field=field, **nnedi_kwargs)
        last = vs.core.std.Transpose(last)
        if last.format.subsampling_w:
            # Apparently always using field=1 for the horizontal pass somehow
            # keeps luma/chroma alignment.
            field = 1
            hshift = hshift*2 - 0.5
        else:
            hshift = -0.5
        last = vs.core.nnedi3.nnedi3(last, field=field, **nnedi_kwargs)
        last = vs.core.std.Transpose(last)

    # Correct vertical shift of the chroma.

    if clip.format.subsampling_h:
        last = vs.core.fmtc.resample(last, w=last.width, h=last.height, **chroma_kwargs, **kernel_kwargs)

    if correct_shift is True:
        last = vs.core.fmtc.resample(last, w=width, h=height, sx=hshift, sy=vshift, **kernel_kwargs)

    if last.format.id != clip.format.id:
        last = vs.core.fmtc.bitdepth(last, csp=clip.format.id)

    return last

def boundary_resize(clip:vs.VideoNode, width:Optional[int]=None,
    height:Optional[int]=None, scale:Optional[Union[int, float]]=None,
    multiple:int=2, crop:bool=False, kernel:str="didée",
    taps:Optional[int]=None, ssim:bool=False, ssim_kwargs:dict={}) -> vs.VideoNode:

    if scale is not None:
        clip = resize(clip, scale=scale)
    else:
        new_height = original_height = clip.height
        new_width  = original_width  = clip.width
        if width is not None and height is not None:
            if new_height > height:
                new_height = height
                new_width = helper.round_to_closest((new_height * original_width) / original_height)
            if new_width > width:
                new_width = width
                new_height = helper.round_to_closest((new_width * original_height) / original_width)
        elif width is not None:
            new_width = width
            new_height = helper.round_to_closest((new_width * original_height) / original_width)
        elif height is not None:
            new_height = height
            new_width = helper.round_to_closest((new_height * original_width) / original_height)
        else:
            raise vs.Error("boundary_resize: Must specify width, height or scale")

        if multiple > 1 and not crop:
            new_width  = helper.mod_m(new_width, multiple)
            new_height = helper.mod_m(new_height, multiple)
        scale_kwargs = {"w": new_width, "h": new_height}

    if ssim:
        clip = ssim_downsample(clip, height=new_height, width=new_width, kernel=kernel, **ssim_kwargs)
    else:
        clip = resize(clip, height=new_height, width=new_width, kernel=kernel)

    if multiple > 1 and crop:
        new_width_div  = helper.mod_m(new_width, multiple)
        new_height_div = helper.mod_m(new_height, multiple)
        if new_height_div != new_height or new_width_div != new_width:
            clip = vs.core.std.CropAbs(clip=clip, height=new_height_div,
                                                    width=new_width_div)

    return clip

def down_to_444(clip:vs.VideoNode, kernel:str="didée") -> vs.VideoNode:
    # 4k 420 -> 1080p 444

    if clip.format.color_family != vs.YUV:
        raise vs.Error('down_to_444: only YUV format is supported')

    y = vs.core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
    u = vs.core.std.ShufflePlanes(clip, planes=1, colorfamily=vs.GRAY)
    v = vs.core.std.ShufflePlanes(clip, planes=2, colorfamily=vs.GRAY)

    height = u.height
    width  = u.width

    y = resize(clip=y, height=height, width=width, kernel=kernel)
    u = resize(clip=u, height=height, width=width, kernel=kernel)
    v = resize(clip=v, height=height, width=width, kernel=kernel)

    clip = vs.core.std.ShufflePlanes(clips=[y,u,v], planes=[0,0,0], colorfamily=vs.YUV)
    return clip

def up_to_444(clip:vs.VideoNode, kernel:str="robidoux") -> vs.VideoNode:
    # 4k 420 -> 4k 444

    if clip.format.color_family != vs.YUV:
        raise vs.Error('down_to_444: only YUV format is supported')

    y = vs.core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
    u = vs.core.std.ShufflePlanes(clip, planes=1, colorfamily=vs.GRAY)
    v = vs.core.std.ShufflePlanes(clip, planes=2, colorfamily=vs.GRAY)

    height = y.height
    width  = y.width

    y = resize(clip=y, height=height, width=width, kernel=kernel)
    u = resize(clip=u, height=height, width=width, kernel=kernel)
    v = resize(clip=v, height=height, width=width, kernel=kernel)

    clip = vs.core.std.ShufflePlanes(clips=[y,u,v], planes=[0,0,0], colorfamily=vs.YUV)
    return clip

def bt2390_ictcp(clip:vs.VideoNode, source_peak:Optional[int]=None,
    target_nits:float=1) -> vs.VideoNode:
    # Stolen from somewhere, idk where.
    # clip = bt2390_ictcp(clip,target_nits=100,source_peak=1000)

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

def imwri_src(dir:str, fpsnum:int, fpsden:int, firstnum:int=0,
    alpha:bool=False, ext:str=".png") -> vs.VideoNode:

    srcs = [dir + src for src in os.listdir(dir) if src.endswith(ext)]
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

    analyse_kwargs = {
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
    vectors = _fun_analyse(mvsuper, isb=True, **analyse_kwargs)

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

    analyse_kwargs = {
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
    block_or_flow_kwargs = {
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
    block_vectors = _fun_analyse(mvsuper, isb=True, **analyse_kwargs)
    flow_vectors = _fun_analyse(mvsuper, isb=False, **analyse_kwargs)

    if clip.fps.numerator/clip.fps.denominator > fpsnum/fpsden:
        clip = _fun_flowblur(clip, mvsuper, block_vectors, flow_vectors, blur=Mblur)

    if block == True:
        out = _fun_blockfps(clip, mvsuper, block_vectors, flow_vectors, **block_or_flow_kwargs, mode=block_mode)
    else:
        out = _fun_flowfps(clip, mvsuper, block_vectors, flow_vectors, **block_or_flow_kwargs, mask=flow_mask)
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

def mv_flux_smooth(clip:vs.VideoNode, temporal_threshold:int=12, 
    super_kwargs:dict={}, analyse_kwargs:dict={}, compensate_kwargs:dict={},
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

    super_kwargs = {"pel":2, "sharp":1, **super_kwargs}
    analyse_kwargs = {"truemotion":False, "delta":1, "blksize":16, "overlap":8, **analyse_kwargs}
    mvsuper = _fun_super(clip, **super_kwargs)
    mvanalyset = _fun_analyse(mvsuper, isb=True, **analyse_kwargs)
    mvanalysef = _fun_analyse(mvsuper, isb=False, **analyse_kwargs)
    mvcompensavet = _fun_compensate(clip, mvsuper, mvanalyset, **compensate_kwargs)
    mvcompensavef = _fun_compensate(clip, mvsuper, mvanalysef, **compensate_kwargs)
    mvinterleave = vs.core.std.Interleave([mvcompensavef, clip, mvcompensavet])
    fluxsmooth = vs.core.flux.SmoothT(mvinterleave, temporal_threshold=temporal_threshold, planes=planes)
    return vs.core.std.SelectEvery(fluxsmooth, 3, 1)

def STPressoMC(clip:vs.VideoNode, limit:int=3, bias:int=24, RGVS_mode:int=4,
    temporal_threshold:int=12, temporal_limit:int=3, temporal_bias:int=49, back:int=1,
    super_kwargs:dict={}, analyse_kwargs:dict={}, compensate_kwargs:dict={}) -> vs.VideoNode:
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
    LIM1 = helper.scale_depth(LIM1,depth)
    #(limit>0) ? round(limit*100.0/bias-1.0) :  round(100.0/bias)
    LIM2 = 1 if limit < 0 else limit
    LIM2 = helper.scale_depth(LIM2,depth)
    #(limit<0) ? 1 : limit
    BIA = bias
    BK = helper.scale_depth(back,depth)
    TBIA = bias
    TLIM1 = round(temporal_limit*100.0/temporal_bias-1.0) if temporal_limit > 0 else round(100.0/temporal_bias)
    TLIM1 = helper.scale_depth(TLIM1,depth)
    #(temporal_limit>0) ? string( round(temporal_limit*100.0/temporal_bias-1.0) ) : string( round(100.0/temporal_bias) )
    TLIM2  = 1 if temporal_limit < 0 else temporal_limit
    TLIM2 = helper.scale_depth(TLIM2,depth)
    #(temporal_limit<0) ? "1" : string(temporal_limit)
    clip_rgvs = vs.core.rgvs.RemoveGrain(clip,RGVS_mode)
    ####
    if limit < 0:
        expr  = f"x y - abs {LIM1} < x x {helper.scale_depth(1, depth)} x y - x y - abs / * - ?"
        texpr = f"x y - abs {TLIM1} < x x {helper.scale_depth(1, depth)} x y - x y - abs / * - ?"
    else:
        expr  =  f"x y - abs {helper.scale_depth(1, depth)} < x x {LIM1} + y < x {LIM2} + x {LIM1} - y > x {LIM2} - x {helper.scale_depth(100, depth)} {BIA} - * y {BIA} * + {helper.scale_depth(100, depth)} / ? ? ?"
        texpr =  f"x y - abs {helper.scale_depth(1, depth)} < x x {TLIM1} + y < x {TLIM2} + x {TLIM1} - y > x {TLIM2} - x {helper.scale_depth(100, depth)} {TBIA} - * y {TBIA} * + {helper.scale_depth(100, depth)} / ? ? ?"
    L=[]
    for plane in range(0,3):
        C = vs.core.std.ShufflePlanes(clip, plane, colorfamily=vs.GRAY)
        B = vs.core.std.ShufflePlanes(clip_rgvs, plane, colorfamily=vs.GRAY)
        O = vs.core.std.Expr([C,B],expr)
        L.append(O)
    if temporal_threshold != 0:
        st = mv_flux_smooth(clip_rgvs, temporal_threshold, super_kwargs, analyse_kwargs, compensate_kwargs, [0,1,2])
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

def HQDeringmod(clip:vs.VideoNode, p:Optional[vs.VideoNode]=None,
    ringmask:Optional[vs.VideoNode]=None, mrad:int=1, msmooth:int=1,
    incedge:bool=False, mthr:int=60, minp:int=1, nrmode:Optional[int]=None,
    sharp:int=1, drrep:int=24, thr:float=12.0, elast:float=2.0,
    darkthr:Optional[float]=None, planes:List[int]=[0], show:bool=False) -> vs.VideoNode:
    # original script by mawen1250, taken from havsfunc

    # Applies deringing by using a smart smoother near edges (where ringing occurs) only

    # Parameters:
    #  mrad (int): Expanding of edge mask, higher value means more aggressive processing. Default is 1
    #  msmooth (int): Inflate of edge mask, smooth boundaries of mask. Default is 1
    #  incedge (bool): Whether to include edge in ring mask, by default ring mask only include area near edges. Default is false
    #  mthr (int): Threshold of prewitt edge mask, lower value means more aggressive processing. Or define your own mask clip "ringmask". Default is 60
    #            But for strong ringing, lower value will treat some ringing as edge, which protects this ringing from being processed.
    #  minp (int): Inpanding of prewitt edge mask, higher value means more aggressive processing. Default is 1
    #  nrmode (int): Kernel of dering - 1: min_blur(radius=1), 2: min_blur(radius=2), 3: min_blur(radius=3). Or define your own smoothed clip "p". Default is 2 for HD / 1 for SD
    #  sharp (int): Whether to use contra-sharpening to resharp deringed clip, 1-3 represents radius, 0 means no sharpening. Default is 1
    #  drrep (int): Use repair for details retention, recommended values are 24/23/13/12/1. Default is 24
    #  thr (float): The same meaning with "thr" in Dither_limit_dif16, valid value range is [0.0, 128.0]. Default is 12.0
    #  elast (float): The same meaning with "elast" in Dither_limit_dif16, valid value range is [1.0, inf). Default is 2.0
    #            Larger "thr" will result in more pixels being taken from processed clip
    #            Larger "thr" will result in less pixels being taken from input clip
    #            Larger "elast" will result in more pixels being blended from processed&input clip, for smoother merging
    #  darkthr (float): Threshold for darker area near edges, set it lower if you think deringing destroys too much lines, etc. Default is thr/4
    #            When "darkthr" is not equal to "thr", "thr" limits darkening while "darkthr" limits brightening
    #  planes (list[int]): Whether to process the corresponding plane. The other planes will be passed through unchanged. Default is [0]
    #  show (bool): Whether to output mask clip instead of filtered clip. Default is false

    if clip.format.color_family == vs.RGB:
        raise vs.Error('HQDeringmod: RGB format is not supported')

    if p is not None and p.format.id != clip.format.id:
        raise vs.Error("HQDeringmod: 'p' must be the same format as clip")

    isGray = (clip.format.color_family == vs.GRAY)

    neutral = 1 << (clip.format.bits_per_sample - 1)
    peak = (1 << clip.format.bits_per_sample) - 1

    if isinstance(planes, int):
        planes = [planes]

    if nrmode is None:
        nrmode = 2 if clip.width > 1024 or clip.height > 576 else 1
    if darkthr is None:
        darkthr = thr / 4

    # Kernel: Smoothing
    if p is None:
        p = util.min_blur(clip, r=nrmode, planes=planes)

    # Post-Process: Contra-Sharpening
    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if sharp <= 0:
        sclp = p
    else:
        pre = p.std.Median(planes=planes)
        if sharp == 1:
            method = pre.std.Convolution(matrix=matrix1, planes=planes)
        elif sharp == 2:
            method = pre.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        else:
            method = pre.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        sharpdiff = vs.core.std.MakeDiff(pre, method, planes=planes)
        allD = vs.core.std.MakeDiff(clip, p, planes=planes)
        ssDD = vs.core.rgvs.Repair(sharpdiff, allD, mode=[1 if i in planes else 0 for i in range(clip.format.num_planes)])
        expr = f'x {neutral} - abs y {neutral} - abs <= x y ?'
        ssDD = vs.core.std.Expr([ssDD, sharpdiff], expr=[expr if i in planes else '' for i in range(clip.format.num_planes)])
        sclp = vs.core.std.MergeDiff(p, ssDD, planes=planes)

    # Post-Process: Repairing
    if drrep <= 0:
        repclp = sclp
    else:
        repclp = vs.core.rgvs.Repair(clip, sclp, mode=[drrep if i in planes else 0 for i in range(clip.format.num_planes)])

    # Post-Process: Limiting
    if (thr <= 0 and darkthr <= 0) or (thr >= 128 and darkthr >= 128):
        limitclp = repclp
    else:
        limitclp = util.limit_filter(repclp, clip, thr=thr, elast=elast, brighten_thr=darkthr, planes=planes)

    # Post-Process: Ringing Mask Generating
    if ringmask is None:
        prewittm = util.edge_detection(clip, planes=[0], thr=mthr, method="prewitt", scale=True)
        fmask = vs.core.misc.Hysteresis(prewittm.std.Median(planes=[0]), prewittm, planes=[0])
        if mrad > 0:
            omask = util.mt_expand_multi(fmask, planes=[0], sw=mrad, sh=mrad)
        else:
            omask = fmask
        if msmooth > 0:
            omask = util.mt_inflate_multi(omask, planes=[0], radius=msmooth)
        if incedge:
            ringmask = omask
        else:
            if minp > 3:
                imask = fmask.std.Minimum(planes=[0]).std.Minimum(planes=[0])
            elif minp > 2:
                imask = fmask.std.Inflate(planes=[0]).std.Minimum(planes=[0]).std.Minimum(planes=[0])
            elif minp > 1:
                imask = fmask.std.Minimum(planes=[0])
            elif minp > 0:
                imask = fmask.std.Inflate(planes=[0]).std.Minimum(planes=[0])
            else:
                imask = fmask
            expr = f'x {peak} y - * {peak} /'
            ringmask = vs.core.std.Expr([omask, imask], expr=[expr] if isGray else [expr, ''])

    # Mask Merging & Output
    if show:
        if isGray:
            return ringmask
        else:
            return ringmask.std.Expr(expr=['', repr(neutral)])
    else:
        return vs.core.std.MaskedMerge(clip, limitclp, ringmask, planes=planes, first_plane=True)

def fine_sharp(clip:vs.VideoNode, mode:int=1, 
    sharp_str:float=2.5, equal_str:Optional[float]=None, xsharp_str:float=0, nl_sharp_mod:float=1.5,
    nl_sharp_exp:float=1.28, low_damp:Optional[float]=None, high_damp:float=0.01, rep:int=12) -> vs.VideoNode:

    # Original author: Didée (https://forum.doom9.org/showthread.php?t=166082)
    # Small and relatively fast realtime-sharpening function, for 1080p,
    # or after scaling 720p → 1080p during playback.
    # (to make 720p look more like being 1080p)
    # It's a generic sharpener. Only for good quality sources!
    # (If the source is crap, FineSharp will happily sharpen the crap) :)
    # Noise/grain will be enhanced, too. The method is GENERIC.

    # Modus operandi: A basic nonlinear sharpening method is performed,
    # then the *blurred* sharp-difference gets subtracted again.

    # Args:
    #     mode (int): 1 to 3, weakest to strongest. When negative -1 to -3,
    #                    a broader kernel for equalisation is used.
    #     sharp_str (float): strength of sharpening.
    #     equal_str (float): strength of equalisation (recommended 0.5 to 1.25)
    #     xsharp_str (float): strength of XSharpen-style final sharpening, 0.0 to 1.0.
    #                    (but, better don't go beyond 0.25...)
    #     nl_sharp_mod (float): modifier for non-linear sharpening.
    #     nl_sharp_exp (float): exponent for non-linear sharpening.
    #     low_damp (float): "low damp", to not over-enhance very small differences.
    #                    (noise coming out of flat areas)
    #     high_damp (float): "high damp", this damping term has a larger effect than low_damp
    #                     when the sharp-difference is larger than 1, vice versa.
    #     rep (int): repair mode used in final sharpening, recommended modes are 1/12/13.

    def spline(x, coordinates):
        def get_matrix(px:int, py:int, l:int):
            matrix = []
            matrix.append([(i == 0) * 1.0 for i in range(l + 1)])
            for i in range(1, l - 1):
                p = [0 for t in range(l + 1)]
                p[i - 1] = px[i] - px[i - 1]
                p[i] = 2 * (px[i + 1] - px[i - 1])
                p[i + 1] = px[i + 1] - px[i]
                p[l] = 6 * (((py[i + 1] - py[i]) / p[i + 1]) - (py[i] - py[i - 1]) / p[i - 1])
                matrix.append(p)
            matrix.append([(i == l - 1) * 1.0 for i in range(l + 1)])
            return matrix
        def equation(matrix, dim:int):
            for i in range(dim):
                num = matrix[i][i]
                for j in range(dim + 1):
                    matrix[i][j] /= num
                for j in range(dim):
                    if i != j:
                        a = matrix[j][i]
                        for k in range(i, dim + 1):
                            matrix[j][k] -= a * matrix[i][k]
        if not isinstance(coordinates, dict):
            raise TypeError("coordinates must be a dict")
        length = len(coordinates)
        if length < 3:
            raise ValueError("coordinates require at least three pairs")
        px = [key for key in coordinates.keys()]
        py = [val for val in coordinates.values()]
        matrix = get_matrix(px, py, length)
        equation(matrix, length)
        for i in range(length + 1):
            if x >= px[i] and x <= px[i + 1]:
                break
        j = i + 1
        h = px[j] - px[i]
        s = matrix[j][length] * (x - px[i]) ** 3
        s -= matrix[i][length] * (x - px[j]) ** 3
        s /= 6 * h
        s += (py[j] / h - h * matrix[j][length] / 6) * (x - px[i])
        s -= (py[i] / h - h * matrix[i][length] / 6) * (x - px[j])
        
        return s

    color = clip.format.color_family
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    mid = 0 if isFLOAT else 1 << (bd - 1)
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    xy = 'x y - {} /'.format(i) if bd != 8 else 'x y -'
    R = vs.core.rgsf.Repair if isFLOAT else vs.core.rgvs.Repair
    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("FineSharp: This is not a clip!")

    if equal_str is None:
        equal_str = spline(sharp_str, {0: 0, 0.5: 0.1, 1: 0.6, 2: 0.9, 2.5: 1, 3: 1.1, 3.5: 1.15, 4: 1.2, 8: 1.25, 255: 1.5})
        equal_str **= 0.8 if mode > 0 else equal_str

    if low_damp is None:
        low_damp = sharp_str

    sharp_str = max(sharp_str, 0)
    equal_str = max(equal_str, 0)
    xsharp_str = min(max(xsharp_str, 0), 1)
    low_damp = max(low_damp, 0)
    high_damp = max(high_damp, 0)

    if sharp_str < 0.01 and equal_str < 0.01 and xsharp_str < 0.01:
        return clip

    tmp = vs.core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV] else clip

    if abs(mode) == 1:
        c2 = vs.core.std.Convolution(tmp, matrix=mat1).std.Median()
    else:
        c2 = vs.core.std.Median(tmp).std.Convolution(matrix=mat1)
    if abs(mode) == 3:
        c2 = c2.std.Median()
    
    if sharp_str >= 0.01:
        expr = 'x y = x dup {} dup dup dup abs {} / {} pow swap3 abs {} + / swap dup * dup {} + / * * {} * + ?'
        shrp = vs.core.std.Expr([tmp, c2], [expr.format(xy, nl_sharp_mod, 1/nl_sharp_exp, high_damp, low_damp, sharp_str*i)])

        if equal_str >= 0.01:
            diff = vs.core.std.MakeDiff(shrp, tmp)
            if equal_str != 1:
                expr = 'x {} *'.format(equal_str) if isFLOAT else 'x {} - {} * {} +'.format(mid, equal_str, mid)
                diff = vs.core.std.Expr([diff], [expr])
            diff = vs.core.std.Convolution(diff, matrix=mat1) if mode > 0 else vs.core.std.Convolution(diff, matrix=mat2)
            shrp = vs.core.std.MakeDiff(shrp, diff)

    if xsharp_str >= 0.01:
        xyshrp = vs.core.std.Expr([shrp, vs.core.std.Convolution(shrp, matrix=mat2)], ['x dup y - 9.69 * +'])
        rpshrp = R(xyshrp, shrp, [rep])
        shrp = vs.core.std.Merge(shrp, rpshrp, [xsharp_str])

    return vs.core.std.ShufflePlanes([shrp, clip], [0, 1, 2], color) if color in [vs.YUV] else shrp

def retinex_deband(clip:vs.VideoNode, preset:str="high/nograin",
    f3kdb_kwargs:dict={}, method:str="p", thr:Optional[int]=None) -> vs.VideoNode:

    # https://f3kdb.readthedocs.io/en/stable/usage.html

    # presets
    # depth 	y=0/cb=0/cr=0/grainy=0/grainc=0
    # low 	    y=32/cb=32/cr=32/grainy=32/grainc=32
    # medium 	y=48/cb=48/cr=48/grainy=48/grainc=48
    # high 	    y=64/cb=64/cr=64/grainy=64/grainc=64
    # veryhigh 	y=80/cb=80/cr=80/grainy=80/grainc=80
    # nograin 	grainy=0/grainc=0
    # luma 	    cb=0/cr=0/grainc=0
    # chroma 	y=0/grainy=0

    mask = util.retinex_edge(clip, method=method, thr=thr).std.Inflate()
    deband = vs.core.neo_f3kdb.Deband(clip, preset=preset, **f3kdb_kwargs)
    return vs.core.std.MaskedMerge(clipa=deband, clipb=clip, mask=mask)
