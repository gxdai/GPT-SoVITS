import os
import traceback,gradio as gr
import sys

sys.path.append(os.getcwd())
import logging
from tools.i18n.i18n import I18nAuto
from tools.my_utils import clean_path
i18n = I18nAuto()

logger = logging.getLogger(__name__)
import librosa,ffmpeg
import soundfile as sf
import torch

from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho
from bsroformer import BsRoformer_Loader

try:
    import gradio.analytics as analytics
    analytics.version_check = lambda:None
except:...


weight_uvr5_root = "tools/uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or name.endswith(".ckpt") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", "").replace(".ckpt", ""))


availabel_models = uvr5_names.copy()
os.environ['TEMP'] = '/home/guoxian/workspace/GPT-SoVITS/TEMP'
# hard-code the following parameters
device="cuda"
is_half=False
webui_port_uvr5=9873
is_share=False

def html_left(text, label='p'):
    return f"""<div style="text-align: left; margin: 0; padding: 0;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""

def html_center(text, label='p'):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""

def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):

    print("model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0: ", model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0)
    infos = []
    try:
        inp_root = clean_path(inp_root)
        save_root_vocal = clean_path(save_root_vocal)
        save_root_ins = clean_path(save_root_ins)
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        elif model_name == "Bs_Roformer" or "bs_roformer" in model_name.lower():
            func = BsRoformer_Loader
            pre_fun = func(
                model_path = os.path.join(weight_uvr5_root, model_name + ".ckpt"),
                device = device,
                is_half=is_half
            )
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]


        for path in paths:
            # this code is bad
            # inp_path = os.path.join(inp_root, path)
            inp_path = path
            if(os.path.isfile(inp_path)==False):
                print(f"Skip: {inp_path}")
                continue
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(
                    f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UVR5 args')
    parser.add_argument('--model-choose', default='HP5_only_main_vocal', choices=uvr5_names)
    parser.add_argument('--dr-wav-input', default='/home/guoxian/workspace/GPT-SoVITS/elon_musk/output_slices')
    parser.add_argument('--wav-inputs', default=None)
    parser.add_argument("--agg", default=10, help="It should be within 1 and 20")
    parser.add_argument('--opt-vocal-root', default='./output/opt_vocal')
    parser.add_argument('--opt-instr-root', default='./output/opt_instr')
    parser.add_argument('--fmt', default="flac", choices=["wav", "flac", "mp3", "m4a"])
    args = parser.parse_args()
    process = uvr(args.model_choose,
        args.dr_wav_input, args.opt_vocal_root, args.wav_inputs, args.opt_instr_root, args.agg, args.fmt)
    
    for _ in process:
        pass