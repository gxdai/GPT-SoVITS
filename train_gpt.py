import json
import os, sys
from subprocess import Popen
import yaml
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share
from tools.my_utils import load_audio, check_for_existance, check_details
version = "v2"
SoVITS_weight_root=["SoVITS_weights_v2","SoVITS_weights"]
GPT_weight_root=["GPT_weights_v2","GPT_weights"]
now_dir = os.getcwd()
sys.path.insert(0, now_dir)

tmp = os.path.join(now_dir, "TEMP")


def fix_gpu_numbers(inputs):
    output=[]
    try:
        for input in inputs.split(","):output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


p_train_GPT=None
def open1Bb(batch_size,total_epoch,exp_name,if_dpo,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1, exp_root):
    print(batch_size,total_epoch,exp_name,if_dpo,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1)
    global p_train_GPT
    # exp_root = cmd.exp_root
    if(p_train_GPT==None):
        with open("GPT_SoVITS/configs/s1longer.yaml"if version=="v1"else "GPT_SoVITS/configs/s1longer-v2.yaml")as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
        if check_for_existance([s1_dir],is_train=True):
            check_details([s1_dir],is_train=True)
        if(is_half==False):
            data["train"]["precision"]="32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["pretrained_s1"]=pretrained_s1
        data["train"]["save_every_n_epoch"]=save_every_epoch
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_dpo"]=if_dpo
        data["train"]["half_weights_save_dir"]=os.path.join(exp_root, GPT_weight_root[-int(version[-1])+2])
        data["train"]["exp_name"]=exp_name
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1"%s1_dir
        # data["version"]=version

        if not os.path.isdir(data["train"]["half_weights_save_dir"]):
            os.makedirs(data["train"]["half_weights_save_dir"])

        os.environ["_CUDA_VISIBLE_DEVICES"]=fix_gpu_numbers(gpu_numbers.replace("-",","))
        os.environ["hz"]="25hz"
        tmp_config_path="%s/tmp_s1.yaml"%exp_root
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" '%(python_exec,tmp_config_path)
        yield "GPT训练开始：%s"%cmd, {"__type__":"update","visible":False}, {"__type__":"update","visible":True}
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT=None
        yield "GPT训练完成", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}
    else:
        yield "已有正在进行的GPT训练任务，需先终止才能开启下一次任务", {"__type__":"update","visible":False}, {"__type__":"update","visible":True}


if __name__ == "__main__":
    import argparse
  
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-root", type=str, default="xxx", required=True,)
    cmd = parser.parse_args()
    print(cmd)
  
    batch_size = 4
    total_epoch = 15
    exp_name = "xxx"
    if_dpo = False
    if_save_latest = True
    if_save_every_weights = True
    save_every_epoch = 5
    gpu_numbers = "0-1"
    pretrained_s1 = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    exp_root = cmd.exp_root
    prs = open1Bb(batch_size,total_epoch,exp_name,if_dpo,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1, exp_root)
    for _ in prs:
        pass