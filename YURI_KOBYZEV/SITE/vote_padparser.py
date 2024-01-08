import argparse
import os
import glob
import votemodules
from votemodules.padparser import padparser
from votemodules.votemodel import Resnext50ml, Resnet50s


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',nargs='+', type=str, default = './images', help='file/dir')
    parser.add_argument('--jpath', type=str,default='padresult', help='save results to *.json')
    parser.add_argument('--thr', type=float, default=0.5, help='threshold multilable vote')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or cpu')
    opt = parser.parse_args()
#    print_args(vars(opt))
    return opt

def run(
        source='images',
        jpath='result',
        thr=0.3,
        device='cpu',
):
    print('source: ',source,' jpath=',jpath,' thr:',thr,' device:',device)
    print("=================================================================")
    vmodel = Resnext50ml('weights/chpt35-col.pth',device, 3)
    smodel = Resnet50s('weights/vote_bce_resnet50_sign.pt',device, 1)
    vp = padparser(vmodel, smodel)
    if isinstance(source,list): 
        images = source
    else:
        if os.path.isfile(source): 
            images=[source]
        else:
            if os.path.isdir(source): 
                images=glob.glob(source+"/*.jpg")
    for doc_path in images:    
        print("==================>file:>>>> ",doc_path)
        vp.path=doc_path
        img = vp.alignimage(doc_path)
        #img = doc_path
        vp.process_data(img,thr)
        vp.save_json_resut(jpath)


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

